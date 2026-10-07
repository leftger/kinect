//! Kinect v2 transport for macOS, via libusb.
//!
//! The Linux build talks to the sensor through `nusb`. That crate's isochronous
//! support exists only on Linux, and on macOS `nusb` opens the device
//! exclusively, so a second library cannot claim the depth interface beside it.
//! This host owns the whole device: control, the colour bulk pipe, and the
//! isochronous depth pipe.
//!
//! Depth transfers follow libfreenect2's pool. Several isochronous transfers
//! stay in flight and are resubmitted as they complete; macOS drops the stream
//! if the host stops asking. Packet counts are the macOS values in
//! [`crate::settings::PacketParams`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use rusb::constants::{LIBUSB_TRANSFER_CANCELLED, LIBUSB_TRANSFER_COMPLETED};
use rusb::ffi::{
    libusb_alloc_transfer, libusb_cancel_transfer, libusb_fill_iso_transfer, libusb_free_transfer,
    libusb_get_max_iso_packet_size, libusb_set_iso_packet_lengths, libusb_submit_transfer,
    libusb_transfer,
};
use rusb::{
    request_type, Device, DeviceHandle, Direction, Recipient, RequestType, TransferType, UsbContext,
};

use crate::device::{PRODUCT_ID, PRODUCT_ID_PREVIEW, VENDOR_ID};
use crate::{Error, USB_TIMEOUT};

const IR_INTERFACE: u8 = 1;
const SET_ISOCH_DELAY: u8 = 0x31;
const ISO_TIMEOUT_MS: u32 = 1000;
const SS_ENDPOINT_COMPANION: u8 = 0x30;

/// libusb session for one Kinect.
///
/// Synchronous control and bulk transfers take [`Self::handle`]. The event
/// thread only calls `libusb_handle_events`, and isochronous callbacks only
/// touch the transfer and [`PacketQueue`]. Those locks are never nested.
pub struct LibusbSession {
    context: rusb::Context,
    device: Device<rusb::Context>,
    handle: Mutex<DeviceHandle<rusb::Context>>,
    event_stop: Arc<AtomicBool>,
    event_thread: Mutex<Option<JoinHandle<()>>>,
    iso: Mutex<Option<IsoPool>>,
    queue: Arc<PacketQueue>,
}

struct PacketQueue {
    packets: Mutex<VecDeque<Vec<u8>>>,
    ready: Condvar,
}

impl PacketQueue {
    fn push_all(&self, packets: impl IntoIterator<Item = Vec<u8>>) {
        let mut queue = self.packets.lock().unwrap();
        let before = queue.len();
        queue.extend(packets);
        if queue.len() != before {
            self.ready.notify_one();
        }
    }

    /// Packets completed since the last drain. Waits up to `timeout` when none
    /// are queued yet.
    fn drain(&self, timeout: Duration) -> Vec<Vec<u8>> {
        let mut queue = self.packets.lock().unwrap();
        if queue.is_empty() {
            let (guard, _) = self.ready.wait_timeout(queue, timeout).unwrap();
            queue = guard;
        }
        queue.drain(..).collect()
    }
}

struct IsoPool {
    slots: Vec<Box<IsoSlot>>,
    enabled: Arc<AtomicBool>,
}

struct IsoSlot {
    buffer: Vec<u8>,
    transfer: *mut libusb_transfer,
    stopped: AtomicBool,
    enabled: Arc<AtomicBool>,
    queue: Arc<PacketQueue>,
}

// The transfer pointer is only cancelled from `shutdown` and completed on the
// libusb event thread. `shutdown` frees it after every slot has stopped.
unsafe impl Send for IsoPool {}
unsafe impl Sync for IsoPool {}

impl LibusbSession {
    pub async fn open(info: &nusb::DeviceInfo, reset: bool) -> Result<Arc<Self>, Error> {
        let serial = info.serial_number().map(str::to_string);
        spawn_blocking(move || Self::open_blocking(serial.as_deref(), reset)).await
    }

    fn open_blocking(serial: Option<&str>, reset: bool) -> Result<Arc<Self>, Error> {
        let context = rusb::Context::new()?;
        let (device, handle) = find_device(&context, serial)?;

        if reset {
            handle.reset()?;
            drop(handle);
            std::thread::sleep(Duration::from_millis(1500));
            let (device, handle) = find_device(&context, serial)?;
            return Self::start(context, device, handle);
        }

        Self::start(context, device, handle)
    }

    fn start(
        context: rusb::Context,
        device: Device<rusb::Context>,
        handle: DeviceHandle<rusb::Context>,
    ) -> Result<Arc<Self>, Error> {
        if handle.active_configuration()? != 1 {
            handle.set_active_configuration(1)?;
        }
        handle.claim_interface(0)?;
        handle.claim_interface(IR_INTERFACE)?;

        // Same delay the Linux open path programs before the depth pipe starts.
        handle.write_control(
            request_type(Direction::Out, RequestType::Standard, Recipient::Device),
            SET_ISOCH_DELAY,
            40,
            0,
            &[],
            USB_TIMEOUT,
        )?;

        let event_stop = Arc::new(AtomicBool::new(false));
        let thread_context = context.clone();
        let thread_stop = Arc::clone(&event_stop);
        let event_thread = std::thread::Builder::new()
            .name("kinect-libusb".into())
            .spawn(move || {
                while !thread_stop.load(Ordering::Relaxed) {
                    let _ = thread_context.handle_events(Some(Duration::from_millis(50)));
                }
            })
            .map_err(|error| Error::Processing(error.into()))?;

        eprintln!("[kinect] opened the sensor with libusb");

        Ok(Arc::new(Self {
            context,
            device,
            handle: Mutex::new(handle),
            event_stop,
            event_thread: Mutex::new(Some(event_thread)),
            iso: Mutex::new(None),
            queue: Arc::new(PacketQueue {
                packets: Mutex::new(VecDeque::new()),
                ready: Condvar::new(),
            }),
        }))
    }

    pub async fn control_out(
        self: &Arc<Self>,
        recipient: Recipient,
        request: u8,
        value: u16,
        index: u16,
        data: &[u8],
    ) -> Result<(), Error> {
        let data = data.to_vec();
        let usb = Arc::clone(self);
        spawn_blocking(move || usb.control_out_blocking(recipient, request, value, index, &data))
            .await
    }

    pub async fn bulk_write(self: &Arc<Self>, endpoint: u8, data: &[u8]) -> Result<(), Error> {
        let data = data.to_vec();
        let usb = Arc::clone(self);
        spawn_blocking(move || usb.bulk_write_blocking(endpoint, &data)).await
    }

    pub async fn bulk_read(self: &Arc<Self>, endpoint: u8, buf: &mut [u8]) -> Result<usize, Error> {
        let mut owned = vec![0; buf.len()];
        let usb = Arc::clone(self);
        let (length, owned) = spawn_blocking(move || {
            let length = usb.bulk_read_blocking(endpoint, &mut owned)?;
            Ok((length, owned))
        })
        .await?;
        buf[..length].copy_from_slice(&owned[..length]);
        Ok(length)
    }

    /// Start or stop the depth pipe. While it is started, completed isochronous
    /// packets accumulate for [`Self::drain_depth`].
    pub async fn set_depth_enabled(
        self: &Arc<Self>,
        enabled: bool,
        endpoint: u8,
        num_transfers: usize,
        num_packets: usize,
        packet_size: usize,
    ) -> Result<(), Error> {
        let usb = Arc::clone(self);
        spawn_blocking(move || {
            usb.set_depth_blocking(enabled, endpoint, num_transfers, num_packets, packet_size)
        })
        .await
    }

    pub async fn drain_depth(self: &Arc<Self>) -> Result<Vec<Vec<u8>>, Error> {
        let usb = Arc::clone(self);
        spawn_blocking(move || Ok(usb.queue.drain(USB_TIMEOUT))).await
    }

    fn control_out_blocking(
        &self,
        recipient: Recipient,
        request: u8,
        value: u16,
        index: u16,
        data: &[u8],
    ) -> Result<(), Error> {
        let handle = self.handle.lock().unwrap();
        handle.write_control(
            request_type(Direction::Out, RequestType::Standard, recipient),
            request,
            value,
            index,
            data,
            USB_TIMEOUT,
        )?;
        Ok(())
    }

    fn bulk_write_blocking(&self, endpoint: u8, mut data: &[u8]) -> Result<(), Error> {
        let handle = self.handle.lock().unwrap();
        while !data.is_empty() {
            let wrote = handle.write_bulk(endpoint, data, USB_TIMEOUT)?;
            if wrote == 0 {
                return Err(Error::Processing("bulk write transferred no data".into()));
            }
            data = &data[wrote..];
        }
        Ok(())
    }

    fn bulk_read_blocking(&self, endpoint: u8, buf: &mut [u8]) -> Result<usize, Error> {
        let handle = self.handle.lock().unwrap();
        Ok(handle.read_bulk(endpoint, buf, USB_TIMEOUT)?)
    }

    fn set_depth_blocking(
        &self,
        enabled: bool,
        endpoint: u8,
        num_transfers: usize,
        num_packets: usize,
        packet_size: usize,
    ) -> Result<(), Error> {
        if !enabled {
            if let Some(pool) = self.iso.lock().unwrap().take() {
                pool.shutdown();
            }
            self.handle
                .lock()
                .unwrap()
                .set_alternate_setting(IR_INTERFACE, 0)?;
            return Ok(());
        }

        if let Some(pool) = self.iso.lock().unwrap().take() {
            pool.shutdown();
        }
        self.handle
            .lock()
            .unwrap()
            .set_alternate_setting(IR_INTERFACE, 1)?;
        let raw = self.handle.lock().unwrap().as_raw();
        let pool = IsoPool::start(
            raw,
            endpoint,
            num_transfers,
            num_packets,
            packet_size,
            Arc::clone(&self.queue),
        )?;
        *self.iso.lock().unwrap() = Some(pool);
        Ok(())
    }

    pub fn max_iso_packet_size(
        &self,
        configuration: u8,
        alternate: u8,
        endpoint: u8,
    ) -> Option<u16> {
        let configs = self.device.device_descriptor().ok()?.num_configurations();
        for index in 0..configs {
            let config = self.device.config_descriptor(index).ok()?;
            if config.number() != configuration {
                continue;
            }
            for interface in config.interfaces() {
                for descriptor in interface.descriptors() {
                    if descriptor.setting_number() != alternate {
                        continue;
                    }
                    for endpoint_descriptor in descriptor.endpoint_descriptors() {
                        if endpoint_descriptor.address() != endpoint
                            || endpoint_descriptor.transfer_type() != TransferType::Isochronous
                        {
                            continue;
                        }
                        if let Some(extra) = endpoint_descriptor.extra() {
                            if let Some(size) = companion_packet_size(extra) {
                                return Some(size);
                            }
                        }
                    }
                }
            }
        }

        let size = unsafe { libusb_get_max_iso_packet_size(self.device.as_raw(), endpoint) };
        (size >= 0x8400).then_some(size as u16)
    }
}

impl Drop for LibusbSession {
    fn drop(&mut self) {
        if let Some(pool) = self.iso.lock().unwrap().take() {
            pool.shutdown();
        }
        self.event_stop.store(true, Ordering::Relaxed);
        self.context.interrupt_handle_events();
        if let Some(thread) = self.event_thread.lock().unwrap().take() {
            let _ = thread.join();
        }
    }
}

impl IsoPool {
    fn start(
        handle: *mut rusb::ffi::libusb_device_handle,
        endpoint: u8,
        num_transfers: usize,
        num_packets: usize,
        packet_size: usize,
        queue: Arc<PacketQueue>,
    ) -> Result<Self, Error> {
        let enabled = Arc::new(AtomicBool::new(true));
        let mut slots = Vec::with_capacity(num_transfers);
        let bytes = num_packets
            .checked_mul(packet_size)
            .ok_or_else(|| Error::Processing("isochronous buffer size overflowed".into()))?;

        for _ in 0..num_transfers {
            let transfer = unsafe { libusb_alloc_transfer(num_packets as i32) };
            if transfer.is_null() {
                abandon(slots, enabled);
                return Err(Error::Processing("libusb_alloc_transfer failed".into()));
            }
            slots.push(Box::new(IsoSlot {
                buffer: vec![0; bytes],
                transfer,
                stopped: AtomicBool::new(false),
                enabled: Arc::clone(&enabled),
                queue: Arc::clone(&queue),
            }));
        }

        for slot in &mut slots {
            let transfer = slot.transfer;
            let user = slot.as_mut() as *mut IsoSlot;
            let result = unsafe {
                let buffer = (*user).buffer.as_mut_ptr();
                libusb_fill_iso_transfer(
                    transfer,
                    handle,
                    endpoint,
                    buffer,
                    bytes as i32,
                    num_packets as i32,
                    on_iso,
                    user.cast(),
                    ISO_TIMEOUT_MS,
                );
                libusb_set_iso_packet_lengths(transfer, packet_size as u32);
                libusb_result(libusb_submit_transfer(transfer))
            };
            if let Err(error) = result {
                abandon(slots, enabled);
                return Err(error.into());
            }
        }

        Ok(Self { slots, enabled })
    }

    fn shutdown(mut self) {
        self.enabled.store(false, Ordering::Release);
        for slot in &self.slots {
            let result = unsafe { libusb_cancel_transfer(slot.transfer) };
            if result < 0 {
                slot.stopped.store(true, Ordering::Release);
            }
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while self
            .slots
            .iter()
            .any(|slot| !slot.stopped.load(Ordering::Acquire))
        {
            if std::time::Instant::now() > deadline {
                // A transfer is still live. Freeing it would race the callback.
                std::mem::forget(self);
                return;
            }
            std::thread::sleep(Duration::from_millis(10));
        }

        for slot in &mut self.slots {
            unsafe { libusb_free_transfer(slot.transfer) };
            slot.transfer = std::ptr::null_mut();
        }
    }
}

fn abandon(slots: Vec<Box<IsoSlot>>, enabled: Arc<AtomicBool>) {
    if slots.is_empty() {
        return;
    }
    IsoPool { slots, enabled }.shutdown();
}

extern "system" fn on_iso(transfer: *mut libusb_transfer) {
    unsafe {
        let slot = &*((*transfer).user_data as *const IsoSlot);
        if (*transfer).status == LIBUSB_TRANSFER_CANCELLED {
            slot.stopped.store(true, Ordering::Release);
            return;
        }

        let mut packets = Vec::new();
        let mut offset = 0usize;
        for index in 0..(*transfer).num_iso_packets {
            let descriptor = &*(*transfer).iso_packet_desc.as_ptr().add(index as usize);
            if descriptor.status == LIBUSB_TRANSFER_COMPLETED {
                let length = descriptor.actual_length as usize;
                let end = offset + length;
                if end <= slot.buffer.len() {
                    packets.push(slot.buffer[offset..end].to_vec());
                }
            }
            offset += descriptor.length as usize;
        }
        slot.queue.push_all(packets);

        if slot.enabled.load(Ordering::Acquire) {
            if libusb_submit_transfer(transfer) < 0 {
                slot.stopped.store(true, Ordering::Release);
            }
        } else {
            slot.stopped.store(true, Ordering::Release);
        }
    }
}

/// `wBytesPerInterval` from a SuperSpeed endpoint companion descriptor.
fn companion_packet_size(extra: &[u8]) -> Option<u16> {
    let mut offset = 0;
    while offset + 1 < extra.len() {
        let length = extra[offset] as usize;
        if length < 2 || offset + length > extra.len() {
            break;
        }
        if extra[offset + 1] == SS_ENDPOINT_COMPANION && length >= 6 {
            return Some(u16::from_le_bytes([extra[offset + 4], extra[offset + 5]]));
        }
        offset += length;
    }
    None
}

fn find_device(
    context: &rusb::Context,
    serial: Option<&str>,
) -> Result<(Device<rusb::Context>, DeviceHandle<rusb::Context>), Error> {
    for device in context.devices()?.iter() {
        let descriptor = match device.device_descriptor() {
            Ok(descriptor) => descriptor,
            Err(_) => continue,
        };
        if descriptor.vendor_id() != VENDOR_ID
            || (descriptor.product_id() != PRODUCT_ID
                && descriptor.product_id() != PRODUCT_ID_PREVIEW)
        {
            continue;
        }

        let handle = device.open()?;
        if let Some(wanted) = serial {
            let Some(index) = descriptor.serial_number_string_index() else {
                continue;
            };
            let got = handle
                .read_string_descriptor_ascii(index)
                .unwrap_or_default();
            if got != wanted {
                continue;
            }
        }
        return Ok((device, handle));
    }

    Err(Error::NoDevice)
}

fn libusb_result(result: i32) -> Result<(), rusb::Error> {
    if result >= 0 {
        return Ok(());
    }
    Err(match result {
        -1 => rusb::Error::Io,
        -2 => rusb::Error::InvalidParam,
        -3 => rusb::Error::Access,
        -4 => rusb::Error::NoDevice,
        -5 => rusb::Error::NotFound,
        -6 => rusb::Error::Busy,
        -7 => rusb::Error::Timeout,
        -8 => rusb::Error::Overflow,
        -9 => rusb::Error::Pipe,
        -10 => rusb::Error::Interrupted,
        -11 => rusb::Error::NoMem,
        -12 => rusb::Error::NotSupported,
        _ => rusb::Error::Other,
    })
}

async fn spawn_blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, Error> + Send + 'static,
) -> Result<T, Error> {
    // `Error` is not `Send` — its `Processing` variant boxes a trait object —
    // so the blocking thread stringifies it before the value crosses threads.
    let work = move || work().map_err(|error| error.to_string());
    match tokio::task::spawn_blocking(work).await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(Error::Processing(error.into())),
        Err(error) => Err(Error::Processing(error.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::companion_packet_size;

    #[test]
    fn reads_superspeed_companion_packet_size() {
        let extra = [6, 0x30, 0, 0, 0x00, 0x84];
        assert_eq!(companion_packet_size(&extra), Some(0x8400));
    }
}
