use std::fmt::{self, Debug};

use crate::Error;

use super::{Device, DeviceId, DeviceInfo, Opened};

#[derive(Clone)]
pub struct Closed {
    pub device_info: nusb::DeviceInfo,
}

impl Device<Closed> {
    /// Open the device.
    pub async fn open(self, reset: bool) -> Result<Device<Opened>, Error> {
        // On macOS the libusb host resets the device itself. Doing it here
        // would open the sensor through nusb, which seizes it exclusively.
        #[cfg(not(target_os = "macos"))]
        if reset {
            self.inner.device_info.open().await?.reset().await?;
        }

        Ok(Device {
            inner: Opened::new(self.inner.device_info, reset).await?,
        })
    }
}

impl DeviceInfo for Device<Closed> {
    fn id(&self) -> DeviceId {
        super::id_of(&self.inner.device_info)
    }
}

impl Debug for Device<Closed> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.device_info.fmt(f)
    }
}

impl From<nusb::DeviceInfo> for Device<Closed> {
    fn from(device_info: nusb::DeviceInfo) -> Self {
        Device {
            inner: Closed { device_info },
        }
    }
}
