//! `scan` — handheld 3D scanner for the Kinect v2.
//!
//! Depth odometry tracks the sensor frame to frame, and every frame whose pose is
//! trusted is fused into a voxel-hashed TSDF. The zero crossing of that field is
//! extracted as a triangle mesh.
//!
//! ```text
//! scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--no-color]
//! scan record  --out capture.k2df [--frames N] [--no-filter] [--no-color]
//! scan replay  --in capture.k2df [--out mesh.ply] [--voxel M]
//! ```
//!
//! `record` and `replay` exist so that odometry can be tuned against
//! byte-identical input instead of a live sensor: the scene changes between live
//! runs, so two parameter sets never see the same data.

mod capture;
mod loop_closure;
mod odometry;
mod recording;
mod scanner;
#[cfg(feature = "wgpu-decode")]
mod wgpu_depth;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use geom::tsdf::TsdfParams;
use loop_closure::LoopClosureConfig;
use scanner::{ClosureReport, FrameReport, Scanner, ScannerConfig};

const DEFAULT_FRAMES: usize = 60;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("scan: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    frames: Option<usize>,
    out: Option<PathBuf>,
    input: Option<PathBuf>,
    tsdf: TsdfParams,
    filters: bool,
    drain_color: bool,
    /// Decode depth frames on the GPU. Needs a build with the `gpu-decode` feature.
    gpu: bool,
    /// Detect revisits and redistribute the accumulated error over the whole
    /// trajectory. Costs memory: the frames have to be kept so the model can be
    /// rebuilt once the poses move.
    loop_closure: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            frames: None,
            out: None,
            input: None,
            tsdf: TsdfParams::default(),
            filters: true,
            drain_color: true,
            gpu: false,
            loop_closure: false,
        }
    }
}

async fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();

    let mut options = Options::default();

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--frames" => {
                options.frames = Some(next_value(&mut args, "--frames")?.parse()?);
            }
            "--out" => options.out = Some(PathBuf::from(next_value(&mut args, "--out")?)),
            "--in" => options.input = Some(PathBuf::from(next_value(&mut args, "--in")?)),
            "--voxel" => {
                let voxel: f32 = next_value(&mut args, "--voxel")?.parse()?;
                if !(voxel > 0.0) {
                    return Err("--voxel must be positive".into());
                }
                // Keep the truncation band proportional: ~4 voxels is the usual
                // choice, and scaling it automatically avoids a silently broken
                // field when someone changes the resolution.
                options.tsdf = TsdfParams {
                    voxel_size: voxel,
                    truncation: voxel * 4.0,
                    ..TsdfParams::default()
                };
            }
            "--no-filter" => options.filters = false,
            "--no-color" => options.drain_color = false,
            "--loop-closure" => options.loop_closure = true,
            "--gpu" => options.gpu = true,
            "-h" | "--help" => {
                print_usage();
                return Ok(());
            }
            other => {
                return Err(format!("unrecognised argument `{other}` (try --help)").into());
            }
        }
    }

    match command.as_str() {
        "live" => live(&options).await,
        "record" => record(&options).await,
        "replay" => replay(&options),
        "" | "help" => {
            print_usage();
            Ok(())
        }
        other => Err(format!("unknown command `{other}` (try --help)").into()),
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("{flag} needs a value").into())
}

fn print_usage() {
    println!(
        "scan - handheld 3D scanner for the Kinect v2\n\
         \n\
         USAGE:\n\
         \x20 scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--no-color]\n\
         \x20 scan record  --out capture.k2df [--frames N] [--no-filter] [--no-color]\n\
         \x20 scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N]\n\
         \n\
         COMMANDS:\n\
         \x20 live     Capture from the sensor and reconstruct as you go.\n\
         \x20 record   Capture depth frames to a file, for offline tuning.\n\
         \x20 replay   Reconstruct from a recording. Needs no sensor.\n\
         \n\
         OPTIONS:\n\
         \x20 --frames N    With live/record: frames to capture (default {DEFAULT_FRAMES}).\n\
         \x20               With replay: how many recorded frames to process. A\n\
         \x20               recording is deterministic, so a prefix of it is a\n\
         \x20               repeatable test case.\n\
         \x20 --out FILE    Output mesh (default scan.ply).\n\
         \x20 --in FILE     Recording to replay.\n\
         \x20 --voxel M     TSDF voxel size in metres (default 0.01). Sets the\n\
         \x20               truncation band to 4x this.\n\
         \x20 --no-filter   Disable the decoder's bilateral/edge filters: roughly\n\
         \x20               doubles frame rate, keeps more junk points.\n\
         \x20 --no-color    Do not drain the colour stream (unused by the scanner).\n\
         \x20 --gpu         Decode depth on the GPU. Needs a build with\n\
         \x20               `--features wgpu-decode` (Vulkan, preferred) or\n\
         \x20               `--features gpu-decode` (OpenCL). OpenCL does not work\n\
         \x20               on this GPU -- Rusticl runs no kernels at all, see\n\
         \x20               examples/ocl_check.rs. The decode is what\n\
         \x20               caps capture rate, so this mainly buys denser frames.\n\
         \x20 --loop-closure  Detect revisits, redistribute the accumulated drift\n\
         \x20               over the whole trajectory, and rebuild the model from\n\
         \x20               the corrected poses. Costs memory: frames are kept\n\
         \x20               (about 850 KB each) so the model can be rebuilt once\n\
         \x20               the poses move. Only helps a scan that returns\n\
         \x20               somewhere it has already been.\n\
         \n\
         A trajectory PLY is written alongside the mesh as <out>.trajectory.ply,\n\
         which is the quickest way to see how badly the pose has drifted."
    );
}

async fn live(options: &Options) -> Result<(), Box<dyn Error>> {
    let frames = options.frames.unwrap_or(DEFAULT_FRAMES);

    let mut capture =
        capture::Capture::open(options.filters, options.drain_color, options.gpu).await?;
    let intrinsics = capture.intrinsics();
    let (width, height) = capture.dimensions();

    println!(
        "[scan] depth intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    );
    println!("[scan] capturing {frames} frames at {width}x{height} - move the sensor slowly");

    let mut scanner = Scanner::new(intrinsics, scanner_config(options));

    for index in 1..=frames {
        let frame = capture.next_frame().await?;
        let report = scanner.add_frame(frame, width, height);
        print_progress(index, &report);
    }

    capture.stop().await?;
    close_loops(&mut scanner, options);
    finish(&scanner, options)
}

async fn record(options: &Options) -> Result<(), Box<dyn Error>> {
    let frames = options.frames.unwrap_or(DEFAULT_FRAMES);
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("capture.k2df"));

    let mut capture =
        capture::Capture::open(options.filters, options.drain_color, options.gpu).await?;
    let intrinsics = capture.intrinsics();
    let (width, height) = capture.dimensions();

    let mut writer = recording::FrameWriter::create(&out, intrinsics, width, height)?;
    println!("[scan] recording {frames} frames to {}", out.display());

    let started = std::time::Instant::now();
    for index in 1..=frames {
        let frame = capture.next_frame().await?;
        // Seconds since capture began, so the recording carries real timing.
        writer.write_frame(started.elapsed().as_secs_f32(), frame)?;

        if index % 10 == 0 || index == frames {
            println!("[scan] recorded {index}/{frames}");
        }
    }

    capture.stop().await?;
    let count = writer.finish()?;

    println!("[scan] wrote {count} frames to {}", out.display());
    Ok(())
}

fn replay(options: &Options) -> Result<(), Box<dyn Error>> {
    let input = options.input.clone().ok_or("replay needs --in <file>")?;
    let recording = recording::read(&input)?;

    let duration = recording
        .frames
        .last()
        .map_or(0.0, |frame| frame.timestamp);

    println!(
        "[scan] {} frames over {:.1} s of {}x{} from {}",
        recording.frames.len(),
        duration,
        recording.width,
        recording.height,
        input.display()
    );
    println!(
        "[scan] depth intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        recording.intrinsics.fx,
        recording.intrinsics.fy,
        recording.intrinsics.cx,
        recording.intrinsics.cy
    );

    let mut scanner = Scanner::new(recording.intrinsics, scanner_config(options));

    // `--frames` limits how much of a recording is replayed. A recording is
    // deterministic, so a prefix of it is a valid and repeatable test case --
    // parameter tuning does not need the full capture every time.
    let limit = options.frames.unwrap_or(usize::MAX);

    for (index, frame) in recording.frames.iter().take(limit).enumerate() {
        let report = scanner.add_frame(&frame.depth, recording.width, recording.height);
        print_progress(index + 1, &report);
    }

    close_loops(&mut scanner, options);
    finish(&scanner, options)
}

fn scanner_config(options: &Options) -> ScannerConfig {
    ScannerConfig {
        tsdf: options.tsdf,
        loop_closure: options
            .loop_closure
            .then(LoopClosureConfig::default),
        ..ScannerConfig::default()
    }
}

/// Run loop closure and report it. Called between the last frame and the
/// summary, because it can change the model `finish` is about to write out.
fn close_loops(scanner: &mut Scanner, options: &Options) {
    if !options.loop_closure {
        return;
    }

    let report = scanner.close_loops();

    if report.loop_edges == 0 {
        println!(
            "loop closure: no revisits verified over {} frames, trajectory unchanged",
            report.nodes
        );
        return;
    }

    println!(
        "loop closure: {} revisits over {} frames, cost {:.3} -> {:.3} in {} iterations",
        report.loop_edges,
        report.nodes,
        report.cost_before,
        report.cost_after,
        report.iterations
    );
    println!(
        "  largest pose change {:.1} cm, {:.1} degrees",
        report.max_correction * 100.0,
        report.max_rotation.to_degrees()
    );

    if report.refused {
        println!(
            "  REFUSED: beyond the {:.0} m / {:.0} degree limit, so the model was \
             left as odometry built it",
            scanner_limits_metres(),
            scanner_limits_degrees()
        );
    } else if report.rebuilt {
        println!("  model rebuilt from the corrected poses");
    } else {
        println!("  nothing to apply");
    }
}

fn scanner_limits_metres() -> f32 {
    scanner::MAX_CORRECTION_METRES
}

fn scanner_limits_degrees() -> f32 {
    scanner::MAX_CORRECTION_RADIANS.to_degrees()
}

fn print_progress(index: usize, report: &FrameReport) {
    let position = report.track.pose.translation.vector;

    let outcome = match report.integration {
        Some(stats) => format!(
            "fused {:>6} blocks {:>6.1} MB  (+{:<6} voxels)",
            report.blocks,
            report.allocated_bytes as f64 / 1e6,
            stats.updated_voxels,
        ),
        None => match report.track.rejection {
            Some(reason) => format!("SKIPPED ({reason})"),
            None => "SKIPPED".to_string(),
        },
    };

    // `move` is the recovered sensor motion for this frame and `iters` the total
    // ICP iterations spent on it: together they are the quickest way to spot a
    // jump that the acceptance gates should have caught, or a solve that is
    // burning its whole budget without settling.
    println!(
        "[scan] {:>4}  pts {:>6}  inliers {:>5.1}%  rmse {:>5.1} mm  move {:>5.1} cm  iters {:>3}  pos [{:>6.3} {:>6.3} {:>6.3}]  {outcome}",
        index,
        report.points,
        report.track.inlier_ratio * 100.0,
        report.track.rmse * 1000.0,
        report.track.translation * 100.0,
        report.track.iterations,
        position.x,
        position.y,
        position.z,
    );
}

fn finish(scanner: &Scanner, options: &Options) -> Result<(), Box<dyn Error>> {
    println!();
    println!(
        "[scan] frames: {} (fused {}, skipped {})",
        scanner.frames(),
        scanner.fused(),
        scanner.rejected()
    );

    let frames = scanner.frames().max(1) as f64;
    println!(
        "[scan] time per frame: tracking {:.0} ms, fusion {:.0} ms",
        scanner.total_tracking().as_secs_f64() * 1000.0 / frames,
        scanner.total_fusion().as_secs_f64() * 1000.0 / frames,
    );
    println!(
        "[scan] TSDF: {} blocks, {:.1} MB",
        scanner.block_count(),
        scanner.allocated_bytes() as f64 / 1e6
    );

    // Final position is the quickest sanity check on drift: for a loop that
    // returns to its start, this should be back near the origin.
    let final_position = scanner.pose().translation.vector;
    println!(
        "[scan] final position [{:.3} {:.3} {:.3}] m, {:.3} m from origin",
        final_position.x,
        final_position.y,
        final_position.z,
        final_position.norm()
    );

    let out = options
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("scan.ply"));

    println!("[scan] extracting surface ...");
    let mesh = scanner.mesh();

    if mesh.is_empty() {
        println!("[scan] no surface extracted - was anything in range?");
        return Ok(());
    }

    if let Some((min, max)) = mesh.bounds() {
        println!(
            "[scan] mesh: {} vertices, {} triangles, extent {:.2} x {:.2} x {:.2} m",
            mesh.vertex_count(),
            mesh.triangle_count(),
            max.x - min.x,
            max.y - min.y,
            max.z - min.z
        );
    }

    mesh.save_ply(&out)?;
    println!("[scan] wrote {}", out.display());

    let trajectory = out.with_extension("trajectory.ply");
    geom::mesh::save_points_ply(&trajectory, scanner.trajectory())?;
    println!(
        "[scan] wrote {} ({} camera positions)",
        trajectory.display(),
        scanner.trajectory().len()
    );

    Ok(())
}
