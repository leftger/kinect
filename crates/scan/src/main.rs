//! `scan` — handheld 3D scanner for the Kinect v2.
//!
//! Depth odometry tracks the sensor frame to frame, and every frame whose pose is
//! trusted is fused into a voxel-hashed TSDF. The zero crossing of that field is
//! extracted as a triangle mesh.
//!
//! ```text
//! scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color]
//!             [--color-mode best|blend|average] [--color-depth-tolerance M]
//!             [--gpu] [--loop-closure] [--viewer] [--dataset DIR]
//! scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu]
//! scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N] [--loop-closure]
//! ```
//!
//! `record` and `replay` exist so that odometry can be tuned against
//! byte-identical input instead of a live sensor: the scene changes between live
//! runs, so two parameter sets never see the same data. A `.k2df` file stores
//! depth only, so `--color` paints a live scan and does nothing on replay.
//!
//! `--dataset` turns a colour scan into something a Gaussian-splatting trainer
//! can read; see [`dataset`]. It needs `live --color`, for the same reason.

mod capture;
mod dataset;
mod loop_closure;
mod odometry;
mod recording;
mod scanner;
mod synthetic;
#[cfg(feature = "viewer")]
mod viewer;
#[cfg(feature = "wgpu-decode")]
mod wgpu_depth;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use geom::coloring::{ColorMode, ColoringParams, ColoringReport};
use geom::export::{self, MeshFormat};
use geom::texturing::TexturingReport;
use geom::tsdf::TsdfParams;
use loop_closure::LoopClosureConfig;
use nalgebra::Vector3;
use odometry::OdometryConfig;
use scanner::{ExportMesh, FrameColor, FrameReport, Scanner, ScannerConfig};

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

#[derive(Clone, Debug)]
struct Options {
    frames: Option<usize>,
    out: Option<PathBuf>,
    input: Option<PathBuf>,
    tsdf: TsdfParams,
    filters: bool,
    /// Read and discard the colour stream. Off by default: the scanner is
    /// depth-only, and waiting for a colour packet costs more than half the frame
    /// time, because that stream delivers at roughly a third of the depth rate.
    drain_color: bool,
    /// Decode depth frames on the GPU. Needs a build with the `gpu-decode` feature.
    gpu: bool,
    /// Detect revisits and redistribute the accumulated error over the whole
    /// trajectory. Costs memory: each depth frame is kept (about 850 KB) so
    /// the model can be rebuilt, and each colour view is about 1.66 MiB
    /// (about 1 GiB at 600 views).
    loop_closure: bool,
    /// Capture colour and paint it onto the mesh.
    color: bool,
    /// Per-vertex PLY colour. `best` unless `--color-mode` says otherwise.
    /// `.obj`, `.gltf`, and `.glb` still assign each triangle to one best
    /// source view.
    color_mode: ColorMode,
    /// How far a vertex may disagree with a view's depth and still be painted,
    /// in metres. Default 0.02.
    color_depth_tolerance: f32,
    /// Open a live window showing the scan as it builds.
    viewer: bool,
    /// Un-mirror the reconstruction horizontally so real-world left and right match.
    /// True by default: the Kinect v2 sensor reads out mirrored frames.
    unmirror: bool,
    /// Track each frame against a render of the fused model instead of against
    /// the previous frame, so tracking error stops accumulating.
    ///
    /// Costs a raycast per frame, which is why it is not the default: at sensor
    /// resolution that is around a second per frame on this hardware. Live
    /// capture becomes a slideshow, so the useful place for it is `replay`,
    /// where wall clock does not matter and the poses it produces are the ones
    /// a scan is ultimately graded on.
    frame_to_model: bool,
    /// Export a Nerfstudio dataset for a Gaussian-splatting trainer: the colour
    /// views, their masks, `transforms.json`, and an initial point cloud.
    /// Needs `--color`, because the dataset is built from the colour views.
    /// The dataset always stays in the sensor frame; the mirror does not apply.
    dataset: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            frames: None,
            out: None,
            input: None,
            tsdf: TsdfParams::default(),
            filters: true,
            drain_color: false,
            gpu: false,
            loop_closure: false,
            color: false,
            color_mode: ColoringParams::default().mode,
            color_depth_tolerance: ColoringParams::default().depth_tolerance,
            viewer: false,
            unmirror: true,
            frame_to_model: false,
            dataset: None,
        }
    }
}

#[derive(Debug)]
enum Command {
    Live,
    Record,
    Replay,
    /// Build a synthetic coloured capture and write it as a dataset.
    Synth,
    Help,
}

async fn run() -> Result<(), Box<dyn Error>> {
    let (command, options) = parse_args(std::env::args().skip(1))?;
    match command {
        Command::Live => live(&options).await,
        Command::Record => record(&options).await,
        Command::Replay => replay(&options),
        Command::Synth => synth(&options),
        Command::Help => {
            print_usage();
            Ok(())
        }
    }
}

fn parse_args(
    args: impl IntoIterator<Item = String>,
) -> Result<(Command, Options), Box<dyn Error>> {
    let mut args = args.into_iter();
    let command = args.next().unwrap_or_default();

    if command == "-h" || command == "--help" || command.is_empty() || command == "help" {
        return Ok((Command::Help, Options::default()));
    }

    let command = match command.as_str() {
        "live" => Command::Live,
        "record" => Command::Record,
        "replay" => Command::Replay,
        "synth" => Command::Synth,
        other => return Err(format!("unknown command `{other}` (try --help)").into()),
    };

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
            "--color" => options.color = true,
            "--color-mode" => {
                options.color_mode = parse_color_mode(&next_value(&mut args, "--color-mode")?)?;
            }
            "--color-depth-tolerance" => {
                let tolerance: f32 = next_value(&mut args, "--color-depth-tolerance")?.parse()?;
                if !(tolerance.is_finite() && tolerance > 0.0) {
                    return Err(
                        "--color-depth-tolerance must be a finite number greater than 0".into(),
                    );
                }
                options.color_depth_tolerance = tolerance;
            }
            "--drain-color" => options.drain_color = true,
            "--viewer" => options.viewer = true,
            "--loop-closure" => options.loop_closure = true,
            "--gpu" => options.gpu = true,
            "--mirror" => options.unmirror = false,
            "--no-mirror" | "--unmirror" => options.unmirror = true,
            "--frame-to-model" => options.frame_to_model = true,
            "--dataset" => {
                options.dataset = Some(PathBuf::from(next_value(&mut args, "--dataset")?))
            }
            "-h" | "--help" => return Ok((Command::Help, options)),
            other => {
                return Err(format!("unrecognised argument `{other}` (try --help)").into());
            }
        }
    }

    // A dataset is built from the colour views, which only a live capture has.
    // Checked here rather than at the end so a mistyped invocation fails before
    // the sensor is opened and a scan is spent.
    if options.dataset.is_some() {
        match &command {
            Command::Live if options.color => {}
            Command::Live => {
                return Err(
                    "--dataset needs --color: the dataset is built from the colour views, \
                     and a depth-only scan has none"
                        .into(),
                )
            }
            Command::Record => {
                return Err(
                    "--dataset cannot be used with `record`: a record run only writes a capture file. \
                     Use `live --color --dataset DIR`, or replay a colour recording with \
                     `scan replay --in FILE --dataset DIR`"
                        .into(),
                )
            }
            // `replay` is allowed with `--dataset`; the recording itself is checked in
            // `replay()` to ensure it carries a colour stream.
            Command::Replay => {}
            // `synth` renders its own colour, so it needs no stream and no flag.
            Command::Synth => {}
            Command::Help => {}
        }
    }

    Ok((command, options))
}

fn parse_color_mode(value: &str) -> Result<ColorMode, Box<dyn Error>> {
    match value {
        "best" => Ok(ColorMode::Best),
        "blend" => Ok(ColorMode::Blend),
        "average" => Ok(ColorMode::Average),
        other => {
            Err(format!("--color-mode must be best, blend, or average (got `{other}`)").into())
        }
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
    println!("{}", usage_text());
}

fn usage_text() -> String {
    format!(
        "scan - handheld 3D scanner for the Kinect v2\n\
         \n\
         USAGE:\n\
         \x20 scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color]\n\
         \x20             [--color-mode best|blend|average] [--color-depth-tolerance M]\n\
         \x20             [--gpu] [--loop-closure] [--viewer] [--dataset DIR]\n\
         \x20 scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu]\n\
         \x20 scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N]\n\
         \x20             [--loop-closure]\n\
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
         \x20 --out FILE    Output mesh (default scan.ply). The extension selects\n\
         \x20               the format: .ply (per-vertex colour), .obj (plus .mtl\n\
         \x20               and .png), .gltf (plus .bin and .png), or .glb (one file).\n\
         \x20 --in FILE     Recording to replay. A .k2df file is depth only.\n\
         \x20 --voxel M     TSDF voxel size in metres (default 0.01). Sets the\n\
         \x20               truncation band to 4x this.\n\
         \x20 --no-filter   Disable the decoder's bilateral/edge filters: roughly\n\
         \x20               doubles frame rate, keeps more junk points.\n\
         \x20 --color       On live: capture colour and paint the mesh. Off by\n\
         \x20               default. The colour stream is slower than depth.\n\
         \x20               Replay cannot paint: the recording has no colour.\n\
         \x20 --color-mode MODE\n\
         \x20               Per-vertex PLY colour: best (default), blend, or\n\
         \x20               average. best keeps the highest-scoring visible\n\
         \x20               sample. blend weights the top three in linear light.\n\
         \x20               average is an equal mean of the stored bytes and\n\
         \x20               skips exposure correction. All three use the same\n\
         \x20               visibility test. An .obj, .gltf, or .glb atlas gives\n\
         \x20               each triangle the one source view that sees it best.\n\
         \x20 --color-depth-tolerance M\n\
         \x20               How far a vertex may disagree with a view's depth, in\n\
         \x20               metres, and still be painted (default 0.02). Must be\n\
         \x20               greater than 0.\n\
         \x20 --drain-color On record: read and discard colour packets. Does not\n\
         \x20               apply to live; use --color there. The packets are not\n\
         \x20               written into the .k2df file.\n\
         \x20 --gpu         Decode depth on the GPU during live and record.\n\
         \x20               Needs `--features wgpu-decode` (Metal on macOS,\n\
         \x20               Vulkan on Linux). OpenCL (`--features gpu-decode`)\n\
         \x20               does not run kernels under Rusticl. Host CPU time\n\
         \x20               drops sharply; a live scan does not finish sooner.\n\
         \x20 --loop-closure  Detect revisits, redistribute the accumulated drift\n\
         \x20               over the whole trajectory, and rebuild the model from\n\
         \x20               the corrected poses. Colour views move onto those\n\
         \x20               poses before the mesh is painted. Costs memory: each\n\
         \x20               depth frame is kept (about 850 KB) so the model can\n\
         \x20               be rebuilt, and each colour view is about 1.66 MiB\n\
         \x20               (about 1 GiB at 600 views). Only helps a scan that\n\
         \x20               returns somewhere it has already been.\n\
         \x20 --viewer      Open a live window. Needs `--features viewer` and a\n\
         \x20               display. Closing the window stops the scan.\n\
         \x20 --frame-to-model\n\
         \x20               Track each frame against a render of the fused model\n\
         \x20               instead of against the previous frame. Frame-to-frame\n\
         \x20               error is added to the next frame, so drift grows with the\n\
         \x20               length of the scan; aligning to the model stops it\n\
         \x20               compounding. Costs a raycast per frame -- roughly a second\n\
         \x20               at sensor resolution on this hardware -- so it is off by\n\
         \x20               default and most useful with `replay`, where wall clock\n\
         \x20               does not matter and the poses are what the scan is graded\n\
         \x20               on.\n\
         \x20 --mirror      Keep the raw sensor mirror orientation instead of flipping\n\
         \x20               X to match real-world coordinates (un-mirrored by default).\n\
         \x20               A dataset is always written in the sensor frame, so this\n\
         \x20               does not apply to --dataset.\n\
         \x20 --dataset DIR With --color on live: also write a dataset a Gaussian-\n\
         \x20               splatting trainer can read. DIR gains images/ (the\n\
         \x20               registered colour, one PNG per kept view), masks/ (white\n\
         \x20               where registration copied a colour sample, so the trainer\n\
         \x20               ignores the holes), transforms.json (depth intrinsics and\n\
         \x20               the pose of every view), and init.ply (the scanned surface,\n\
         \x20               coloured, as the initial Gaussian positions).\n\
         \x20               Images are at depth resolution, 512x424, not the colour\n\
         \x20               sensor's own size: registration copies colour into the\n\
         \x20               depth grid, and that is what keeps the intrinsics and the\n\
         \x20               poses exact. Needs `live`: a .k2df recording has no colour.\n\
         \n\
         A trajectory PLY is written alongside the mesh as <out>.trajectory.ply,\n\
         which is the quickest way to see how badly the pose has drifted."
    )
}

async fn live(options: &Options) -> Result<(), Box<dyn Error>> {
    if options.viewer {
        #[cfg(feature = "viewer")]
        return viewer::run(options);

        #[cfg(not(feature = "viewer"))]
        return Err("this build has no viewer; rebuild with `--features viewer`".into());
    }

    let mut scanner = run_live(options, |_, _| true).await?;
    finish_scan(&mut scanner, options)
}

/// The live capture loop, shared by the plain path and the viewer window.
///
/// `on_frame` runs after each frame is integrated and returns whether to keep
/// going. The plain path ignores it; the viewer uses it to hand the scan thread's
/// latest picture to the GUI, and to stop when the window closes.
pub(crate) async fn run_live<F>(
    options: &Options,
    mut on_frame: F,
) -> Result<Scanner, Box<dyn Error>>
where
    F: FnMut(usize, &Scanner) -> bool,
{
    let frames = options.frames.unwrap_or(DEFAULT_FRAMES);

    let mut capture = capture::Capture::open(options.filters, options.color, options.gpu).await?;
    let intrinsics = capture.intrinsics();
    let (width, height) = capture.dimensions();

    println!(
        "[scan] depth intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    );
    println!("[scan] capturing {frames} frames at {width}x{height} - move the sensor slowly");

    let mut scanner = Scanner::new(intrinsics, scanner_config(options));

    for index in 1..=frames {
        // Owned, because `next_frame` borrows the capture mutably for as long as
        // its result lives, and the colour has to be read afterwards. One 850 KB
        // copy per frame against a 270 ms frame is not worth restructuring the
        // capture API over.
        let frame = capture.next_frame().await?.to_vec();

        // Both come from `capture`, and the scanner needs them together to pair
        // them with one pose.
        let color = capture.color().map(|captured| FrameColor {
            rgb: captured.rgb.as_slice(),
            depth: captured.depth.as_slice(),
            valid: captured.valid.as_slice(),
            exposure: captured.exposure,
            gain: captured.gain,
            gamma: captured.gamma,
        });
        let report = scanner.add_frame_with_color(&frame, width, height, color);
        print_progress(index, &report);

        if !on_frame(index, &scanner) {
            println!("[scan] stopped after {index} frames");
            break;
        }
    }

    capture.stop().await?;
    Ok(scanner)
}

async fn record(options: &Options) -> Result<(), Box<dyn Error>> {
    let frames = options.frames.unwrap_or(DEFAULT_FRAMES);
    let out = options
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("capture.k2df"));

    let color_enabled = options.color;
    let drain = if color_enabled { false } else { options.drain_color };
    let mut capture =
        capture::Capture::open(options.filters, color_enabled || drain, options.gpu).await?;
    let intrinsics = capture.intrinsics();
    let (width, height) = capture.dimensions();

    let mut writer =
        recording::FrameWriter::create_with_color(&out, intrinsics, width, height, color_enabled)?;
    let mode_desc = if color_enabled { "depth + colour" } else { "depth" };
    println!("[scan] recording {frames} {mode_desc} frames to {}", out.display());

    let started = std::time::Instant::now();
    for index in 1..=frames {
        let frame = capture.next_frame().await?.to_vec();
        let color = if color_enabled {
            capture.color().map(|c| FrameColor {
                rgb: &c.rgb,
                depth: &c.depth,
                valid: &c.valid,
                exposure: c.exposure,
                gain: c.gain,
                gamma: c.gamma,
            })
        } else {
            None
        };
        // Seconds since capture began, so the recording carries real timing.
        writer.write_frame_with_color(started.elapsed().as_secs_f32(), &frame, color)?;

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

    let duration = recording.frames.last().map_or(0.0, |frame| frame.timestamp);

    println!(
        "[scan] {} frames over {:.1} s of {}x{} from {} (colour: {})",
        recording.frames.len(),
        duration,
        recording.width,
        recording.height,
        input.display(),
        if recording.has_color { "yes" } else { "none" },
    );
    println!(
        "[scan] depth intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        recording.intrinsics.fx,
        recording.intrinsics.fy,
        recording.intrinsics.cx,
        recording.intrinsics.cy
    );
    if !recording.has_color && options.color {
        println!(
            "[scan] this .k2df recording has no colour stream; painting is unavailable"
        );
    }
    if !recording.has_color && options.dataset.is_some() {
        return Err(
            "cannot export dataset from this recording: it was captured without colour".into(),
        );
    }

    let mut scanner = Scanner::new(recording.intrinsics, scanner_config(options));

    // `--frames` limits how much of a recording is replayed. A recording is
    // deterministic, so a prefix of it is a valid and repeatable test case --
    // parameter tuning does not need the full capture every time.
    let limit = options.frames.unwrap_or(usize::MAX);

    for (index, frame) in recording.frames.iter().take(limit).enumerate() {
        let color = frame.color.as_ref().map(|c| FrameColor {
            rgb: &c.rgb,
            depth: &c.depth,
            valid: &c.valid,
            exposure: c.exposure,
            gain: c.gain,
            gamma: c.gamma,
        });
        let report = scanner.add_frame_with_color(
            &frame.depth,
            recording.width,
            recording.height,
            color,
        );
        print_progress(index + 1, &report);
    }

    finish_scan(&mut scanner, options)
}

/// Build a synthetic coloured capture and write it as a dataset.
///
/// The point is to exercise the exporter's whole path -- registration, colour
/// views, the mesh, the poses -- on a machine with no sensor, against a scene
/// whose true poses are known. A real capture cannot do that: there is no
/// reference trajectory to check against, so a wrong pose convention or a
/// transposed matrix looks exactly like a hard scan.
fn synth(options: &Options) -> Result<(), Box<dyn Error>> {
    if options.dataset.is_none() {
        return Err("`synth` needs --dataset DIR: writing one is the whole point".into());
    }

    let count = options.frames.unwrap_or(24).max(2);
    let intrinsics = synthetic::intrinsics();
    let shots = synthetic::shots(count);

    println!(
        "[scan] synthetic room, {} shots of {}x{}",
        shots.len(),
        synthetic::WIDTH,
        synthetic::HEIGHT,
    );

    let mut scanner = Scanner::new(intrinsics, scanner_config(options));

    for (index, shot) in shots.iter().enumerate() {
        let rendered = synthetic::render(shot, &intrinsics);

        let color = FrameColor {
            rgb: &rendered.rgb,
            // Registration is a no-op here: the render is already on the depth
            // grid with the depth camera's intrinsics, which is exactly the
            // state registration exists to produce.
            depth: &rendered.depth,
            valid: &rendered.valid,
            exposure: shot.brightness,
            gain: 1.0,
            gamma: 1.0,
        };

        let report = scanner.add_frame_with_color(
            &rendered.depth,
            synthetic::WIDTH,
            synthetic::HEIGHT,
            Some(color),
        );

        print_progress(index + 1, &report);
    }

    report_tracking_error(&scanner, &shots);
    finish_scan(&mut scanner, options)
}

/// How far the tracked trajectory ended from where the camera really was.
///
/// This is the one number in the project that has a right answer. Every other
/// drift figure comes from a capture with no reference, so it can only be
/// compared against another run of the same broken thing.
fn report_tracking_error(scanner: &Scanner, shots: &[synthetic::Shot]) {
    let lived = scanner.trajectory();
    if lived.is_empty() {
        println!("[scan] ground truth: nothing was tracked");
        return;
    }

    // The scanner defines the world frame as the *first frame's camera frame*, so
    // a later pose is comparable only after the anchor has been divided out.
    // Comparing against the raw eye positions is off by the anchor's own pose,
    // which is a metre of nonsense on this path.
    let anchor = shots[0].pose;
    let aligned: Vec<Vector3<f32>> = shots
        .iter()
        .map(|shot| (anchor.inverse() * shot.pose).translation.vector)
        .collect();

    let truth = aligned[aligned.len() - 1];
    let ended = lived[lived.len() - 1];

    // The path the camera actually took, for comparison against how far the
    // tracker believes it went.
    let truth_path: f32 = aligned.windows(2).map(|pair| (pair[1] - pair[0]).norm()).sum();
    let lived_path: f32 = lived.windows(2).map(|pair| (pair[1] - pair[0]).norm()).sum();

    println!(
        "[scan] ground truth: path {:.3} m tracked as {:.3} m over {} of {} frames",
        truth_path,
        lived_path,
        lived.len(),
        shots.len(),
    );

    // Only the same frame is being compared when every frame was tracked; with
    // rejections this is the last tracked pose against the path's end, which
    // overstates the error by however far the tail travelled.
    let note = if lived.len() == shots.len() {
        ""
    } else {
        "  (skewed: frames were rejected, so this is not the same frame)"
    };
    println!(
        "[scan] ground truth: final position off by {:.4} m{note}",
        (ended - truth).norm(),
    );
}

fn scanner_config(options: &Options) -> ScannerConfig {
    ScannerConfig {
        tsdf: options.tsdf,
        loop_closure: options.loop_closure.then(LoopClosureConfig::default),
        odometry: OdometryConfig {
            frame_to_model: options.frame_to_model,
            ..OdometryConfig::default()
        },
        coloring: ColoringParams {
            mode: options.color_mode,
            depth_tolerance: options.color_depth_tolerance,
            ..ColoringParams::default()
        },
        ..ScannerConfig::default()
    }
}

/// Close loops, then write the mesh. Live, replay, and the viewer window all
/// end here, so closing the window still corrects the trajectory before export.
pub(crate) fn finish_scan(scanner: &mut Scanner, options: &Options) -> Result<(), Box<dyn Error>> {
    close_loops(scanner, options);
    finish(scanner, options)
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
        report.loop_edges, report.nodes, report.cost_before, report.cost_after, report.iterations
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

    // A frame that closes a gap jumped by that many frames' worth of motion. It
    // is worth calling out: a scan that is only recovering between failures
    // should look like one, rather than like a scan that is tracking.
    let recovered = match report.track.recovered {
        0 => String::new(),
        gap => format!("  RECOVERED a {gap}-frame gap"),
    };

    // `move` is the recovered sensor motion for this frame and `iters` the total
    // ICP iterations spent on it: together they are the quickest way to spot a
    // jump that the acceptance gates should have caught, or a solve that is
    // burning its whole budget without settling.
    println!(
        "[scan] {:>4}  pts {:>6}  inliers {:>5.1}%  rmse {:>5.1} mm  move {:>5.1} cm  iters {:>3}  pos [{:>6.3} {:>6.3} {:>6.3}]  {outcome}{recovered}",
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

fn print_mesh_extent(vertices: usize, triangles: usize, min: Vector3<f32>, max: Vector3<f32>) {
    println!(
        "[scan] mesh: {} vertices, {} triangles, extent {:.2} x {:.2} x {:.2} m",
        vertices,
        triangles,
        max.x - min.x,
        max.y - min.y,
        max.z - min.z
    );
}

fn print_color_views(scanner: &Scanner) {
    let kept = scanner.color_view_count();
    let supplied = scanner.color_supplied();
    if kept == 0 && supplied == 0 {
        return;
    }
    println!(
        "{}",
        format_color_views(kept, scanner.color_dropped(), scanner.rejected())
    );
}

fn format_color_views(kept: usize, dropped: usize, rejected: usize) -> String {
    format!(
        "[scan] colour: {kept} views kept, {dropped} dropped with rejected frames ({rejected} frames rejected)"
    )
}

fn format_paint(report: &ColoringReport) -> String {
    format!(
        "[scan] paint: {}/{} vertices painted ({:.1}%), {:.1} samples per painted vertex",
        report.painted(),
        report.vertices,
        report.coverage() * 100.0,
        report.mean_samples(),
    )
}

fn format_atlas(report: &TexturingReport) -> String {
    let scale = if report.downscaled {
        ", downscaled"
    } else {
        ""
    };
    format!(
        "[scan] atlas: {}x{}, {} charts, {}/{} triangles painted ({:.1}%), {} unobserved{}",
        report.atlas_width,
        report.atlas_height,
        report.charts,
        report.painted(),
        report.triangles,
        report.coverage() * 100.0,
        report.unobserved,
        scale,
    )
}

fn position_bounds(positions: &[Vector3<f32>]) -> Option<(Vector3<f32>, Vector3<f32>)> {
    let mut iter = positions.iter();
    let first = *iter.next()?;
    let mut min = first;
    let mut max = first;
    for position in iter {
        min = min.inf(position);
        max = max.sup(position);
    }
    Some((min, max))
}

pub(crate) fn finish(scanner: &Scanner, options: &Options) -> Result<(), Box<dyn Error>> {
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
    print_color_views(scanner);
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
    let format = MeshFormat::from_path(&out)?;

    println!("[scan] extracting surface ...");
    // The atlas is built in the volume's frame. Mirroring happens inside
    // `export_mesh`, after that, so positions, normals and winding all flip
    // and the texture coordinates stay on the image that was just painted.
    let Some(exported) = scanner.export_mesh(options.unmirror, format.uses_texture_atlas()) else {
        println!("[scan] no surface extracted - was anything in range?");
        return Ok(());
    };

    match &exported {
        ExportMesh::Geometry(mesh, coloring) => {
            if let Some((min, max)) = mesh.bounds() {
                print_mesh_extent(mesh.vertex_count(), mesh.triangle_count(), min, max);
            }
            if let Some(report) = coloring {
                println!("{}", format_paint(report));
            }
        }
        ExportMesh::Atlas(textured, report) => {
            if let Some((min, max)) = position_bounds(&textured.positions) {
                print_mesh_extent(
                    textured.positions.len(),
                    textured.triangle_count(),
                    min,
                    max,
                );
            }
            println!("{}", format_atlas(report));
        }
    }

    match exported {
        ExportMesh::Geometry(mesh, _) => export::save_mesh(&mesh, &out)?,
        ExportMesh::Atlas(textured, _) => export::save_textured(&textured, &out)?,
    }
    println!("[scan] wrote {}", out.display());

    let trajectory = out.with_extension("trajectory.ply");
    let trajectory_points: Vec<Vector3<f32>> = if options.unmirror {
        scanner
            .trajectory()
            .iter()
            .map(|p| Vector3::new(-p.x, p.y, p.z))
            .collect()
    } else {
        scanner.trajectory().to_vec()
    };
    geom::mesh::save_points_ply(&trajectory, &trajectory_points)?;
    println!(
        "[scan] wrote {} ({} camera positions)",
        trajectory.display(),
        trajectory_points.len()
    );

    if let Some(dir) = &options.dataset {
        // Written last, and after the poses are final: `close_loops` ran in
        // `finish_scan`, and a correction moves the views onto the rebuilt
        // trajectory. A dataset assembled before that would carry the drifted
        // poses the trainer is most sensitive to.
        println!("[scan] writing dataset to {} ...", dir.display());
        let report = dataset::write_dataset(scanner, dir)?;

        println!(
            "[scan] dataset: {} views, {} initial points in {}",
            report.views,
            report.points,
            report.dir.display()
        );
        if report.skipped > 0 {
            println!(
                "[scan]   {} views skipped: their pose could not be written",
                report.skipped
            );
        }
        println!(
            "[scan]   images, poses and init.ply are all in the sensor frame, so they agree; \
             --mirror does not apply to a dataset"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use geom::Intrinsics;

    #[test]
    fn finish_names_an_unsupported_extension() {
        let scanner = Scanner::new(
            Intrinsics {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
            },
            ScannerConfig::default(),
        );
        let mut options = Options::default();
        options.out = Some(PathBuf::from("not-a-mesh.stl"));
        let error = finish(&scanner, &options).expect_err("stl");
        let message = error.to_string();
        assert!(message.contains("\".stl\""), "{message}");
        assert!(
            message.contains(".ply") && message.contains(".glb"),
            "{message}"
        );
    }

    fn words(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_string()).collect()
    }

    #[test]
    fn usage_lists_the_color_controls_and_the_mesh_extensions() {
        let usage = usage_text();
        assert!(usage.contains("--color-mode best|blend|average"), "{usage}");
        assert!(
            usage.contains("Per-vertex PLY colour"),
            "help should say --color-mode is the PLY painter: {usage}"
        );
        assert!(
            usage.contains("one source view that sees it best"),
            "help should say atlases pick one view per triangle: {usage}"
        );
        assert!(usage.contains("1.66 MiB"), "{usage}");
        assert!(usage.contains("850 KB"), "{usage}");
        assert!(usage.contains("--color-depth-tolerance M"), "{usage}");
        assert!(usage.contains("default 0.02"), "{usage}");
        assert!(usage.contains(".mtl"), "{usage}");
        assert!(usage.contains(".bin"), "{usage}");
        assert!(usage.contains(".glb"), "{usage}");
        assert!(usage.contains("depth only"), "{usage}");
    }

    #[test]
    fn color_flags_default_to_best_and_two_centimetres_and_reach_the_config() {
        let (command, options) = parse_args(words(&["live", "--color"])).expect("parse");
        assert!(matches!(command, Command::Live));
        assert!(options.color);
        let config = scanner_config(&options);
        assert_eq!(config.coloring.mode, ColorMode::Best);
        assert!((config.coloring.depth_tolerance - 0.02).abs() < 1e-6);
        assert_eq!(config.coloring.fallback, ColoringParams::default().fallback);

        let (_, blend) = parse_args(words(&[
            "live",
            "--color-mode",
            "blend",
            "--color-depth-tolerance",
            "0.04",
        ]))
        .expect("parse");
        let config = scanner_config(&blend);
        assert_eq!(config.coloring.mode, ColorMode::Blend);
        assert!((config.coloring.depth_tolerance - 0.04).abs() < 1e-6);
        assert_eq!(
            config.coloring.image_margin,
            ColoringParams::default().image_margin
        );

        let (_, average) =
            parse_args(words(&["replay", "--color-mode", "average"])).expect("parse");
        assert_eq!(scanner_config(&average).coloring.mode, ColorMode::Average);
    }

    #[test]
    fn bad_color_flags_are_rejected() {
        let mode = parse_args(words(&["live", "--color-mode", "Best"])).expect_err("case");
        assert!(
            mode.to_string().contains("best, blend, or average"),
            "{mode}"
        );

        let missing = parse_args(words(&["live", "--color-mode"])).expect_err("missing");
        assert!(missing.to_string().contains("needs a value"), "{missing}");

        for value in ["0", "-0.01", "nan", "inf"] {
            let error =
                parse_args(words(&["live", "--color-depth-tolerance", value])).expect_err(value);
            assert!(
                error.to_string().contains("greater than 0"),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn a_dataset_path_is_parsed_and_a_colour_scan_carries_it() {
        let (command, options) =
            parse_args(words(&["live", "--color", "--dataset", "out/room-dataset"]))
                .expect("parse");

        assert!(matches!(command, Command::Live));
        assert_eq!(
            options.dataset,
            Some(PathBuf::from("out/room-dataset")),
            "--dataset should carry its path"
        );
        // The dataset is built from the colour views, so the flag that produces
        // them has to survive alongside it.
        assert!(options.color);
    }

    #[test]
    fn dataset_without_color_is_refused_before_the_sensor_is_opened() {
        let error = parse_args(words(&["live", "--dataset", "out/d"])).expect_err("no colour");

        assert!(
            error.to_string().contains("--dataset needs --color"),
            "{error}"
        );
    }

    #[test]
    fn dataset_is_refused_for_record_command() {
        let error = parse_args(words(&["record", "--dataset", "out/d"])).expect_err("record");
        assert!(
            error.to_string().contains("--dataset cannot be used with `record`"),
            "{error}"
        );
    }

    #[test]
    fn dataset_is_allowed_for_replay_at_parse_time() {
        let (command, options) = parse_args(words(&["replay", "--in", "capture.k2df", "--dataset", "out/d"])).expect("parse");
        assert!(matches!(command, Command::Replay));
        assert_eq!(options.dataset, Some(PathBuf::from("out/d")));
    }

    #[test]
    fn dataset_needs_a_value() {
        let error = parse_args(words(&["live", "--color", "--dataset"])).expect_err("missing");
        assert!(error.to_string().contains("needs a value"), "{error}");
    }

    #[test]
    fn usage_documents_the_dataset_and_its_limits() {
        let usage = usage_text();

        assert!(usage.contains("--dataset DIR"), "{usage}");
        assert!(usage.contains("transforms.json"), "{usage}");
        assert!(usage.contains("init.ply"), "{usage}");
        assert!(usage.contains("masks/"), "{usage}");
        // The resolution ceiling is stated rather than discovered.
        assert!(usage.contains("512x424"), "{usage}");
        // And that a recording cannot be turned into one after the fact.
        assert!(usage.contains(".k2df recording has no colour"), "{usage}");
    }

    #[test]
    fn end_of_scan_lines_report_kept_views_and_coverage() {
        let views = format_color_views(12, 3, 3);
        assert!(views.contains("12 views kept"), "{views}");
        assert!(views.contains("3 dropped"), "{views}");
        assert!(views.contains("3 frames rejected"), "{views}");

        let paint = format_paint(&ColoringReport {
            vertices: 10,
            unobserved: 4,
            samples: 12,
            exposure_scales: vec![1.0],
        });
        assert!(paint.contains("6/10"), "{paint}");
        assert!(paint.contains("60.0%"), "{paint}");
        assert!(paint.contains("2.0 samples"), "{paint}");

        let atlas = format_atlas(&TexturingReport {
            triangles: 8,
            unobserved: 2,
            charts: 3,
            atlas_width: 64,
            atlas_height: 32,
            downscaled: true,
            exposure_scales: Vec::new(),
        });
        assert!(atlas.contains("64x32"), "{atlas}");
        assert!(atlas.contains("6/8"), "{atlas}");
        assert!(atlas.contains("75.0%"), "{atlas}");
        assert!(atlas.contains("2 unobserved"), "{atlas}");
        assert!(atlas.contains("downscaled"), "{atlas}");
    }

    #[test]
    fn finish_scan_closes_loops_before_an_empty_mesh_returns() {
        // No surface, so `finish` returns before it writes a file. Closure is
        // recorded only when loop closure actually ran, which has to happen
        // first: the viewer, live, and replay paths all end in `finish_scan`.
        let intrinsics = Intrinsics {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
        };

        let mut closed = Scanner::new(intrinsics, ScannerConfig::default());
        let mut with_loops = Options::default();
        with_loops.loop_closure = true;
        finish_scan(&mut closed, &with_loops).expect("empty scan");
        assert!(
            closed.closure().is_some(),
            "finish_scan returned without closing loops"
        );

        let mut left = Scanner::new(intrinsics, ScannerConfig::default());
        finish_scan(&mut left, &Options::default()).expect("empty scan");
        assert!(left.closure().is_none());
    }
}
