//! `scan` — handheld 3D scanner for the Kinect v2.
//!
//! Depth odometry tracks the sensor frame to frame, and every frame whose pose is
//! trusted is fused into a voxel-hashed TSDF. The zero crossing of that field is
//! extracted as a triangle mesh.
//!
//! ```text
//! scan gui    [--in FILE | --synthetic] [--frames N] [--out mesh.ply] [--color]
//!             [--auto-start] [--preview-every N] [--playback-fps F]
//! scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color]
//!             [--color-mode best|blend|average] [--color-depth-tolerance M]
//!             [--gpu] [--loop-closure] [--viewer] [--dataset DIR]
//! scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu] [--color]
//! scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N] [--loop-closure]
//! ```
//!
//! `gui` is the same scanner with a window: choose the source, start, pause,
//! stop, and export. It can replay a recording or run the synthetic room, so the
//! real-time path is usable on a machine with no sensor.
//!
//! `record` and `replay` exist so that odometry can be tuned against
//! byte-identical input instead of a live sensor: the scene changes between live
//! runs, so two parameter sets never see the same data. A `.k2df` file may carry
//! colour or be depth only; `replay` paints from the colour views when the
//! recording has them.
//!
//! `--dataset` turns a colour scan into something a Gaussian-splatting trainer
//! can read; see [`dataset`]. It needs colour views, from `live --color` or a
//! colour recording replayed with `--dataset`.

mod capture;
mod dataset;
#[cfg(feature = "gui")]
mod gui;
mod loop_closure;
mod odometry;
mod recording;
mod scanner;
mod synthetic;
#[cfg(feature = "wgpu-decode")]
mod wgpu_depth;

use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;

use geom::DepthImage;
use geom::coloring::{ColorMode, ColoringParams, ColoringReport};
use geom::export::{self, MeshFormat};
use geom::frozen_map::{FrozenMap, FrozenMapParams};
use geom::level::{self, LevelParams, Leveling};
use geom::texturing::TexturingReport;
use geom::tsdf::TsdfParams;
use loop_closure::LoopClosureConfig;
use nalgebra::{Isometry3, Vector3};
use odometry::OdometryConfig;
use scanner::{
    ExportMesh, FrameColor, FrameReport, MotionFilterConfig, ScanQuality, Scanner, ScannerConfig,
};

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
    /// GUI source: use the analytic room instead of a sensor. Needs `gui`.
    synthetic: bool,
    /// GUI: start the selected source as soon as the window opens.
    auto_start: bool,
    /// GUI: extract and publish a preview mesh every N frames.
    preview_every: usize,
    /// GUI: replay and synthetic frames per second. Zero is unthrottled.
    playback_fps: f32,
    /// Un-mirror the reconstruction horizontally so real-world left and right match.
    /// True by default: the Kinect v2 sensor reads out mirrored frames.
    unmirror: bool,
    /// Rotate the exported model so its dominant floor or ceiling is horizontal.
    /// Off by default: it assumes a large flat surface and an upright sensor.
    level: bool,
    /// Track each frame against a render of the fused model instead of against
    /// the previous frame, so tracking error stops accumulating.
    ///
    /// Costs a raycast per frame, which is why it is not the default: at sensor
    /// resolution that is around a second per frame on this hardware. Live
    /// capture becomes a slideshow, so the useful place for it is `replay`,
    /// where wall clock does not matter and the poses it produces are the ones
    /// a scan is ultimately graded on.
    frame_to_model: bool,
    /// When the projective tracker rejects a frame, retry it as a spatial ICP
    /// against the last accepted frame's cloud. The wide search can recover
    /// motion that has left the previous view; strict inlier and residual
    /// gates decide whether to trust it. Costs a cloud build on rejected
    /// frames only, so a clean scan pays nothing.
    relocalize: bool,
    /// Start a new submap when a frame cannot be placed in the current one.
    /// The frame becomes the new submap's first frame at the failed alignment's
    /// candidate pose. On by default; `--no-submaps` restores the old
    /// drop-only behaviour for comparison.
    submaps: bool,
    /// Apply a verified loop as soon as it arrives, instead of waiting for the
    /// end-of-scan pass. Implies `--loop-closure`.
    online_loops: bool,
    /// Run loop ICP verification on a worker thread. Implies `--loop-closure`.
    loop_worker: bool,
    /// How many keyframes on each side of a loop candidate to merge into the
    /// ICP target.
    loop_window: usize,
    /// How many recent accepted keyframes the spatial re-localisation map
    /// keeps. Implies `--relocalize`.
    relocalize_keyframes: usize,
    /// Correct the final model by moving the cached submap surfaces instead of
    /// re-integrating every frame. Implies `--loop-closure`.
    incremental_rebuild: bool,
    /// Skip fusion for frames that barely moved since the last fused frame.
    fusion_filter: bool,
    /// Finish the active TSDF into a cached chunk every N fused frames.
    tsdf_chunk_frames: usize,
    /// Let spatial re-localisation fall back to the 4-DOF global search.
    relocalize_global: bool,
    /// Let loop verification fall back to the 4-DOF global search.
    loop_global: bool,
    /// Minimum likelihood for a loop's final alignment. Zero disables.
    match_score: f32,
    /// Optimise submap nodes instead of keyframe nodes.
    submap_graph: bool,
    /// Finish a tracking submap after this many fused frames. Zero disables.
    submap_max_frames: usize,
    /// Finish a tracking submap after this much camera motion. Zero disables.
    submap_max_translation: f32,
    /// Run the online pose-graph solve on a worker thread.
    graph_worker: bool,
    /// Export a Nerfstudio dataset for a Gaussian-splatting trainer: the colour
    /// views, their masks, `transforms.json`, and an initial point cloud.
    /// Needs `--color`, because the dataset is built from the colour views.
    /// The dataset follows the mesh: un-mirrored together by default, left in the
    /// sensor frame with `--mirror`.
    dataset: Option<PathBuf>,
    /// Prior point cloud for `localize`, as ASCII or binary little-endian PLY.
    map: Option<PathBuf>,
    /// `localize`: use the live sensor instead of a recording.
    localize_live: bool,
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
            synthetic: false,
            auto_start: false,
            preview_every: 2,
            playback_fps: 10.0,
            unmirror: true,
            level: false,
            frame_to_model: false,
            relocalize: false,
            submaps: true,
            online_loops: false,
            loop_worker: false,
            loop_window: LoopClosureConfig::default().loop_target_keyframes,
            relocalize_keyframes: OdometryConfig::default().relocalize_keyframes,
            incremental_rebuild: false,
            fusion_filter: false,
            tsdf_chunk_frames: 0,
            relocalize_global: false,
            loop_global: false,
            match_score: 0.0,
            submap_graph: false,
            submap_max_frames: 0,
            submap_max_translation: 0.0,
            graph_worker: false,
            dataset: None,
            map: None,
            localize_live: false,
        }
    }
}

#[derive(Debug)]
enum Command {
    /// The scanner window: source picker, live preview, transport and export.
    Gui,
    Live,
    Record,
    Replay,
    /// Localise a recording against a frozen prior map.
    Localize,
    /// Build a synthetic coloured capture and write it as a dataset.
    Synth,
    Help,
}

async fn run() -> Result<(), Box<dyn Error>> {
    let (command, options) = parse_args(std::env::args().skip(1))?;
    match command {
        Command::Gui => gui(&options),
        Command::Live => live(&options).await,
        Command::Record => record(&options).await,
        Command::Replay => replay(&options),
        Command::Localize => localize(&options).await,
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
        "gui" | "app" => Command::Gui,
        "live" => Command::Live,
        "record" => Command::Record,
        "replay" => Command::Replay,
        "localize" => Command::Localize,
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
            "--map" => options.map = Some(PathBuf::from(next_value(&mut args, "--map")?)),
            "--live" => options.localize_live = true,
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
            "--synthetic" => options.synthetic = true,
            "--auto-start" => options.auto_start = true,
            "--preview-every" => {
                let every: usize = next_value(&mut args, "--preview-every")?.parse()?;
                if every == 0 {
                    return Err("--preview-every must be at least 1".into());
                }
                options.preview_every = every;
            }
            "--playback-fps" => {
                let fps: f32 = next_value(&mut args, "--playback-fps")?.parse()?;
                if !(fps.is_finite() && fps >= 0.0) {
                    return Err("--playback-fps must be a finite number of zero or more".into());
                }
                options.playback_fps = fps;
            }
            "--loop-closure" => options.loop_closure = true,
            "--gpu" => options.gpu = true,
            "--mirror" => options.unmirror = false,
            "--no-mirror" | "--unmirror" => options.unmirror = true,
            "--level" => options.level = true,
            "--frame-to-model" => options.frame_to_model = true,
            "--relocalize" => options.relocalize = true,
            "--relocalize-keyframes" => {
                let count: usize = next_value(&mut args, "--relocalize-keyframes")?.parse()?;
                if count == 0 {
                    return Err("--relocalize-keyframes must be at least 1".into());
                }
                options.relocalize_keyframes = count;
                options.relocalize = true;
            }
            "--relocalize-global" => {
                options.relocalize_global = true;
                options.relocalize = true;
            }
            "--fusion-filter" => options.fusion_filter = true,
            "--tsdf-chunk" => {
                let frames: usize = next_value(&mut args, "--tsdf-chunk")?.parse()?;
                if frames == 0 {
                    return Err("--tsdf-chunk must be at least 1".into());
                }
                options.tsdf_chunk_frames = frames;
            }
            "--loop-window" => {
                options.loop_window = next_value(&mut args, "--loop-window")?.parse()?;
            }
            "--loop-global" => {
                options.loop_global = true;
                options.loop_closure = true;
            }
            "--match-score" => {
                let score: f32 = next_value(&mut args, "--match-score")?.parse()?;
                if !(score.is_finite() && (0.0..=1.0).contains(&score)) {
                    return Err("--match-score must be between 0 and 1".into());
                }
                options.match_score = score;
                options.loop_closure = true;
            }
            "--submap-graph" => {
                options.submap_graph = true;
                options.loop_closure = true;
            }
            "--submap-frames" => {
                options.submap_max_frames = next_value(&mut args, "--submap-frames")?.parse()?;
            }
            "--submap-metres" => {
                let metres: f32 = next_value(&mut args, "--submap-metres")?.parse()?;
                if !(metres.is_finite() && metres >= 0.0) {
                    return Err("--submap-metres must be zero or positive".into());
                }
                options.submap_max_translation = metres;
            }
            "--graph-worker" => {
                options.graph_worker = true;
                options.online_loops = true;
                options.loop_closure = true;
            }
            "--online-loops" => {
                options.online_loops = true;
                options.loop_closure = true;
            }
            "--loop-worker" => {
                options.loop_worker = true;
                options.loop_closure = true;
            }
            "--incremental-rebuild" => {
                options.incremental_rebuild = true;
                options.loop_closure = true;
            }
            "--no-submaps" => options.submaps = false,
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
                );
            }
            Command::Record => return Err(
                "--dataset cannot be used with `record`: a record run only writes a capture file. \
                     Use `live --color --dataset DIR`, or replay a colour recording with \
                     `scan replay --in FILE --dataset DIR`"
                    .into(),
            ),
            // `replay` is allowed with `--dataset`; the recording itself is checked in
            // `replay()` to ensure it carries a colour stream.
            Command::Replay => {}
            // Localisation is against a frozen map; it builds no new dataset.
            Command::Localize => {
                return Err(
                    "--dataset cannot be used with `localize`: it builds no new map".into(),
                );
            }
            // `synth` renders its own colour, so it needs no stream and no flag.
            Command::Synth => {}
            // The window has its own source picker. It refuses a live dataset
            // without colour before the scan, and a replay or synthetic source
            // either has colour views or reports their absence at export.
            Command::Gui => {}
            Command::Help => {}
        }
    }

    if matches!(command, Command::Localize) {
        if options.map.is_none() {
            return Err("`localize` needs --map <prior.ply>".into());
        }
        match (options.input.is_some(), options.localize_live) {
            (true, false) | (false, true) => {}
            (true, true) => {
                return Err("`localize` accepts --in or --live, not both".into());
            }
            (false, false) => {
                return Err("`localize` needs --in <capture.k2df> or --live".into());
            }
        }
    } else if options.map.is_some() {
        return Err("--map only applies to `scan localize`".into());
    } else if options.localize_live {
        return Err("--live only applies to `scan localize`".into());
    }

    // A GUI source is either a sensor, a recording, or the synthetic room.
    // Asking for two at once is a typo, not a preference.
    if options.synthetic && options.input.is_some() {
        return Err("--synthetic and --in select different GUI sources".into());
    }
    if options.synthetic && !matches!(command, Command::Gui) {
        return Err("--synthetic only applies to `scan gui`".into());
    }
    if options.auto_start
        && !matches!(command, Command::Gui)
        && !(matches!(command, Command::Live) && options.viewer)
    {
        return Err("--auto-start only applies to `scan gui` or `live --viewer`".into());
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
         \x20 scan gui    [--in FILE | --synthetic] [--auto-start] [--frames N]\n\
         \x20             [--out mesh.ply] [--color] [--preview-every N]\n\
         \x20             [--playback-fps F]\n\
         \x20 scan live   [--frames N] [--out mesh.ply] [--voxel M] [--no-filter] [--color]\n\
         \x20             [--color-mode best|blend|average] [--color-depth-tolerance M]\n\
         \x20             [--gpu] [--loop-closure] [--viewer] [--dataset DIR]\n\
         \x20 scan record  --out capture.k2df [--frames N] [--no-filter] [--drain-color] [--gpu]\n\
         \x20             [--color]\n\
         \x20 scan replay  --in capture.k2df [--out mesh.ply] [--voxel M] [--frames N]\n\
         \x20             [--loop-closure]\n\
         \x20 scan localize (--in capture.k2df | --live) --map prior.ply [--frames N]\n\
         \x20             [--out trajectory.ply]\n\
         \x20 scan synth   --dataset DIR [--frames N] [--out mesh.ply] [--voxel M]\n\
         \n\
         COMMANDS:\n\
         \x20 gui      Open the scanner window: pick a source, scan, export.\n\
         \x20          It can replay a recording or run a synthetic room, so it\n\
         \x20          needs no sensor. Needs `--features gui` and a display.\n\
         \x20 live     Capture from the sensor and reconstruct as you go.\n\
         \x20 record   Capture depth frames to a file, for offline tuning.\n\
         \x20 replay   Reconstruct from a recording. Needs no sensor.\n\
         \x20 localize Localise a recording or live sensor against a frozen prior map. The map\n\
         \x20          is never modified; the output is a trajectory.\n\
         \x20 synth    Render the analytic room and write a colour dataset. Needs\n\
         \x20          no sensor or display; the result feeds Brush directly.\n\
         \n\
         OPTIONS:\n\
         \x20 --frames N    With live/record: frames to capture (default {DEFAULT_FRAMES}).\n\
         \x20               With replay: how many recorded frames to process. A\n\
         \x20               recording is deterministic, so a prefix of it is a\n\
         \x20               repeatable test case.\n\
         \x20 --out FILE    Output mesh (default scan.ply). The extension selects\n\
         \x20               the format: .ply (per-vertex colour), .obj (plus .mtl\n\
         \x20               and .png), .gltf (plus .bin and .png), or .glb (one file).\n\
         \x20 --in FILE     Recording to replay. A .k2df file may carry colour\n\
         \x20               or be depth only. With `gui`, preselects replay.\n\
         \x20 --map FILE    `localize`: prior map as ASCII or binary little-endian\n\
         \x20               PLY. With --live, use the sensor instead of --in. Only the vertex positions are read.\n\
         \x20               A map exported with or without --mirror both work: the\n\
         \x20               orientation is detected on the first frames.\n\
         \x20 --synthetic   On `gui`: use the analytic room, no sensor needed.\n\
         \x20 --auto-start  On `gui`: start the selected source when the window\n\
         \x20               opens. `live --viewer` does this too.\n\
         \x20 --preview-every N\n\
         \x20               On `gui`: extract a preview mesh every N frames\n\
         \x20               (default 2). One updates every frame, at a cost.\n\
         \x20 --playback-fps F\n\
         \x20               On `gui`: replay and synthetic playback rate (default\n\
         \x20               10). Zero runs as fast as the reconstruction allows.\n\
         \x20 --voxel M     TSDF voxel size in metres (default 0.01). Sets the\n\
         \x20               truncation band to 4x this.\n\
         \x20 --fusion-filter\n\
         \x20               Track but do not fuse a frame that moved less than\n\
         \x20               1 cm / 0.005 rad since the last fused frame. A forced\n\
         \x20               refresh every 5 frames keeps a stationary model\n\
         \x20               averaging. Fusion is the largest per-frame cost, so\n\
         \x20               this is the cheapest way to recover time on slow scans.\n\
         \x20 --tsdf-chunk N\n\
         \x20               Finish the active TSDF into a cached chunk every N\n\
         \x20               fused frames. Bounds the active volume and fusion cost\n\
         \x20               without resetting the odometry or creating a pose\n\
         \x20               boundary. Zero (default) disables.\n\
         \x20 --no-filter   Disable the decoder's bilateral/edge filters: roughly\n\
         \x20               doubles frame rate, keeps more junk points.\n\
         \x20 --color       On live: capture colour and paint the mesh. Off by\n\
         \x20               default. The colour stream is slower than depth.\n\
         \x20               On record: store the registered colour in the .k2df\n\
         \x20               file, so replay can paint from it.\n\
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
         \x20 --viewer      Open the scanner window on the live source and start\n\
         \x20               immediately. The same window as `scan gui`, and the\n\
         \x20               same feature. Needs `--features gui` (or the older\n\
         \x20               `viewer` alias) and a display. Closing the window\n\
         \x20               stops the scan and exports it.\n\
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
         \x20 --relocalize  When the projective tracker rejects a frame, retry it\n\
         \x20               as a spatial ICP against the last accepted frame's\n\
         \x20               cloud. The wider search can recover motion that left\n\
         \x20               the previous view; strict inlier and residual gates\n\
         \x20               decide whether to trust it. Pays a cloud build on\n\
         \x20               rejected frames only, so a clean scan is unaffected.\n\
         \x20 --relocalize-keyframes N\n\
         \x20               Keep N recent accepted keyframes as a local map for\n\
         \x20               that spatial retry (default 6). The target is the\n\
         \x20               merged set within the map radius, not just the last\n\
         \x20               frame, which gives a turning sensor more to match.\n\
         \x20               Implies --relocalize.\n\
         \x20 --relocalize-global\n\
         \x20               If the spatial retry's ICP still fails, run the\n\
         \x20               4-DOF yaw/translation branch-and-bound search against\n\
         \x20               the local keyframe map. ICP still decides whether the\n\
         \x20               result is real. Implies --relocalize.\n\
         \x20 --loop-window N\n\
         \x20               Merge N keyframes on each side of a loop candidate\n\
         \x20               into the ICP target (default 2). Zero restores the\n\
         \x20               single-keyframe target.\n\
         \x20 --loop-global\n\
         \x20               If loop ICP fails from the drifted pose, run the\n\
         \x20               4-DOF branch-and-bound search before rejecting the\n\
         \x20               candidate. The reciprocal and likelihood gates still\n\
         \x20               apply. Implies --loop-closure.\n\
         \x20 --match-score S\n\
         \x20               Minimum mean likelihood of a verified loop against the\n\
         \x20               target's surface grid (0..1). Zero disables. This is a\n\
         \x20               different signal from ICP inliers and catches repetitive\n\
         \x20               geometry that happens to fit. Implies --loop-closure.\n\
         \x20 --online-loops\n\
         \x20               Apply a verified loop during the scan instead of\n\
         \x20               waiting for the end. Future frames use the corrected\n\
         \x20               trajectory; cached submaps move rigidly. Implies\n\
         \x20               --loop-closure.\n\
         \x20 --loop-worker\n\
         \x20               Run loop ICP on a worker thread so a candidate does\n\
         \x20               not stall capture. Implies --loop-closure.\n\
         \x20 --incremental-rebuild\n\
         \x20               Correct the final model by moving cached submap\n\
         \x20               surfaces instead of re-integrating every frame.\n\
         \x20               Faster, but cannot fix drift inside a submap.\n\
         \x20               Implies --loop-closure.\n\
         \x20 --submap-graph\n\
         \x20               Optimise submap nodes rather than keyframe nodes,\n\
         \x20               A sparse pose-adjustment shape: smaller,\n\
         \x20               and rigid per submap, so it matches the cached surfaces\n\
         \x20               but cannot deform inside one. Off by default.\n\
         \x20               Implies --loop-closure.\n\
         \x20 --graph-worker\n\
         \x20               Run the online pose-graph solve (keyframe or submap)\n\
         \x20               on a worker thread, so capture does not stall. Implies\n\
         \x20               --online-loops.\n\
         \x20 --submap-frames N\n\
         \x20               Finish a tracking submap after N fused frames. Odometry\n\
         \x20               continues across the boundary, so the edge is trusted.\n\
         \x20               Zero disables.\n\
         \x20 --submap-metres M\n\
         \x20               Finish a tracking submap once the camera has moved M\n\
         \x20               metres from its first pose. Zero disables.\n\
         \x20 --no-submaps  Do not start a new submap when a frame cannot be\n\
         \x20               placed in the current one. Submaps are on by default:\n\
         \x20               the boundary keeps the fragments separate instead of\n\
         \x20               dropping frames or smearing them onto the old model.\n\
         \x20 --mirror      Keep the raw sensor mirror orientation instead of flipping\n\
         \x20               X to match real-world coordinates (un-mirrored by default).\n\
         \x20               A --dataset follows the same choice: its images, poses and\n\
         \x20               seed cloud are mirrored together by default.\n\
         \x20 --level       Rotate the exported mesh and trajectory so the floor or\n\
         \x20               ceiling is horizontal. The world frame is the first camera\n\
         \x20               frame, so a tilted start gives a tilted model. Needs a\n\
         \x20               large flat surface within 25 degrees of level; otherwise\n\
         \x20               the model is left as scanned. Not applied to --dataset.\n\
         \x20 --dataset DIR With colour views (live --color, or a colour recording):\n\
         \x20               also write a dataset a Gaussian-\n\
         \x20               splatting trainer can read. DIR gains images/ (the\n\
         \x20               registered colour, one PNG per kept view), masks/ (white\n\
         \x20               where registration copied a colour sample, so the trainer\n\
         \x20               ignores the holes), transforms.json (depth intrinsics and\n\
         \x20               the pose of every view), and init.ply (the scanned surface,\n\
         \x20               coloured, as the initial Gaussian positions).\n\
         \x20               Images are at depth resolution, 512x424, not the colour\n\
         \x20               sensor's own size: registration copies colour into the\n\
         \x20               depth grid, and that is what keeps the intrinsics and the\n\
         \x20               poses exact. A depth-only recording cannot: it has no\n\
         \x20               colour views.\n\
         \n\
         A trajectory PLY is written alongside the mesh as <out>.trajectory.ply,\n\
         which is the quickest way to see how badly the pose has drifted."
    )
}

fn gui(options: &Options) -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "gui")]
    return gui::run(options);

    #[cfg(not(feature = "gui"))]
    {
        let _ = options;
        Err("this build has no scanner window; rebuild with `--features gui`".into())
    }
}

async fn live(options: &Options) -> Result<(), Box<dyn Error>> {
    if options.viewer {
        #[cfg(feature = "gui")]
        {
            // `--viewer` is the live source, started immediately. It shares the
            // whole window with `scan gui`, so there is one preview to maintain.
            let mut options = options.clone();
            options.auto_start = true;
            return gui::run(&options);
        }

        #[cfg(not(feature = "gui"))]
        return Err("this build has no scanner window; rebuild with `--features gui`".into());
    }

    let mut scanner = run_live(options, |_, _| true).await?;
    finish_scan(&mut scanner, options)
}

/// The live capture loop behind the plain `live` command.
///
/// The scanner window does not use this loop: it needs pause, stop, preview and
/// export, so it runs its own worker in `gui::session`. `on_frame` runs after
/// each frame is integrated and returns whether to keep going; `live` always
/// returns `true`.
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
        let captured = capture.color();
        let color = captured.map(|captured| FrameColor {
            rgb: captured.rgb.as_slice(),
            depth: captured.depth.as_slice(),
            valid: captured.valid.as_slice(),
            exposure: captured.exposure,
            gain: captured.gain,
            gamma: captured.gamma,
        });
        let raw = captured.and_then(|c| c.raw.clone());
        let report = scanner.add_frame_with_raw_color(&frame, width, height, color, raw);
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
    let drain = if color_enabled {
        false
    } else {
        options.drain_color
    };
    let mut capture =
        capture::Capture::open(options.filters, color_enabled || drain, options.gpu).await?;
    let intrinsics = capture.intrinsics();
    let (width, height) = capture.dimensions();

    let mut writer = if color_enabled {
        recording::FrameWriter::create_with_raw_color(&out, intrinsics, width, height)?
    } else {
        recording::FrameWriter::create(&out, intrinsics, width, height)?
    };
    let mode_desc = if color_enabled {
        "depth + colour"
    } else {
        "depth"
    };
    println!(
        "[scan] recording {frames} {mode_desc} frames to {}",
        out.display()
    );

    let started = std::time::Instant::now();
    for index in 1..=frames {
        let frame = capture.next_frame().await?.to_vec();
        let captured = if color_enabled { capture.color() } else { None };
        let color = captured.map(|c| FrameColor {
            rgb: &c.rgb,
            depth: &c.depth,
            valid: &c.valid,
            exposure: c.exposure,
            gain: c.gain,
            gamma: c.gamma,
        });
        let raw = captured.and_then(|c| c.raw.as_ref());
        // Seconds since capture began, so the recording carries real timing.
        writer.write_frame_with_raw_color(started.elapsed().as_secs_f32(), &frame, color, raw)?;

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
        println!("[scan] this .k2df recording has no colour stream; painting is unavailable");
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
        let report = scanner.add_frame_with_raw_color(
            &frame.depth,
            recording.width,
            recording.height,
            color,
            frame.raw_color.clone(),
        );
        print_progress(index + 1, &report);
    }

    finish_scan(&mut scanner, options)
}

/// Localise a recording or the live sensor against a frozen prior point cloud.
///
/// This is a pure-localisation mode, reduced to the part a depth
/// camera can support: the map is fixed, each frame gets a global search for an
/// initial guess, ICP refines it, and the result is a trajectory. Nothing is
/// fused, so the prior's accuracy is preserved.
async fn localize(options: &Options) -> Result<(), Box<dyn Error>> {
    let map_path = options
        .map
        .clone()
        .ok_or("`localize` needs --map <prior.ply>")?;

    let prior_points = geom::mesh::read_points_ply(&map_path)?;
    let defaults = FrozenMapParams::default();
    let params = FrozenMapParams {
        // The map is a prior surface, not a TSDF. A 1 cm `--voxel` would spend
        // minutes estimating normals over millions of mesh vertices; four
        // centimetres is plenty for a localisation prior.
        voxel: options.tsdf.voxel_size.max(defaults.voxel),
        ..defaults
    };
    let map = FrozenMap::from_points(&prior_points, &params)
        .ok_or("prior map is too small after downsampling")?;

    println!(
        "[localize] prior {} points -> {} after downsampling from {}",
        prior_points.len(),
        map.len(),
        map_path.display()
    );
    let mut prior = Prior::new(prior_points, params, map);

    let mut pose = Isometry3::identity();
    let mut trajectory = Vec::new();
    let mut located = 0usize;
    let mut global_hits = 0usize;
    let total;

    if options.localize_live {
        let frames = options.frames.unwrap_or(DEFAULT_FRAMES);
        total = frames;

        let mut capture = capture::Capture::open(options.filters, false, options.gpu).await?;
        let intrinsics = capture.intrinsics();
        let (width, height) = capture.dimensions();
        println!("[localize] live {frames} frames at {width}x{height}");

        for index in 0..frames {
            let frame = capture.next_frame().await?.to_vec();
            let image = DepthImage::new(width, height, &frame);
            let mut points = image.to_points(&intrinsics);
            points.retain(|point| {
                point.z >= options.tsdf.min_depth && point.z <= options.tsdf.max_depth
            });
            if let Some(global) = prior.locate(index, &points, &mut pose, &mut trajectory) {
                located += 1;
                global_hits += global as usize;
            }
        }
    } else {
        let input = options
            .input
            .clone()
            .ok_or("`localize` needs --in <capture.k2df> or --live")?;
        let recording = recording::read(&input)?;
        println!(
            "[localize] {} frames of {}x{} from {}",
            recording.frames.len(),
            recording.width,
            recording.height,
            input.display()
        );

        let limit = options.frames.unwrap_or(usize::MAX);
        total = limit.min(recording.frames.len());

        for (index, frame) in recording.frames.iter().take(limit).enumerate() {
            let image = DepthImage::new(recording.width, recording.height, &frame.depth);
            let mut points = image.to_points(&recording.intrinsics);
            points.retain(|point| {
                point.z >= options.tsdf.min_depth && point.z <= options.tsdf.max_depth
            });
            if let Some(global) = prior.locate(index, &points, &mut pose, &mut trajectory) {
                located += 1;
                global_hits += global as usize;
            }
        }
    }

    if let Some(out) = &options.out {
        geom::mesh::save_points_ply(out, &trajectory)?;
        println!(
            "[localize] wrote {} poses to {}",
            trajectory.len(),
            out.display()
        );
    }

    println!(
        "[localize] located {} of {} frames ({:.0}%, {global_hits} after a global search)",
        located,
        total,
        if total == 0 {
            0.0
        } else {
            located as f32 / total as f32 * 100.0
        }
    );
    Ok(())
}

/// A frozen prior map that may be stored mirrored.
///
/// `scan` flips X on export by default, so a PLY written without `--mirror` is
/// the mirror image of the sensor frame that `localize` compares live depth
/// against. Rather than make the user remember that, the map is used as given
/// until the first frame is located; a frame that fails before that point is
/// retried against the other orientation, built lazily. The first orientation
/// that locates wins and is kept for the rest of the run.
struct Prior {
    points: Vec<Vector3<f32>>,
    params: FrozenMapParams,
    map: FrozenMap,
    flipped: Option<FrozenMap>,
    use_flipped: bool,
    located_any: bool,
    flip_attempts: usize,
}

/// How many failed frames may trigger a try of the other orientation. A bad
/// frame should not decide the orientation, but a genuinely lost scan should
/// not keep paying for a second solve forever.
const MAX_FLIP_ATTEMPTS: usize = 5;

impl Prior {
    fn new(points: Vec<Vector3<f32>>, params: FrozenMapParams, map: FrozenMap) -> Self {
        Self {
            points,
            params,
            map,
            flipped: None,
            use_flipped: false,
            located_any: false,
            flip_attempts: 0,
        }
    }

    fn active(&self) -> &FrozenMap {
        match (&self.flipped, self.use_flipped) {
            (Some(flipped), true) => flipped,
            _ => &self.map,
        }
    }

    /// Localise one frame, auto-detecting a mirrored prior before the first hit.
    fn locate(
        &mut self,
        index: usize,
        points: &[Vector3<f32>],
        pose: &mut Isometry3<f32>,
        trajectory: &mut Vec<Vector3<f32>>,
    ) -> Option<bool> {
        let mut found = self.active().localize(points, *pose, &self.params);

        if found.is_none() && !self.located_any && self.flip_attempts < MAX_FLIP_ATTEMPTS {
            self.flip_attempts += 1;
            if self.flipped.is_none() {
                let mirrored: Vec<Vector3<f32>> = self
                    .points
                    .iter()
                    .map(|point| Vector3::new(-point.x, point.y, point.z))
                    .collect();
                self.flipped = FrozenMap::from_points(&mirrored, &self.params);
            }
            let other = !self.use_flipped;
            let candidate = if other {
                self.flipped.as_ref()
            } else {
                Some(&self.map)
            };
            if let Some(result) =
                candidate.and_then(|map| map.localize(points, *pose, &self.params))
            {
                self.use_flipped = other;
                println!(
                    "[localize] prior matches only when {}; using it that way",
                    if other {
                        "mirrored in X (it was exported without --mirror)"
                    } else {
                        "as stored"
                    }
                );
                found = Some(result);
            }
        }

        match found {
            Some(found) => {
                self.located_any = true;
                *pose = found.pose;
                trajectory.push(pose.translation.vector);
                let global = found.score > 0.0;
                println!(
                    "[localize] {:5} t=({:+.3},{:+.3},{:+.3}) inliers {:.2} rmse {:.3}{}",
                    index + 1,
                    pose.translation.x,
                    pose.translation.y,
                    pose.translation.z,
                    found.inlier_ratio,
                    found.rmse,
                    if global { " global" } else { "" }
                );
                Some(global)
            }
            None => {
                println!("[localize] {:5} lost; keeping the last pose", index + 1);
                None
            }
        }
    }
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
    let truth_path: f32 = aligned
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).norm())
        .sum();
    let lived_path: f32 = lived
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).norm())
        .sum();

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
    let mut loops = LoopClosureConfig::default();
    loops.loop_target_keyframes = options.loop_window;
    loops.background = options.loop_worker;
    loops.global_search = options.loop_global;
    loops.min_match_score = options.match_score;

    ScannerConfig {
        tsdf: options.tsdf,
        loop_closure: options.loop_closure.then_some(loops),
        odometry: OdometryConfig {
            frame_to_model: options.frame_to_model,
            relocalize: options.relocalize,
            relocalize_keyframes: options.relocalize_keyframes,
            relocalize_global_search: options.relocalize_global,
            ..OdometryConfig::default()
        },
        submaps: options.submaps,
        submap_max_frames: options.submap_max_frames,
        submap_max_translation: options.submap_max_translation,
        online_loops: options.online_loops,
        incremental_rebuild: options.incremental_rebuild,
        fusion_filter: options.fusion_filter.then(MotionFilterConfig::default),
        tsdf_chunk_frames: options.tsdf_chunk_frames,
        submap_graph: options.submap_graph,
        graph_worker: options.graph_worker,
        coloring: ColoringParams {
            mode: options.color_mode,
            depth_tolerance: options.color_depth_tolerance,
            ..ColoringParams::default()
        },
        ..ScannerConfig::default()
    }
}

/// What a finished export wrote, for callers that display it.
///
/// The CLI ignores this; the scanner window uses it to report the exported mesh
/// and the dataset hand-off without scraping stdout.
#[derive(Debug)]
pub(crate) struct FinishSummary {
    /// The mesh path actually written.
    pub output: PathBuf,
    /// Vertices and triangles in the exported mesh, or `None` when there was
    /// no surface to extract.
    pub mesh: Option<(usize, usize)>,
    /// Present when `--dataset` was set and a surface existed.
    pub dataset: Option<dataset::DatasetReport>,
    /// Pre-training sanity check, including unmerged-submap warnings.
    pub quality: ScanQuality,
    /// Loop edges kept after outlier trimming.
    pub loops: usize,
    /// Verified loops discarded because the correction was implausible.
    pub loops_dropped: usize,
    /// Spatial ICP attempts spent by the loop finder.
    pub loop_attempts: usize,
    /// What `--level` did, so a caller with no terminal can say so.
    pub level: LevelOutcome,
}

/// The result of the optional export-time levelling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum LevelOutcome {
    /// Levelling was not requested, or there was no surface to level.
    Off,
    /// The model was rotated by this many degrees.
    Applied { tilt_degrees: f32 },
    /// Levelling was requested but no large horizontal surface was found, so
    /// the model was written exactly as scanned.
    NotFound,
}

/// Level an exported model in place and say what happened.
///
/// Estimated from, and applied to, the mesh as it will be written, so the mesh
/// and the trajectory (which the caller rotates with the returned [`Leveling`])
/// stay in the same frame. `None` means the model was left exactly as scanned.
fn level_export(exported: &mut ExportMesh) -> Option<Leveling> {
    let params = LevelParams::default();
    let found = match exported {
        ExportMesh::Geometry(mesh, _) => level::estimate(&mesh.vertices, &params),
        ExportMesh::Atlas(textured, _) => level::estimate(&textured.positions, &params),
    };

    let Some(leveling) = found else {
        println!(
            "[scan] level: no large horizontal surface within {:.0} degrees of level; \
             left as scanned",
            params.max_tilt.to_degrees()
        );
        return None;
    };

    match exported {
        ExportMesh::Geometry(mesh, _) => mesh.rotate(&leveling.rotation),
        ExportMesh::Atlas(textured, _) => textured.rotate(&leveling.rotation),
    }
    println!(
        "[scan] level: rotated {:.1} degrees to put the dominant floor/ceiling plane \
         horizontal ({:.1}% of sampled points)",
        leveling.tilt.to_degrees(),
        leveling.inlier_fraction * 100.0
    );
    Some(leveling)
}

/// Close loops, then write the mesh. Live, replay, and the scanner window all
/// end here, so closing the window still corrects the trajectory before export.
pub(crate) fn finish_scan(scanner: &mut Scanner, options: &Options) -> Result<(), Box<dyn Error>> {
    finish_scan_with_report(scanner, options).map(|_| ())
}

/// The same ending as [`finish_scan`], but returns what was written so the
/// window can show a dataset result card instead of reading stdout.
pub(crate) fn finish_scan_with_report(
    scanner: &mut Scanner,
    options: &Options,
) -> Result<FinishSummary, Box<dyn Error>> {
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
    let graph = if report.submap_nodes > 0 {
        format!("{} submap nodes", report.submap_nodes)
    } else {
        format!("{} keyframes", report.keyframe_nodes)
    };

    if report.loop_edges == 0 && report.dropped_loops == 0 {
        println!("loop closure: no revisits verified over {graph}, trajectory unchanged");
        return;
    }

    if report.loop_edges == 0 {
        println!(
            "loop closure: all {} verified loops were discarded over {graph}",
            report.dropped_loops
        );
    } else {
        println!(
            "loop closure: {} revisits over {graph}, cost {:.3} -> {:.3} in {} iterations",
            report.loop_edges, report.cost_before, report.cost_after, report.iterations
        );
    }
    println!(
        "  {} graph nodes, {} odometry edges",
        report.nodes, report.odometry_edges
    );
    if !report.converged {
        println!("  optimiser stopped before convergence");
    }
    println!(
        "  correction applied {}",
        if report.online {
            "during the scan"
        } else {
            "at the end"
        }
    );
    if report.dropped_loops > 0 {
        println!(
            "  dropped {} weak loop(s) after an implausible correction",
            report.dropped_loops
        );
    }
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
    let relocalized = if report.track.relocalized {
        "  RELOCALIZED"
    } else {
        ""
    };
    let submap = if report.new_submap {
        match report.submap_reason {
            Some(reason) => format!("  SUBMAP {} ({reason})", report.submap + 1),
            None => format!("  SUBMAP {}", report.submap + 1),
        }
    } else {
        String::new()
    };

    // `move` is the recovered sensor motion for this frame and `iters` the total
    // ICP iterations spent on it: together they are the quickest way to spot a
    // jump that the acceptance gates should have caught, or a solve that is
    // burning its whole budget without settling.
    println!(
        "[scan] {:>4}  pts {:>6}  inliers {:>5.1}%  rmse {:>5.1} mm  move {:>5.1} cm  iters {:>3}  pos [{:>6.3} {:>6.3} {:>6.3}]  {outcome}{recovered}{relocalized}{submap}",
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

pub(crate) fn finish(
    scanner: &Scanner,
    options: &Options,
) -> Result<FinishSummary, Box<dyn Error>> {
    println!();
    println!(
        "[scan] frames: {} (fused {}, skipped {}, submaps {})",
        scanner.frames(),
        scanner.fused(),
        scanner.rejected(),
        scanner.submaps()
    );
    if scanner.submaps() > 1 {
        let starts = scanner
            .submap_starts()
            .iter()
            .map(|index| (index + 1).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        println!("[scan] submap starts: {starts}");
    }

    let frames = scanner.frames().max(1) as f64;
    println!(
        "[scan] time per frame: tracking {:.0} ms, fusion {:.0} ms, submap {:.0} ms",
        scanner.total_tracking().as_secs_f64() * 1000.0 / frames,
        scanner.total_fusion().as_secs_f64() * 1000.0 / frames,
        scanner.total_submap().as_secs_f64() * 1000.0 / frames,
    );
    print_color_views(scanner);
    println!(
        "[scan] memory: {} TSDF blocks, {:.1} MB total ({:.1} MB submap surfaces)",
        scanner.block_count(),
        scanner.allocated_bytes() as f64 / 1e6,
        scanner.submap_surface_bytes() as f64 / 1e6,
    );

    // Final position is the quickest sanity check on drift: for a loop that
    // returns to its start, this should be back near the origin.
    let final_position = scanner.pose().translation.vector;
    let position_note = if scanner.submaps() > 1 {
        " (current submap origin guess)"
    } else {
        ""
    };
    println!(
        "[scan] final position [{:.3} {:.3} {:.3}] m, {:.3} m from origin{position_note}",
        final_position.x,
        final_position.y,
        final_position.z,
        final_position.norm()
    );

    let quality = scanner.quality();
    if quality.trainable() {
        println!("[scan] quality: trainable");
    } else {
        println!(
            "[scan] quality: NOT trainable ({} warning(s))",
            quality.warnings.len()
        );
        for warning in &quality.warnings {
            println!("[scan] warning: {warning}");
        }
    }

    let (loops, loops_dropped) = scanner
        .closure()
        .map(|report| (report.loop_edges, report.dropped_loops))
        .unwrap_or((scanner.loops(), 0));
    let loop_attempts = scanner.loop_attempts();

    let out = options
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("scan.ply"));
    let format = MeshFormat::from_path(&out)?;

    println!("[scan] extracting surface ...");
    // The atlas is built in the volume's frame. Mirroring happens inside
    // `export_mesh`, after that, so positions, normals and winding all flip
    // and the texture coordinates stay on the image that was just painted.
    let Some(mut exported) = scanner.export_mesh(options.unmirror, format.uses_texture_atlas())
    else {
        println!("[scan] no surface extracted - was anything in range?");
        return Ok(FinishSummary {
            output: out,
            mesh: None,
            dataset: None,
            quality: quality.clone(),
            loops,
            loops_dropped,
            loop_attempts,
            level: LevelOutcome::Off,
        });
    };

    // After the mirror, in whatever frame the file will be written in: `y` is
    // untouched by the X flip, so "level" means the same thing either way.
    let leveling = if options.level {
        level_export(&mut exported)
    } else {
        None
    };

    let mesh_summary = match &exported {
        ExportMesh::Geometry(mesh, coloring) => {
            if let Some((min, max)) = mesh.bounds() {
                print_mesh_extent(mesh.vertex_count(), mesh.triangle_count(), min, max);
            }
            if let Some(report) = coloring {
                println!("{}", format_paint(report));
            }
            Some((mesh.vertex_count(), mesh.triangle_count()))
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
            Some((textured.positions.len(), textured.triangle_count()))
        }
    };

    match exported {
        ExportMesh::Geometry(mesh, _) => export::save_mesh(&mesh, &out)?,
        ExportMesh::Atlas(textured, _) => export::save_textured(&textured, &out)?,
    }
    println!("[scan] wrote {}", out.display());

    let trajectory = out.with_extension("trajectory.ply");
    let mut trajectory_points: Vec<Vector3<f32>> = if options.unmirror {
        scanner
            .trajectory()
            .iter()
            .map(|p| Vector3::new(-p.x, p.y, p.z))
            .collect()
    } else {
        scanner.trajectory().to_vec()
    };
    if let Some(leveling) = &leveling {
        for point in &mut trajectory_points {
            *point = leveling.apply(point);
        }
    }
    geom::mesh::save_points_ply(&trajectory, &trajectory_points)?;
    println!(
        "[scan] wrote {} ({} camera positions)",
        trajectory.display(),
        trajectory_points.len()
    );

    let mut dataset_report = None;
    if let Some(dir) = &options.dataset {
        // Written last, and after the poses are final: `close_loops` ran in
        // `finish_scan`, and a correction moves the views onto the rebuilt
        // trajectory. A dataset assembled before that would carry the drifted
        // poses the trainer is most sensitive to.
        println!("[scan] writing dataset to {} ...", dir.display());
        let report = dataset::write_dataset(scanner, dir, options.unmirror)?;

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
            "[scan]   {}",
            if options.unmirror {
                "images, poses, principal point and init.ply are mirrored together to match \
                 the un-mirrored mesh (--mirror keeps the sensor frame)"
            } else {
                "images, poses and init.ply are in the sensor's mirrored frame, as --mirror asked"
            }
        );
        dataset_report = Some(report);
    }

    Ok(FinishSummary {
        output: out,
        mesh: mesh_summary,
        dataset: dataset_report,
        quality,
        loops,
        loops_dropped,
        loop_attempts,
        level: match (options.level, &leveling) {
            (true, Some(leveling)) => LevelOutcome::Applied {
                tilt_degrees: leveling.tilt.to_degrees(),
            },
            (true, None) => LevelOutcome::NotFound,
            (false, _) => LevelOutcome::Off,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use geom::Intrinsics;
    use geom::mesh::Mesh;

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
            error
                .to_string()
                .contains("--dataset cannot be used with `record`"),
            "{error}"
        );
    }

    #[test]
    fn dataset_is_allowed_for_replay_at_parse_time() {
        let (command, options) = parse_args(words(&[
            "replay",
            "--in",
            "capture.k2df",
            "--dataset",
            "out/d",
        ]))
        .expect("parse");
        assert!(matches!(command, Command::Replay));
        assert_eq!(options.dataset, Some(PathBuf::from("out/d")));
    }

    #[test]
    fn relocalize_is_opt_in_and_reaches_the_scanner_config() {
        let (_, options) = parse_args(words(&["replay", "--in", "capture.k2df"])).expect("parse");
        assert!(!options.relocalize);

        let (_, options) =
            parse_args(words(&["replay", "--in", "capture.k2df", "--relocalize"])).expect("parse");
        assert!(options.relocalize);
        assert!(scanner_config(&options).odometry.relocalize);
    }

    #[test]
    fn submaps_are_on_by_default_and_can_be_disabled() {
        let (_, options) = parse_args(words(&["replay", "--in", "capture.k2df"])).expect("parse");
        assert!(options.submaps);
        assert!(scanner_config(&options).submaps);

        let (_, options) =
            parse_args(words(&["replay", "--in", "capture.k2df", "--no-submaps"])).expect("parse");
        assert!(!options.submaps);
        assert!(!scanner_config(&options).submaps);
    }

    #[test]
    fn lio_style_loop_flags_reach_the_scanner_config() {
        let (_, options) = parse_args(words(&["replay", "--in", "capture.k2df"])).expect("parse");
        assert!(!options.online_loops);
        assert!(!options.loop_worker);
        assert!(!options.incremental_rebuild);
        assert_eq!(
            options.loop_window,
            LoopClosureConfig::default().loop_target_keyframes
        );
        assert_eq!(
            options.relocalize_keyframes,
            OdometryConfig::default().relocalize_keyframes
        );

        let (_, options) = parse_args(words(&[
            "replay",
            "--in",
            "capture.k2df",
            "--online-loops",
            "--loop-worker",
            "--incremental-rebuild",
            "--loop-window",
            "4",
            "--relocalize-keyframes",
            "8",
        ]))
        .expect("parse");

        // The three loop behaviours imply loop closure rather than silently
        // doing nothing when the base flag is missing.
        assert!(options.loop_closure);
        assert!(options.online_loops);
        assert!(options.loop_worker);
        assert!(options.incremental_rebuild);

        let config = scanner_config(&options);
        assert!(config.online_loops);
        assert!(config.incremental_rebuild);
        let loops = config.loop_closure.expect("loop config");
        assert!(loops.background);
        assert_eq!(loops.loop_target_keyframes, 4);
        assert_eq!(config.odometry.relocalize_keyframes, 8);
        assert!(config.odometry.relocalize, "the ring needs relocalisation");
    }

    #[test]
    fn advanced_loop_flags_reach_the_scanner_config() {
        let (_, options) = parse_args(words(&[
            "replay",
            "--in",
            "capture.k2df",
            "--fusion-filter",
            "--tsdf-chunk",
            "40",
            "--relocalize-global",
            "--loop-global",
            "--match-score",
            "0.6",
            "--submap-graph",
            "--submap-frames",
            "24",
            "--submap-metres",
            "1.5",
            "--graph-worker",
        ]))
        .expect("parse");

        assert!(options.fusion_filter);
        assert_eq!(options.tsdf_chunk_frames, 40);
        assert!(options.relocalize_global);
        assert!(options.loop_global);
        assert!((options.match_score - 0.6).abs() < 1e-6);
        assert!(options.submap_graph);
        assert_eq!(options.submap_max_frames, 24);
        assert!((options.submap_max_translation - 1.5).abs() < 1e-6);
        assert!(options.graph_worker);
        assert!(options.online_loops, "--graph-worker implies online loops");

        let config = scanner_config(&options);
        assert!(config.fusion_filter.is_some());
        assert_eq!(config.tsdf_chunk_frames, 40);
        assert!(config.odometry.relocalize_global_search);
        assert!(config.submap_graph);
        assert_eq!(config.submap_max_frames, 24);
        assert!((config.submap_max_translation - 1.5).abs() < 1e-6);
        assert!(config.graph_worker);
        assert!(config.online_loops);
        let loops = config.loop_closure.expect("loop config");
        assert!(loops.global_search);
        assert!((loops.min_match_score - 0.6).abs() < 1e-6);
    }

    #[test]
    fn graph_worker_uses_the_configured_graph() {
        let (_, options) = parse_args(words(&["replay", "--in", "capture.k2df", "--graph-worker"]))
            .expect("parse");

        assert!(options.graph_worker);
        assert!(
            !options.submap_graph,
            "--graph-worker should not force submap nodes"
        );
        assert!(options.online_loops);

        let config = scanner_config(&options);
        assert!(config.graph_worker);
        assert!(!config.submap_graph, "the keyframe graph is the default");
    }

    #[test]
    fn localize_requires_a_map_and_a_recording() {
        let error = parse_args(words(&["localize", "--in", "capture.k2df"]))
            .expect_err("a map is required");
        assert!(error.to_string().contains("--map"), "{error}");

        let error = parse_args(words(&["localize", "--map", "prior.ply"]))
            .expect_err("an input is required");
        assert!(error.to_string().contains("--in"), "{error}");

        let (command, options) = parse_args(words(&[
            "localize",
            "--in",
            "capture.k2df",
            "--map",
            "prior.ply",
        ]))
        .expect("parse");
        assert!(matches!(command, Command::Localize));
        assert_eq!(
            options.map.as_deref(),
            Some(std::path::Path::new("prior.ply"))
        );
        assert!(!options.localize_live);

        let (_, options) =
            parse_args(words(&["localize", "--live", "--map", "prior.ply"])).expect("live parse");
        assert!(options.localize_live);
        assert!(options.input.is_none());

        let error = parse_args(words(&[
            "localize",
            "--in",
            "capture.k2df",
            "--live",
            "--map",
            "prior.ply",
        ]))
        .expect_err("--in and --live are exclusive");
        assert!(error.to_string().contains("not both"), "{error}");

        let error = parse_args(words(&["replay", "--in", "capture.k2df", "--live"]))
            .expect_err("--live only belongs to localize");
        assert!(error.to_string().contains("--live"), "{error}");

        let error = parse_args(words(&[
            "replay",
            "--in",
            "capture.k2df",
            "--map",
            "prior.ply",
        ]))
        .expect_err("--map only belongs to localize");
        assert!(error.to_string().contains("--map"), "{error}");
    }

    #[test]
    fn a_zero_relocalization_ring_is_rejected() {
        let error = parse_args(words(&[
            "replay",
            "--in",
            "capture.k2df",
            "--relocalize-keyframes",
            "0",
        ]))
        .expect_err("zero must fail");
        assert!(error.to_string().contains("at least 1"), "{error}");
    }

    #[test]
    fn usage_documents_the_lio_style_flags() {
        let usage = usage_text();
        for flag in [
            "--online-loops",
            "--loop-worker",
            "--incremental-rebuild",
            "--loop-window",
            "--relocalize-keyframes",
        ] {
            assert!(
                usage.contains(flag),
                "{flag} is missing from the usage text"
            );
        }
    }

    #[test]
    fn usage_documents_the_advanced_loop_flags() {
        let usage = usage_text();
        for flag in [
            "--fusion-filter",
            "--tsdf-chunk",
            "--relocalize-global",
            "--loop-global",
            "--match-score",
            "--submap-graph",
            "--submap-frames",
            "--submap-metres",
            "--graph-worker",
            "--map",
            "--live",
            "scan localize",
        ] {
            assert!(
                usage.contains(flag),
                "{flag} is missing from the usage text"
            );
        }
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
        // And that a depth-only recording cannot be turned into one after the
        // fact, while a colour recording can.
        assert!(usage.contains("depth-only recording cannot"), "{usage}");
        assert!(usage.contains("colour recording"), "{usage}");
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
    fn gui_flags_are_parsed_and_carry_a_source() {
        let (command, options) = parse_args(words(&[
            "gui",
            "--synthetic",
            "--auto-start",
            "--preview-every",
            "5",
            "--playback-fps",
            "7.5",
        ]))
        .expect("parse");

        assert!(matches!(command, Command::Gui));
        assert!(options.synthetic);
        assert!(options.auto_start);
        assert_eq!(options.preview_every, 5);
        assert!((options.playback_fps - 7.5).abs() < 1e-6);
    }

    #[test]
    fn gui_source_flags_conflict_is_rejected() {
        let error = parse_args(words(&["gui", "--synthetic", "--in", "capture.k2df"]))
            .expect_err("two sources");
        assert!(
            error.to_string().contains("different GUI sources"),
            "{error}"
        );
    }

    #[test]
    fn synthetic_outside_the_window_is_rejected() {
        let error = parse_args(words(&["live", "--synthetic"])).expect_err("not the gui");
        assert!(
            error.to_string().contains("only applies to `scan gui`"),
            "{error}"
        );
    }

    #[test]
    fn bad_gui_preview_and_playback_values_are_rejected() {
        let every = parse_args(words(&["gui", "--preview-every", "0"])).expect_err("zero");
        assert!(every.to_string().contains("at least 1"), "{every}");

        for value in ["-1", "nan", "inf"] {
            let fps = parse_args(words(&["gui", "--playback-fps", value])).expect_err(value);
            assert!(fps.to_string().contains("zero or more"), "{value}: {fps}");
        }
    }

    #[test]
    fn usage_documents_the_window_and_its_options() {
        let usage = usage_text();
        assert!(usage.contains("scan gui"), "{usage}");
        assert!(usage.contains("scan synth"), "{usage}");
        assert!(usage.contains("analytic room"), "{usage}");
        assert!(usage.contains("Open the scanner window"), "{usage}");
        assert!(usage.contains("--synthetic"), "{usage}");
        assert!(usage.contains("--auto-start"), "{usage}");
        assert!(usage.contains("--preview-every N"), "{usage}");
        assert!(usage.contains("--playback-fps F"), "{usage}");
        assert!(usage.contains("needs no sensor"), "{usage}");
    }

    #[test]
    fn finish_scan_closes_loops_before_an_empty_mesh_returns() {
        // No surface, so `finish` returns before it writes a file. Closure is
        // recorded only when loop closure actually ran, which has to happen
        // first: the window, live, and replay paths all end in `finish_scan`.
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


    /// The sensor-frame points of the first synthetic view, as the prior a user
    /// would have scanned of the same room.
    fn synthetic_prior() -> (Vec<Vector3<f32>>, Vec<Vector3<f32>>, FrozenMapParams) {
        let intrinsics = synthetic::intrinsics();
        let shot = &synthetic::shots(2)[0];
        let rendered = synthetic::render(shot, &intrinsics);
        let image = DepthImage::new(synthetic::WIDTH, synthetic::HEIGHT, &rendered.depth);
        let points = image.to_points(&intrinsics);
        let params = FrozenMapParams {
            voxel: 0.04,
            ..FrozenMapParams::default()
        };
        (points.clone(), points, params)
    }

    fn mirror_x(points: &[Vector3<f32>]) -> Vec<Vector3<f32>> {
        points
            .iter()
            .map(|point| Vector3::new(-point.x, point.y, point.z))
            .collect()
    }

    #[test]
    fn a_prior_in_the_sensor_frame_is_used_as_stored() {
        let (stored, query, params) = synthetic_prior();
        let map = FrozenMap::from_points(&stored, &params).expect("map");
        let mut prior = Prior::new(stored, params, map);

        let mut pose = Isometry3::identity();
        let mut trajectory = Vec::new();
        assert!(prior.locate(0, &query, &mut pose, &mut trajectory).is_some());
        assert!(!prior.use_flipped, "a matching prior must not be flipped");
        assert!(prior.flipped.is_none(), "the flipped copy should stay unbuilt");
        assert_eq!(trajectory.len(), 1);
    }

    #[test]
    fn a_mirrored_prior_is_detected_and_flipped_back() {
        // What `scan` writes by default: the sensor frame, mirrored in X.
        let (sensor, query, params) = synthetic_prior();
        let stored = mirror_x(&sensor);
        let map = FrozenMap::from_points(&stored, &params).expect("map");
        let mut prior = Prior::new(stored, params, map);

        let mut pose = Isometry3::identity();
        let mut trajectory = Vec::new();
        for index in 0..3 {
            assert!(
                prior
                    .locate(index, &query, &mut pose, &mut trajectory)
                    .is_some(),
                "frame {index} was not located against the mirrored prior"
            );
        }
        assert!(prior.use_flipped, "the mirrored prior should be flipped");
        assert_eq!(trajectory.len(), 3);
        assert!(
            pose.translation.vector.norm() < 0.05,
            "the pose should stay near the origin: {:?}",
            pose.translation.vector
        );
    }

    #[test]
    fn a_prior_that_never_matches_stops_trying_the_flip() {
        let (sensor, _, params) = synthetic_prior();
        let map = FrozenMap::from_points(&sensor, &params).expect("map");
        let mut prior = Prior::new(sensor, params, map);

        // A query nowhere near the map: a plane far outside the working volume.
        let nothing: Vec<Vector3<f32>> = (0..400)
            .map(|index| Vector3::new((index % 20) as f32 * 0.05, (index / 20) as f32 * 0.05, 50.0))
            .collect();
        let mut pose = Isometry3::identity();
        let mut trajectory = Vec::new();
        for index in 0..(MAX_FLIP_ATTEMPTS + 3) {
            let _ = prior.locate(index, &nothing, &mut pose, &mut trajectory);
        }
        assert!(trajectory.is_empty());
        assert!(prior.flip_attempts <= MAX_FLIP_ATTEMPTS);
    }


    #[test]
    fn level_is_opt_in_and_parses() {
        let (_, options) = parse_args(words(&["replay", "--in", "capture.k2df"])).expect("parse");
        assert!(!options.level, "levelling must be off unless asked for");

        let (_, options) =
            parse_args(words(&["replay", "--in", "capture.k2df", "--level"])).expect("parse");
        assert!(options.level);
    }

    /// A floor and a wall as a vertex-only mesh, with the sensor pitched by
    /// `pitch` radians so the floor is not perpendicular to `y`.
    fn tilted_floor_and_wall(pitch: f32) -> Mesh {
        let rotation = nalgebra::Rotation3::from_euler_angles(pitch, 0.0, 0.0);
        let mut vertices = Vec::new();
        for i in 0..80 {
            for j in 0..80 {
                let u = -2.0 + 4.0 * i as f32 / 79.0;
                let v = 0.5 + 4.0 * j as f32 / 79.0;
                vertices.push(rotation * Vector3::new(u, 1.4, v));
                vertices.push(rotation * Vector3::new(u, -1.2 + 2.6 * j as f32 / 79.0, 4.5));
            }
        }
        Mesh {
            vertices,
            triangles: Vec::new(),
            colors: None,
        }
    }

    #[test]
    fn level_export_flattens_a_tilted_floor_and_reports_the_rotation() {
        let mut exported = ExportMesh::Geometry(tilted_floor_and_wall(8.0f32.to_radians()), None);

        let leveling = level_export(&mut exported).expect("a floor is present");
        assert!((leveling.tilt.to_degrees() - 8.0).abs() < 0.7, "{}", leveling.tilt.to_degrees());

        let ExportMesh::Geometry(mesh, _) = &exported else {
            panic!("the export kind must not change");
        };
        // The floor vertices (every other one) now sit at one height.
        let floor: Vec<f32> = mesh.vertices.iter().step_by(2).map(|v| v.y).collect();
        let low = floor.iter().cloned().fold(f32::MAX, f32::min);
        let high = floor.iter().cloned().fold(f32::MIN, f32::max);
        assert!(high - low < 0.06, "the floor still spans {:.3} m", high - low);
        assert!((floor[0] - 1.4).abs() < 0.08);
    }

    #[test]
    fn level_export_leaves_a_scan_without_a_floor_untouched() {
        // A single vertical wall: nothing horizontal to level against.
        let wall: Vec<Vector3<f32>> = (0..60 * 60)
            .map(|i| Vector3::new((i % 60) as f32 * 0.05, -1.2 + (i / 60) as f32 * 0.05, 3.0))
            .collect();
        let before = wall.clone();
        let mut exported = ExportMesh::Geometry(
            Mesh {
                vertices: wall,
                triangles: Vec::new(),
                colors: None,
            },
            None,
        );

        assert!(level_export(&mut exported).is_none());
        let ExportMesh::Geometry(mesh, _) = &exported else {
            panic!("the export kind must not change");
        };
        assert_eq!(mesh.vertices, before, "an unlevelled scan must be unchanged");
    }


    /// Run a short synthetic scan through the real export path and report what
    /// levelling did, so the GUI's "levelled" / "not found" log lines have a
    /// guarantee behind them.
    fn export_synthetic(level: bool, name: &str) -> FinishSummary {
        let intrinsics = synthetic::intrinsics();
        let mut scanner = Scanner::new(intrinsics, scanner_config(&Options::default()));
        for shot in synthetic::shots(8) {
            let rendered = synthetic::render(&shot, &intrinsics);
            scanner.add_frame(&rendered.depth, synthetic::WIDTH, synthetic::HEIGHT);
        }

        let dir = std::env::temp_dir().join(format!("scan-level-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut options = Options::default();
        options.level = level;
        options.out = Some(dir.join("room.ply"));
        let summary = finish_scan_with_report(&mut scanner, &options).expect("export");
        std::fs::remove_dir_all(&dir).expect("remove temp dir");
        summary
    }

    #[test]
    fn export_reports_that_levelling_was_not_requested() {
        assert_eq!(export_synthetic(false, "off").level, LevelOutcome::Off);
    }

    #[test]
    fn export_reports_a_levelling_outcome_when_requested() {
        // The synthetic room is 2.4 m tall, the camera sits at mid-height with a
        // narrow field of view, and the far wall is only 2.4 m away, so the floor
        // and ceiling barely reach the edge of the image. There is no large
        // horizontal surface to find, and the right answer is `NotFound`: the
        // request reached the exporter, nothing was moved, and the caller is
        // told. (`Applied` is covered by `level_export_flattens_a_tilted_floor`.)
        assert_eq!(export_synthetic(true, "on").level, LevelOutcome::NotFound);
    }
}
