//! A trainer child process, kept off the UI thread.
//!
//! The scanner window owns the dataset hand-off but not the trainer. This
//! module is the bridge: find Brush, build its command for whichever CLI
//! revision is installed, run it as a child, stream its output into a bounded
//! log, track how far training has got, and let Cancel actually kill it.
//!
//! Progress comes from two sources, and the window degrades gracefully when
//! either is missing, because Brush's log format is not a stable API:
//!
//! * `export_<iteration>.ply` files in the output directory. These are the
//!   reliable signal: the iteration is in the file name, and each file is a
//!   complete splat model the window can show as a training preview.
//! * `Refine iter N, M splats.` log lines, which Brush prints at `info` level
//!   and which give a finer iteration count and the live splat count. A
//!   different log format simply leaves those fields empty.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime};

/// How many lines of trainer output to keep. Brush can be noisy; the operator
/// needs the tail, not the whole run in memory.
const MAX_LOG_LINES: usize = 240;

/// How often the output directory is scanned for new preview exports.
const SCAN_EVERY: Duration = Duration::from_millis(400);

/// An export is only trusted once it has stopped changing for this long, so the
/// window never tries to read a splat file Brush is still writing. The final
/// scan, after the process exits, ignores the rule: nothing is writing then.
const SETTLE: Duration = Duration::from_millis(1200);

/// Executable names Brush ships under, in the order they are preferred. The
/// 0.3.0 release archive contains `brush_app`; the project's README says
/// `brush`; older notes in this repo say `brush-cli`.
const BRUSH_NAMES: [&str; 3] = ["brush", "brush_app", "brush-cli"];

/// Where Brush's releases are published.
pub(crate) const BRUSH_RELEASES_URL: &str = "https://github.com/ArthurBrussee/brush/releases";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TrainerStatus {
    Idle,
    Running,
    Finished,
    Failed,
    Cancelled,
}

impl TrainerStatus {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "training",
            Self::Finished => "finished",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub(crate) fn active(self) -> bool {
        matches!(self, Self::Running)
    }
}

/// An installed Brush.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BrushInstall {
    pub path: PathBuf,
    /// First line of `--version`, or empty when the binary would not say.
    pub version: String,
}

/// Find Brush on `PATH`.
pub(crate) fn find_brush() -> Option<BrushInstall> {
    let dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|path| std::env::split_paths(&path).collect())
        .unwrap_or_default();
    find_brush_in(&dirs)
}

/// Find Brush in `dirs`, preferring earlier names, then earlier directories.
pub(crate) fn find_brush_in(dirs: &[PathBuf]) -> Option<BrushInstall> {
    for name in BRUSH_NAMES {
        for dir in dirs {
            let path = dir.join(name);
            if is_executable_file(&path) {
                let version = run_capture(&path, &["--version"], Duration::from_secs(3))
                    .and_then(|text| {
                        text.lines()
                            .map(str::trim)
                            .find(|line| !line.is_empty())
                            .map(str::to_string)
                    })
                    .unwrap_or_default();
                return Some(BrushInstall { path, version });
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable_file(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

/// Run `program args` and return its combined stdout and stderr, or `None` if
/// it could not start or did not finish inside `timeout`. A program that does
/// not exit is killed: this is used to probe binaries found on `PATH`, and a
/// probe must never hang the window.
///
/// The output is collected on reader threads that report over a channel, and
/// they are never joined. A program can leave a child of its own holding the
/// pipes open after it is killed or has exited (a shell script that runs
/// `sleep` does exactly that), and joining a reader would then wait for that
/// grandchild. A reader that is not heard from in time is simply abandoned; it
/// ends when the pipe finally closes.
fn run_capture(program: &Path, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;

    // Tagged with their stream, so the result reads stdout then stderr whatever
    // order the reader threads happen to finish in.
    let (sender, receiver) = mpsc::channel::<(usize, String)>();
    let mut readers = 0;
    for (index, stream) in [
        child
            .stdout
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|s| Box::new(s) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .enumerate()
    {
        let Some(stream) = stream else { continue };
        readers += 1;
        let sender = sender.clone();
        thread::spawn(move || {
            let mut text = String::new();
            let mut stream = stream;
            let _ = stream.read_to_string(&mut text);
            let _ = sender.send((index, text));
        });
    }
    drop(sender);

    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }

    // The program has exited, so its output is complete unless a grandchild
    // still holds a pipe; give each reader a short grace period, no more.
    let mut parts = Vec::new();
    for _ in 0..readers {
        if let Ok(part) = receiver.recv_timeout(Duration::from_millis(500)) {
            parts.push(part);
        }
    }
    parts.sort_by_key(|(index, _)| *index);
    Some(
        parts
            .into_iter()
            .map(|(_, text)| text)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Which Brush command line to speak.
///
/// The released 0.3.0 CLI takes `--total-steps` and understands an `{iter}`
/// placeholder in `--export-name`, so every preview keeps its own file. The
/// older flag set this window was first written against used
/// `--total-train-iters` and a fixed export name, so only the final model is
/// written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CliStyle {
    Modern,
    Legacy,
}

impl CliStyle {
    fn steps_flag(self) -> &'static str {
        match self {
            Self::Modern => "--total-steps",
            Self::Legacy => "--total-train-iters",
        }
    }

    fn export_name(self) -> &'static str {
        match self {
            Self::Modern => "export_{iter}.ply",
            Self::Legacy => "export.ply",
        }
    }

    /// Decide from `--help` output. Only the legacy flag with no sign of the
    /// modern one selects the legacy style; anything unrecognised is treated as
    /// modern, which is what a current install speaks.
    fn from_help(help: &str) -> Self {
        if help.contains("--total-train-iters") && !help.contains("--total-steps") {
            Self::Legacy
        } else {
            Self::Modern
        }
    }
}

fn detect_style(program: &str) -> CliStyle {
    run_capture(Path::new(program), &["--help"], Duration::from_secs(10))
        .map(|help| CliStyle::from_help(&help))
        .unwrap_or(CliStyle::Modern)
}

/// What to run.
#[derive(Clone, Debug)]
pub(crate) struct TrainerConfig {
    pub program: String,
    pub dataset: PathBuf,
    pub output: PathBuf,
    pub iterations: usize,
    /// Write a complete splat model every this many steps. These are the
    /// training previews; the final model is always written as well.
    pub preview_every: usize,
    /// Also open Brush's own live viewer in a separate window.
    pub viewer: bool,
}

impl TrainerConfig {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.program.trim().is_empty() {
            return Err("trainer executable is empty".to_string());
        }
        if !self.dataset.is_dir() {
            return Err(format!(
                "dataset {} is not a directory",
                self.dataset.display()
            ));
        }
        if self.iterations == 0 {
            return Err("training iterations must be at least 1".to_string());
        }
        super::check_directory_writable(&self.output)
    }
}

/// A complete splat model Brush wrote part-way through training.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Preview {
    pub iteration: usize,
    pub path: PathBuf,
}

/// A snapshot of the worker's state, cheap enough to copy every repaint.
#[derive(Clone, Debug)]
pub(crate) struct TrainerView {
    pub status: TrainerStatus,
    pub log: Vec<String>,
    pub message: String,
    pub output: Option<PathBuf>,
    pub started: Option<Instant>,
    pub finished: Option<Instant>,
    /// The step count asked for.
    pub total: usize,
    /// The latest step known, from a refine line or a preview file. Never goes
    /// backwards.
    pub iteration: usize,
    /// Live splat count from the most recent refine line, when Brush logged one.
    pub splats: Option<usize>,
    /// Training views Brush reported loading.
    pub views: Option<usize>,
    /// Settled preview exports, oldest first.
    pub previews: Vec<Preview>,
    /// The Brush command style in use, once detected.
    pub style: Option<CliStyle>,
}

impl Default for TrainerView {
    fn default() -> Self {
        Self {
            status: TrainerStatus::Idle,
            log: Vec::new(),
            message: String::new(),
            output: None,
            started: None,
            finished: None,
            total: 0,
            iteration: 0,
            splats: None,
            views: None,
            previews: Vec::new(),
            style: None,
        }
    }
}

impl TrainerView {
    fn push_log(&mut self, line: String) {
        let progress = parse_line(&line);
        if let Some(iteration) = progress.iteration {
            self.iteration = self.iteration.max(iteration);
        }
        if progress.splats.is_some() {
            self.splats = progress.splats;
        }
        if progress.views.is_some() {
            self.views = progress.views;
        }

        if self.log.len() >= MAX_LOG_LINES {
            self.log.remove(0);
        }
        self.log.push(line);
    }

    fn set_previews(&mut self, previews: Vec<Preview>) {
        if let Some(last) = previews.last() {
            self.iteration = self.iteration.max(last.iteration);
        }
        self.previews = previews;
    }

    /// How far along training is, `0..=1`, or `None` before anything is known.
    pub(crate) fn fraction(&self) -> Option<f32> {
        (self.total > 0 && self.iteration > 0)
            .then(|| (self.iteration as f32 / self.total as f32).clamp(0.0, 1.0))
    }
}

/// What one line of Brush output tells us.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    pub iteration: Option<usize>,
    pub splats: Option<usize>,
    pub views: Option<usize>,
}

/// Read progress out of a Brush log line. Anything else is ignored.
///
/// Recognised, from the 0.3.0 release at `RUST_LOG=brush_cli=info`:
///
/// ```text
/// Loaded dataset with 3 training, 0 eval views
/// Refine iter 201, 10000 splats.
/// ```
pub(crate) fn parse_line(line: &str) -> Progress {
    let mut progress = Progress::default();

    if let Some(rest) = line.split("Refine iter ").nth(1) {
        let mut parts = rest.splitn(2, ',');
        progress.iteration = parts.next().and_then(leading_number);
        // A line whose iteration cannot be read has drifted from the format
        // this understands, so nothing else on it is trusted either.
        if progress.iteration.is_some() {
            progress.splats = parts
                .next()
                .filter(|tail| tail.contains("splats"))
                .and_then(leading_number);
        }
    } else if let Some(rest) = line.split("Loaded dataset with ").nth(1) {
        if rest.contains("training") {
            progress.views = leading_number(rest);
        }
    }
    progress
}

fn leading_number(text: &str) -> Option<usize> {
    let text = text.trim_start();
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The iteration in a Brush export file name, `export_<n>.ply`.
fn export_iteration(name: &str) -> Option<usize> {
    name.strip_prefix("export_")?
        .strip_suffix(".ply")?
        .parse()
        .ok()
}

/// The preview exports in `dir`, oldest first.
///
/// With `settled` false, a file is listed only once it has been left alone for
/// [`SETTLE`], which avoids handing the window a half-written model.
fn scan_exports(dir: &Path, settled: bool) -> Vec<Preview> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };

    let now = SystemTime::now();
    let mut previews: Vec<Preview> = entries
        .flatten()
        .filter_map(|entry| {
            let iteration = export_iteration(entry.file_name().to_str()?)?;
            let meta = entry.metadata().ok()?;
            if !meta.is_file() || meta.len() == 0 {
                return None;
            }
            if !settled {
                let age = now
                    .duration_since(meta.modified().ok()?)
                    .unwrap_or_default();
                if age < SETTLE {
                    return None;
                }
            }
            Some(Preview {
                iteration,
                path: entry.path(),
            })
        })
        .collect();
    previews.sort_by_key(|preview| preview.iteration);
    previews
}

/// One line for the progress bar: steps, splats, elapsed and an estimate of
/// what is left. The estimate is linear in steps, which is rough (refinement
/// makes later steps slower) and is labelled with a `~` accordingly.
pub(crate) fn progress_summary(
    iteration: usize,
    total: usize,
    splats: Option<usize>,
    elapsed: Duration,
    running: bool,
) -> String {
    let mut parts = Vec::new();
    if total > 0 {
        parts.push(format!("{} / {} steps", iteration.min(total), total));
    }
    if let Some(splats) = splats {
        parts.push(format!("{} splats", group_thousands(splats)));
    }
    parts.push(format!("{} elapsed", clock(elapsed)));
    if running && iteration > 0 && total > iteration {
        let per_step = elapsed.as_secs_f64() / iteration as f64;
        let left = Duration::from_secs_f64(per_step * (total - iteration) as f64);
        parts.push(format!("~{} left", clock(left)));
    }
    parts.join("  ·  ")
}

fn clock(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let (hours, minutes, seconds) = (seconds / 3600, seconds / 60 % 60, seconds % 60);
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn group_thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// Owns the child thread. Dropping it cancels and joins, so a window close
/// cannot leave a trainer running in the background.
pub(crate) struct TrainerHandle {
    cancel: Arc<AtomicBool>,
    view: Arc<Mutex<TrainerView>>,
    join: Option<JoinHandle<()>>,
}

impl TrainerHandle {
    pub(crate) fn start(config: TrainerConfig) -> Result<Self, String> {
        config.validate()?;

        let cancel = Arc::new(AtomicBool::new(false));
        let view = Arc::new(Mutex::new(TrainerView {
            status: TrainerStatus::Running,
            log: vec![format!(
                "starting {} ({} steps)",
                config.program, config.iterations
            )],
            message: "training".to_string(),
            started: Some(Instant::now()),
            total: config.iterations,
            ..TrainerView::default()
        }));

        let worker_cancel = Arc::clone(&cancel);
        let worker_view = Arc::clone(&view);
        let join = thread::spawn(move || worker(config, worker_cancel, worker_view));

        Ok(Self {
            cancel,
            view,
            join: Some(join),
        })
    }

    pub(crate) fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub(crate) fn snapshot(&self) -> TrainerView {
        self.view.lock().unwrap().clone()
    }

    pub(crate) fn join(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }

    /// Wait for the worker, but never long enough to hang a window close.
    /// Returns false if the thread had to be detached.
    pub(crate) fn join_timeout(&mut self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while self.join.as_ref().is_some_and(|join| !join.is_finished()) {
            if Instant::now() >= deadline {
                self.join.take();
                return false;
            }
            thread::sleep(Duration::from_millis(20));
        }
        self.join();
        true
    }
}

impl Drop for TrainerHandle {
    fn drop(&mut self) {
        self.cancel();
        if !self.join_timeout(Duration::from_secs(5)) {
            eprintln!("scan gui: the trainer did not stop within 5 s; leaving it to process exit");
        }
    }
}

fn fail(view: &Arc<Mutex<TrainerView>>, message: String) {
    let mut view = view.lock().unwrap();
    view.status = TrainerStatus::Failed;
    view.message = message;
    view.finished = Some(Instant::now());
}

fn worker(config: TrainerConfig, cancel: Arc<AtomicBool>, view: Arc<Mutex<TrainerView>>) {
    let style = detect_style(&config.program);
    {
        let mut view = view.lock().unwrap();
        view.style = Some(style);
        view.push_log(format!(
            "Brush command style: {} ({})",
            match style {
                CliStyle::Modern => "current",
                CliStyle::Legacy => "legacy, final model only",
            },
            style.steps_flag()
        ));
    }

    // Brush writes a kernel cache into its working directory, and fails with an
    // opaque panic when that directory is not writable. The window may have been
    // started from anywhere, so run it in the output directory instead.
    if let Err(error) = fs::create_dir_all(&config.output) {
        fail(
            &view,
            format!("cannot create {}: {error}", config.output.display()),
        );
        return;
    }

    // A preview interval beyond the run would write nothing useful before the
    // final model, and zero is rejected by Brush.
    let preview_every = config.preview_every.clamp(1, config.iterations);

    let mut command = Command::new(&config.program);
    command
        .arg(&config.dataset)
        .arg(style.steps_flag())
        .arg(config.iterations.to_string())
        .arg("--export-every")
        .arg(
            match style {
                // The legacy style only ever wrote one fixed name, so intermediate
                // exports would just overwrite it.
                CliStyle::Legacy => config.iterations,
                CliStyle::Modern => preview_every,
            }
            .to_string(),
        )
        .arg("--export-path")
        .arg(&config.output)
        .arg("--export-name")
        .arg(style.export_name());
    if config.viewer {
        command.arg("--with-viewer");
    }
    command
        .current_dir(&config.output)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Brush logs nothing by default. This is the narrowest filter that makes it
    // report refinement progress; an operator's own RUST_LOG wins.
    if std::env::var_os("RUST_LOG").is_none() {
        command.env("RUST_LOG", "brush_cli=info");
    }

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            fail(
                &view,
                format!("could not start {}: {error}", config.program),
            );
            return;
        }
    };

    let (sender, receiver) = mpsc::channel();
    let mut readers = Vec::new();
    if let Some(stdout) = child.stdout.take() {
        readers.push(read_lines(stdout, sender.clone()));
    }
    if let Some(stderr) = child.stderr.take() {
        readers.push(read_lines(stderr, sender.clone()));
    }
    drop(sender);

    let mut last_scan = Instant::now();
    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            break TrainerStatus::Cancelled;
        }

        drain(&receiver, &view);
        if last_scan.elapsed() >= SCAN_EVERY {
            last_scan = Instant::now();
            let previews = scan_exports(&config.output, false);
            view.lock().unwrap().set_previews(previews);
        }

        match child.try_wait() {
            Ok(Some(exit)) => {
                drain(&receiver, &view);
                break if exit.success() {
                    TrainerStatus::Finished
                } else {
                    TrainerStatus::Failed
                };
            }
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let mut view = view.lock().unwrap();
                view.push_log(format!("wait failed: {error}"));
                break TrainerStatus::Failed;
            }
        }

        thread::sleep(Duration::from_millis(50));
    };

    for reader in readers {
        let _ = reader.join();
    }
    drain(&receiver, &view);

    // Nothing is writing now, so every export on disk is complete.
    let previews = scan_exports(&config.output, true);

    let mut view = view.lock().unwrap();
    view.set_previews(previews);
    view.status = status;
    view.finished = Some(Instant::now());

    let legacy_output = config.output.join("export.ply");
    let final_model = view
        .previews
        .last()
        .map(|preview| preview.path.clone())
        .or_else(|| legacy_output.is_file().then_some(legacy_output.clone()));

    match status {
        TrainerStatus::Finished => match final_model {
            Some(path) => {
                view.iteration = view.iteration.max(config.iterations);
                view.message = format!("wrote {}", path.display());
                view.output = Some(path);
            }
            None => {
                view.status = TrainerStatus::Failed;
                view.message = format!(
                    "trainer exited successfully but wrote no model in {}",
                    config.output.display()
                );
            }
        },
        TrainerStatus::Cancelled => {
            view.message = "cancelled".to_string();
        }
        TrainerStatus::Failed => {
            view.message = "trainer exited with an error".to_string();
        }
        TrainerStatus::Idle | TrainerStatus::Running => {}
    }
}

fn drain(receiver: &Receiver<String>, view: &Arc<Mutex<TrainerView>>) {
    let mut view = view.lock().unwrap();
    while let Ok(line) = receiver.try_recv() {
        view.push_log(line);
    }
}

fn read_lines<R: Read + Send + 'static>(reader: R, sender: Sender<String>) -> JoinHandle<()> {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(Result::ok) {
            if sender.send(line).is_err() {
                break;
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("scan-trainer-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn script_named(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
    }

    fn script(dir: &Path, body: &str) -> PathBuf {
        script_named(dir, "train.sh", body)
    }

    /// A stand-in for the current Brush: answers `--help` with the modern flag.
    const MODERN_HELP: &str =
        "if [ \"$1\" = \"--help\" ]; then echo '  --total-steps <TOTAL_STEPS>'; exit 0; fi";
    /// A stand-in for the older Brush this window was first written against.
    const LEGACY_HELP: &str =
        "if [ \"$1\" = \"--help\" ]; then echo '  --total-train-iters <N>'; exit 0; fi";

    fn config(program: &Path, dir: &Path, iterations: usize) -> TrainerConfig {
        TrainerConfig {
            program: program.display().to_string(),
            dataset: dir.to_path_buf(),
            output: dir.join("out"),
            iterations,
            preview_every: iterations,
            viewer: false,
        }
    }

    fn wait_for(handle: &TrainerHandle, timeout: Duration) -> TrainerView {
        let deadline = Instant::now() + timeout;
        loop {
            let view = handle.snapshot();
            if !view.status.active() || Instant::now() > deadline {
                return view;
            }
            thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    fn a_missing_dataset_is_rejected_before_spawning() {
        let dir = temp_dir("missing");
        let mut cfg = config(Path::new("sh"), &dir, 1);
        cfg.dataset = dir.join("not-a-dataset");
        let error = match TrainerHandle::start(cfg) {
            Ok(_) => panic!("a missing dataset must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("dataset"), "{error}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_successful_run_reports_its_final_model_and_progress() {
        let dir = temp_dir("success");
        // $7 is the export path. Two exports, a refine line and a views line.
        let script = script(
            &dir,
            &format!(
                "{MODERN_HELP}\n\
                 echo '[t INFO  brush_cli] Loaded dataset with 3 training, 0 eval views' >&2\n\
                 echo '[t INFO  brush_cli] Refine iter 7, 5000 splats.' >&2\n\
                 printf a > \"$7/export_5.ply\"\n\
                 printf b > \"$7/export_10.ply\""
            ),
        );
        let output = dir.join("out");
        let mut cfg = config(&script, &dir, 10);
        cfg.preview_every = 5;
        let mut handle = TrainerHandle::start(cfg).expect("start");

        wait_for(&handle, Duration::from_secs(10));
        handle.join();
        let view = handle.snapshot();

        assert_eq!(view.status, TrainerStatus::Finished, "{}", view.message);
        assert_eq!(view.output, Some(output.join("export_10.ply")));
        assert_eq!(
            view.previews
                .iter()
                .map(|p| p.iteration)
                .collect::<Vec<_>>(),
            vec![5, 10]
        );
        assert_eq!(view.iteration, 10);
        assert_eq!(view.total, 10);
        assert_eq!(view.splats, Some(5000));
        assert_eq!(view.views, Some(3));
        assert_eq!(view.style, Some(CliStyle::Modern));
        assert!(
            view.log.iter().any(|line| line.contains("starting")),
            "{:?}",
            view.log
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_current_cli_gets_current_flags_and_the_legacy_cli_gets_legacy_ones() {
        for (help, steps_flag, export_name) in [
            (MODERN_HELP, "--total-steps", "export_{iter}.ply"),
            (LEGACY_HELP, "--total-train-iters", "export.ply"),
        ] {
            let dir = temp_dir(&format!("flags-{steps_flag}"));
            let script = script(
                &dir,
                &format!(
                    "{help}\nprintf '%s\\n' \"$@\" > \"$7/args.txt\"\nprintf x > \"$7/{}\"",
                    if steps_flag == "--total-steps" {
                        "export_10.ply"
                    } else {
                        "export.ply"
                    }
                ),
            );
            let mut handle = TrainerHandle::start(config(&script, &dir, 10)).expect("start");
            let view = wait_for(&handle, Duration::from_secs(10));
            handle.join();
            assert!(!view.status.active());

            let args = fs::read_to_string(dir.join("out/args.txt")).expect("args written");
            let args: Vec<&str> = args.lines().collect();
            assert_eq!(args[1], steps_flag, "{args:?}");
            assert_eq!(args[2], "10");
            assert_eq!(args[3], "--export-every");
            assert_eq!(args[5], "--export-path");
            assert_eq!(args[7], "--export-name");
            assert_eq!(args[8], export_name);
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_legacy_run_still_reports_its_single_export() {
        let dir = temp_dir("legacy");
        let script = script(
            &dir,
            &format!("{LEGACY_HELP}\nprintf splat > \"$7/export.ply\""),
        );
        let output = dir.join("out");
        let mut handle = TrainerHandle::start(config(&script, &dir, 10)).expect("start");
        wait_for(&handle, Duration::from_secs(10));
        handle.join();

        let view = handle.snapshot();
        assert_eq!(view.status, TrainerStatus::Finished, "{}", view.message);
        assert_eq!(view.output, Some(output.join("export.ply")));
        assert!(view.previews.is_empty(), "legacy has no numbered previews");
        assert_eq!(view.style, Some(CliStyle::Legacy));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_viewer_flag_the_log_filter_and_the_working_directory_reach_the_child() {
        let dir = temp_dir("env");
        let script = script(
            &dir,
            &format!(
                "{MODERN_HELP}\n\
                 printf '%s\\n' \"$@\" > \"$7/args.txt\"\n\
                 printf '%s' \"$RUST_LOG\" > \"$7/rust_log.txt\"\n\
                 pwd > \"$7/cwd.txt\"\n\
                 printf x > \"$7/export_10.ply\""
            ),
        );
        let mut cfg = config(&script, &dir, 10);
        cfg.viewer = true;
        let mut handle = TrainerHandle::start(cfg).expect("start");
        wait_for(&handle, Duration::from_secs(10));
        handle.join();

        let args = fs::read_to_string(dir.join("out/args.txt")).unwrap();
        assert!(args.lines().any(|line| line == "--with-viewer"), "{args}");

        // Only meaningful when the test runner has no RUST_LOG of its own.
        if std::env::var_os("RUST_LOG").is_none() {
            assert_eq!(
                fs::read_to_string(dir.join("out/rust_log.txt")).unwrap(),
                "brush_cli=info"
            );
        }

        let cwd = fs::read_to_string(dir.join("out/cwd.txt")).unwrap();
        assert_eq!(
            fs::canonicalize(cwd.trim()).unwrap(),
            fs::canonicalize(dir.join("out")).unwrap(),
            "Brush must run in the writable output directory"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_trainer_that_writes_nothing_is_reported_as_failed() {
        let dir = temp_dir("nothing");
        let script = script(&dir, MODERN_HELP);
        let mut handle = TrainerHandle::start(config(&script, &dir, 10)).expect("start");
        wait_for(&handle, Duration::from_secs(10));
        handle.join();

        let view = handle.snapshot();
        assert_eq!(view.status, TrainerStatus::Failed);
        assert!(view.message.contains("wrote no model"), "{}", view.message);
        assert!(view.output.is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cancel_stops_a_running_child() {
        let dir = temp_dir("cancel");
        let script = script(&dir, &format!("{MODERN_HELP}\nsleep 10"));
        let mut handle = TrainerHandle::start(config(&script, &dir, 10)).expect("start");

        // Give the child a moment to be spawned before killing it.
        thread::sleep(Duration::from_millis(300));
        handle.cancel();
        handle.join();

        let view = handle.snapshot();
        assert_eq!(view.status, TrainerStatus::Cancelled, "{}", view.message);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn log_lines_are_parsed_and_unrelated_lines_are_ignored() {
        assert_eq!(
            parse_line("[2026-11-13T08:47:21Z INFO  brush_cli] Refine iter 301, 10000 splats."),
            Progress {
                iteration: Some(301),
                splats: Some(10000),
                views: None
            }
        );
        assert_eq!(
            parse_line("[x INFO  brush_cli] Loaded dataset with 120 training, 4 eval views"),
            Progress {
                iteration: None,
                splats: None,
                views: Some(120)
            }
        );
        // A refine line in some other shape still yields the iteration it can.
        assert_eq!(
            parse_line("Refine iter 50, grew"),
            Progress {
                iteration: Some(50),
                splats: None,
                views: None
            }
        );
        for ignored in [
            "",
            "Done training! Took FormattedDuration(1s).",
            "Refine iter lots, 10 splats.",
            "Loaded dataset with no training views",
            "[x INFO  brush_dataset::scene] load_scene_img; self=LoadImage { vfs: BrushVfs {",
        ] {
            assert_eq!(parse_line(ignored), Progress::default(), "{ignored:?}");
        }
    }

    #[test]
    fn export_file_names_carry_the_iteration() {
        assert_eq!(export_iteration("export_5000.ply"), Some(5000));
        assert_eq!(export_iteration("export_0.ply"), Some(0));
        for other in [
            "export.ply",
            "export_.ply",
            "export_12.txt",
            "model_12.ply",
            "export_1_2.ply",
        ] {
            assert_eq!(export_iteration(other), None, "{other}");
        }
    }

    #[test]
    fn exports_are_listed_in_order_and_unsettled_ones_are_held_back() {
        let dir = temp_dir("scan");
        for (name, body) in [
            ("export_200.ply", "b"),
            ("export_100.ply", "a"),
            ("export_300.ply", ""),
            ("notes.txt", "x"),
        ] {
            fs::write(dir.join(name), body).unwrap();
        }

        // Settled: every non-empty export, oldest first; the empty file is not a model.
        let settled = scan_exports(&dir, true);
        assert_eq!(
            settled.iter().map(|p| p.iteration).collect::<Vec<_>>(),
            vec![100, 200]
        );

        // Just written, so not yet trusted while the trainer may still be writing.
        assert!(scan_exports(&dir, false).is_empty());
        assert!(scan_exports(&dir.join("missing"), true).is_empty());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn help_text_selects_the_command_style() {
        assert_eq!(
            CliStyle::from_help("  --total-steps <N>\n  --export-every"),
            CliStyle::Modern
        );
        assert_eq!(
            CliStyle::from_help("  --total-train-iters <N>"),
            CliStyle::Legacy
        );
        // A build that mentions both is current.
        assert_eq!(
            CliStyle::from_help("--total-steps --total-train-iters"),
            CliStyle::Modern
        );
        // Unrecognised output must not strand a current install on the legacy flags.
        assert_eq!(CliStyle::from_help(""), CliStyle::Modern);
        assert_eq!(
            CliStyle::from_help("usage: something else"),
            CliStyle::Modern
        );
    }

    #[test]
    fn brush_is_found_by_preferred_name_and_reports_its_version() {
        let first = temp_dir("find-a");
        let second = temp_dir("find-b");
        script_named(&first, "brush_app", "echo 'brush-cli 0.3.0'");
        script_named(&second, "brush", "echo 'brush 9.9.9'");
        // Present but not executable: must never be picked.
        fs::write(first.join("brush-cli"), "not a program").unwrap();

        let found = find_brush_in(&[first.clone(), second.clone()]).expect("found");
        assert_eq!(
            found.path,
            second.join("brush"),
            "`brush` outranks `brush_app`"
        );
        assert_eq!(found.version, "brush 9.9.9");

        let only_app = find_brush_in(std::slice::from_ref(&first)).expect("found");
        assert_eq!(only_app.path, first.join("brush_app"));
        assert_eq!(only_app.version, "brush-cli 0.3.0");

        assert_eq!(find_brush_in(&[temp_dir("find-empty")]), None);
        assert_eq!(find_brush_in(&[]), None);
        for dir in [first, second] {
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn a_probe_that_hangs_is_killed_instead_of_freezing_the_window() {
        let dir = temp_dir("hang");
        let hang = script_named(&dir, "brush", "sleep 30");
        let started = Instant::now();
        let found = find_brush_in(std::slice::from_ref(&dir)).expect("still found");
        assert_eq!(found.path, hang);
        assert!(found.version.is_empty(), "no version from a hung probe");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "took {:?}",
            started.elapsed()
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_progress_line_shows_steps_splats_elapsed_and_an_estimate() {
        let line = progress_summary(
            10_000,
            30_000,
            Some(1_234_567),
            Duration::from_secs(100),
            true,
        );
        assert_eq!(
            line,
            "10000 / 30000 steps  ·  1,234,567 splats  ·  1:40 elapsed  ·  ~3:20 left"
        );

        // Finished or not started: no estimate. Never more steps than the total.
        let done = progress_summary(31_000, 30_000, None, Duration::from_secs(3725), false);
        assert_eq!(done, "30000 / 30000 steps  ·  1:02:05 elapsed");
        let early = progress_summary(0, 30_000, None, Duration::from_secs(4), true);
        assert_eq!(early, "0 / 30000 steps  ·  0:04 elapsed");
    }

    #[test]
    fn iteration_only_moves_forward_and_drives_the_fraction() {
        let mut view = TrainerView {
            total: 1000,
            ..TrainerView::default()
        };
        assert_eq!(
            view.fraction(),
            None,
            "nothing is known before the first signal"
        );

        view.push_log("Refine iter 400, 10 splats.".to_string());
        assert_eq!(view.fraction(), Some(0.4));

        // A stale or out-of-order line must not move the bar backwards.
        view.push_log("Refine iter 100, 10 splats.".to_string());
        assert_eq!(view.iteration, 400);

        view.set_previews(vec![Preview {
            iteration: 600,
            path: PathBuf::from("export_600.ply"),
        }]);
        assert_eq!(view.fraction(), Some(0.6));

        view.iteration = 5000;
        assert_eq!(view.fraction(), Some(1.0), "clamped at the total");
    }

    #[test]
    fn thousands_are_grouped() {
        assert_eq!(group_thousands(0), "0");
        assert_eq!(group_thousands(999), "999");
        assert_eq!(group_thousands(1000), "1,000");
        assert_eq!(group_thousands(10_000_000), "10,000,000");
    }

    /// The real thing: the installed Brush trains a freshly exported synthetic
    /// dataset through `TrainerHandle`, writes its previews, and the window's
    /// loader can read the result. Skipped unless asked for, because it needs
    /// Brush on PATH and a GPU:
    ///
    /// `cargo test -p scan --features gui real_brush -- --ignored`
    #[test]
    #[ignore = "needs Brush on PATH and a GPU; run with --ignored"]
    fn real_brush_trains_a_synthetic_dataset_and_reports_previews() {
        let brush = find_brush().expect("Brush must be on PATH for this test");
        let dir = temp_dir("real-brush");
        let dataset = dir.join("dataset");
        let report = crate::dataset::write_synthetic_dataset(&dataset, 12);

        // 250 steps crosses Brush's first refinement (step 201), so the log
        // reports a splat count, and 125 gives one preview before the final one.
        let cfg = TrainerConfig {
            program: brush.path.display().to_string(),
            dataset,
            output: dir.join("out"),
            iterations: 250,
            preview_every: 125,
            viewer: false,
        };
        let mut handle = TrainerHandle::start(cfg).expect("start");
        let view = wait_for(&handle, Duration::from_secs(300));
        handle.join();
        let view = handle
            .snapshot()
            .status
            .active()
            .then_some(view)
            .unwrap_or_else(|| handle.snapshot());

        assert_eq!(
            view.status,
            TrainerStatus::Finished,
            "{}\n{:#?}",
            view.message,
            view.log
        );
        assert_eq!(view.style, Some(CliStyle::Modern));
        assert_eq!(
            view.previews
                .iter()
                .map(|p| p.iteration)
                .collect::<Vec<_>>(),
            vec![125, 250]
        );
        assert_eq!(view.iteration, 250);
        assert_eq!(view.views, Some(report.views), "{:#?}", view.log);
        assert!(
            view.splats.is_some_and(|n| n > 0),
            "no splat count in {:#?}",
            view.log
        );

        let output = view.output.expect("a final model");
        assert!(output.ends_with("export_250.ply"), "{}", output.display());
        let cloud = crate::gui::splat::SplatCloud::load(&output).expect("the window can load it");
        assert!(cloud.len() > 0);
        let _ = fs::remove_dir_all(dir);
    }
}
