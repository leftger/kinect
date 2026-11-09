//! A trainer child process, kept off the UI thread.
//!
//! The scanner window owns the dataset hand-off but not the trainer. This
//! module is the smallest useful bridge: build a Brush command, run it as a
//! child, stream its output into a bounded log, and let Cancel actually kill
//! it. It deliberately does not try to parse training loss: Brush's log format
//! is not a stable API, and a GUI that guesses at it is worse than one that
//! shows the lines it received.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// How many lines of trainer output to keep. Brush can be noisy; the operator
/// needs the tail, not the whole run in memory.
const MAX_LOG_LINES: usize = 240;

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

/// What to run. The default fields build the same command as the README's
/// Brush smoke test; a different executable can be used for testing or for a
/// Brush build outside `PATH`.
#[derive(Clone, Debug)]
pub(crate) struct TrainerConfig {
    pub program: String,
    pub dataset: PathBuf,
    pub output: PathBuf,
    pub iterations: usize,
}

impl TrainerConfig {
    pub(crate) fn output_file(&self) -> PathBuf {
        self.output.join("export.ply")
    }

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

/// A snapshot of the worker's state, cheap enough to copy every repaint.
#[derive(Clone, Debug)]
pub(crate) struct TrainerView {
    pub status: TrainerStatus,
    pub log: Vec<String>,
    pub message: String,
    pub output: Option<PathBuf>,
    pub started: Option<Instant>,
    pub finished: Option<Instant>,
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
        }
    }
}

impl TrainerView {
    fn push_log(&mut self, line: String) {
        if self.log.len() >= MAX_LOG_LINES {
            self.log.remove(0);
        }
        self.log.push(line);
    }
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
                "starting {} ({} iterations)",
                config.program, config.iterations
            )],
            message: "training".to_string(),
            started: Some(Instant::now()),
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

fn worker(config: TrainerConfig, cancel: Arc<AtomicBool>, view: Arc<Mutex<TrainerView>>) {
    let output = config.output_file();
    let mut command = Command::new(&config.program);
    command
        .arg(&config.dataset)
        .arg("--total-train-iters")
        .arg(config.iterations.to_string())
        .arg("--export-every")
        .arg(config.iterations.to_string())
        .arg("--export-path")
        .arg(&config.output)
        .arg("--export-name")
        .arg("export.ply")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => {
            let mut view = view.lock().unwrap();
            view.status = TrainerStatus::Failed;
            view.message = format!("could not start {}: {error}", config.program);
            view.finished = Some(Instant::now());
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

    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            break TrainerStatus::Cancelled;
        }

        drain(&receiver, &view);

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

    let mut view = view.lock().unwrap();
    view.status = status;
    view.finished = Some(Instant::now());
    match status {
        TrainerStatus::Finished if output.is_file() => {
            view.output = Some(output.clone());
            view.message = format!("wrote {}", output.display());
        }
        TrainerStatus::Finished => {
            view.status = TrainerStatus::Failed;
            view.message = format!(
                "trainer exited successfully but did not write {}",
                output.display()
            );
        }
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
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("scan-trainer-{label}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn script(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("train.sh");
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&path, permissions).unwrap();
        path
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
        let config = TrainerConfig {
            program: "sh".to_string(),
            dataset: dir.join("not-a-dataset"),
            output: dir.join("out"),
            iterations: 1,
        };
        let error = match TrainerHandle::start(config) {
            Ok(_) => panic!("a missing dataset must be rejected"),
            Err(error) => error,
        };
        assert!(error.contains("dataset"), "{error}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_successful_run_reports_its_splat_file() {
        let dir = temp_dir("success");
        let script = script(&dir, "mkdir -p \"$7\"\nprintf splat > \"$7/export.ply\"");
        let output = dir.join("out");
        let mut handle = TrainerHandle::start(TrainerConfig {
            program: script.display().to_string(),
            dataset: dir.clone(),
            output: output.clone(),
            iterations: 10,
        })
        .expect("start");

        let view = wait_for(&handle, Duration::from_secs(5));
        handle.join();

        assert_eq!(view.status, TrainerStatus::Finished, "{}", view.message);
        assert_eq!(view.output, Some(output.join("export.ply")));
        assert!(
            view.log.iter().any(|line| line.contains("starting")),
            "{:?}",
            view.log
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cancel_stops_a_running_child() {
        let dir = temp_dir("cancel");
        let script = script(&dir, "sleep 10");
        let mut handle = TrainerHandle::start(TrainerConfig {
            program: script.display().to_string(),
            dataset: dir.clone(),
            output: dir.join("out"),
            iterations: 10,
        })
        .expect("start");

        // Give the child a moment to be spawned before killing it.
        thread::sleep(Duration::from_millis(100));
        handle.cancel();
        handle.join();

        let view = handle.snapshot();
        assert_eq!(view.status, TrainerStatus::Cancelled, "{}", view.message);
        let _ = fs::remove_dir_all(dir);
    }
}
