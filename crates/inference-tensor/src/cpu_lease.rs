//! Worker slots every process on the machine shares, so concurrent processes' compute pools do not oversubscribe the
//! cores between them. A slot is a lock file; a process takes slots for the extra workers of its parallel ops, keeps
//! them while it stays busy and gives them back when it goes idle or has held them a while. The kernel drops a dead
//! process's locks, so a crash or a killed test leaks none.

use std::time::{Duration, Instant};

// how often a process short of slots tries for more
const REFRESH: Duration = Duration::from_millis(2);
// a process that ran no parallel op this long gives its slots back
const IDLE: Duration = Duration::from_millis(20);
// a process holding slots this long gives them back once, so newer processes get a turn
const TURN: Duration = Duration::from_millis(50);
// `INFERENCE_CPU_SLOTS=0` turns leasing off: every op takes the workers it asks for
const SLOTS_ENV: &str = "INFERENCE_CPU_SLOTS";
const SLOT_DIR: &str = "inference-rs-cpu-slots";
const SHARED_MEMORY: &str = "/dev/shm";

/// How many of `want` extra workers the next parallel op may run: all of them when leasing is off, otherwise the slots
/// this process holds.
pub fn extra_workers(want: usize) -> usize {
    if want == 0 {
        return 0;
    }
    match imp::lease() {
        Some(lease) => lease.workers(want),
        None => want,
    }
}

/// The machine-wide slot count: the physical cores less the calling thread every process brings, or the
/// `INFERENCE_CPU_SLOTS` override.
fn slot_count(default: usize) -> Option<usize> {
    match std::env::var(SLOTS_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) => None,
        Some(slots) => Some(slots),
        None => Some(default.saturating_sub(1).max(1)),
    }
}

struct Holding {
    held: Vec<usize>,
    last_refresh: Option<Instant>,
    last_use: Instant,
    since: Option<Instant>,
}

impl Holding {
    // what to do before an op wanting `want` extras: give the turn back, try for more, or keep what is held
    fn plan(&mut self, want: usize, now: Instant) -> Plan {
        self.last_use = now;
        if self.since.is_some_and(|since| now - since >= TURN) {
            self.since = None;
            self.last_refresh = Some(now);
            return Plan::Release;
        }
        let due = self.last_refresh.is_none_or(|last| now - last >= REFRESH);
        if self.held.len() < want && due {
            self.last_refresh = Some(now);
            return Plan::Acquire;
        }
        Plan::Keep
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    Keep,
    Acquire,
    Release,
}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::fs::{File, OpenOptions};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    // lock files readable and writable by every user, as /dev/shm is shared
    const SLOT_MODE: u32 = 0o666;

    pub(super) struct Lease {
        files: Vec<File>,
        // a process starts its search at its own offset so processes do not all contend for slot 0
        start: usize,
        state: Mutex<Holding>,
    }

    fn try_lock(file: &File) -> bool {
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
    }

    fn unlock(file: &File) {
        unsafe {
            libc::flock(file.as_raw_fd(), libc::LOCK_UN);
        }
    }

    fn slot_dir() -> PathBuf {
        let base = match std::path::Path::new(SHARED_MEMORY).is_dir() {
            true => PathBuf::from(SHARED_MEMORY),
            false => std::env::temp_dir(),
        };
        base.join(format!("{SLOT_DIR}-{}", unsafe { libc::getuid() }))
    }

    impl Lease {
        fn open(slots: usize) -> Option<Self> {
            let dir = slot_dir();
            std::fs::create_dir_all(&dir).ok()?;
            let files = (0..slots)
                .map(|i| {
                    OpenOptions::new()
                        .read(true)
                        .write(true)
                        .create(true)
                        .truncate(false)
                        .mode(SLOT_MODE)
                        .open(dir.join(format!("slot-{i}")))
                })
                .collect::<std::io::Result<Vec<_>>>()
                .ok()?;
            Some(Self {
                files,
                start: std::process::id() as usize % slots,
                state: Mutex::new(Holding {
                    held: Vec::new(),
                    last_refresh: None,
                    last_use: Instant::now(),
                    since: None,
                }),
            })
        }

        pub(super) fn workers(&self, want: usize) -> usize {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            match state.plan(want, Instant::now()) {
                Plan::Keep => {}
                Plan::Release => self.release(&mut state),
                Plan::Acquire => {
                    let n = self.files.len();
                    for i in (0..n).map(|k| (self.start + k) % n) {
                        if state.held.len() >= want {
                            break;
                        }
                        if !state.held.contains(&i) && try_lock(&self.files[i]) {
                            state.held.push(i);
                        }
                    }
                    if !state.held.is_empty() && state.since.is_none() {
                        state.since = Some(Instant::now());
                    }
                }
            }
            state.held.len().min(want)
        }

        fn release(&self, state: &mut Holding) {
            for i in state.held.drain(..) {
                unlock(&self.files[i]);
            }
            state.since = None;
        }

        // the reaper's tick: an idle process gives its slots back
        fn reap(&self) {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.held.is_empty() && state.last_use.elapsed() >= IDLE {
                self.release(&mut state);
            }
        }
    }

    pub(super) fn lease() -> Option<&'static Lease> {
        static LEASE: OnceLock<Option<Lease>> = OnceLock::new();
        LEASE
            .get_or_init(|| {
                let opened = Lease::open(slot_count(num_cpus::get_physical())?)?;
                std::thread::Builder::new()
                    .name("cpu-lease-reaper".into())
                    .spawn(|| {
                        loop {
                            std::thread::sleep(IDLE);
                            if let Some(held) = lease() {
                                held.reap();
                            }
                        }
                    })
                    .ok()?;
                Some(opened)
            })
            .as_ref()
    }
}

#[cfg(not(unix))]
mod imp {
    pub(super) struct Lease;

    impl Lease {
        pub(super) fn workers(&self, want: usize) -> usize {
            want
        }
    }

    pub(super) fn lease() -> Option<&'static Lease> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holding(now: Instant) -> Holding {
        Holding {
            held: Vec::new(),
            last_refresh: None,
            last_use: now,
            since: None,
        }
    }

    // a process short of slots tries at most once a refresh interval, and a full one never does
    #[test]
    fn a_short_process_retries_each_refresh() {
        let now = Instant::now();
        let mut state = holding(now);
        assert_eq!(state.plan(3, now), Plan::Acquire);
        assert_eq!(state.plan(3, now + REFRESH / 2), Plan::Keep);
        assert_eq!(state.plan(3, now + REFRESH), Plan::Acquire);
        state.held = vec![0, 1, 2];
        assert_eq!(state.plan(3, now + 3 * REFRESH), Plan::Keep);
    }

    // a holder gives its slots back once a turn, then tries again a refresh later
    #[test]
    fn a_long_holder_gives_back_its_turn() {
        let now = Instant::now();
        let mut state = holding(now);
        state.held = vec![0, 1];
        state.since = Some(now);
        assert_eq!(state.plan(4, now + TURN), Plan::Release);
        state.held.clear();
        assert_eq!(state.plan(4, now + TURN + REFRESH / 2), Plan::Keep);
        assert_eq!(state.plan(4, now + TURN + REFRESH), Plan::Acquire);
    }
}
