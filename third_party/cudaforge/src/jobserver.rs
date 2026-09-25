//! Job slots shared with cargo's jobserver, so kernel crates building in parallel stay within `cargo -j`.

use std::sync::{Condvar, Mutex, OnceLock};

struct State {
    // Cargo already counts this build script as one running job; that slot is ours without asking.
    implicit_free: bool,
    tokens: Vec<jobserver::Acquired>,
    waiters: usize,
    requested: usize,
}

struct Pool {
    state: Mutex<State>,
    ready: Condvar,
    helper: Option<jobserver::HelperThread>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        // SAFETY: first called from `init`, before the builder spawns threads or opens files.
        let client = unsafe { jobserver::Client::from_env() };
        let helper = client.and_then(|client| {
            client
                .into_helper_thread(|token| {
                    let pool = POOL.get().expect("helper runs after the pool is set");
                    let mut state = pool.state.lock().unwrap();
                    state.requested -= 1;
                    // A token nobody waits for any more goes straight back to cargo when dropped.
                    if let (Ok(token), true) = (token, state.waiters > 0) {
                        state.tokens.push(token);
                        pool.ready.notify_one();
                    }
                })
                .ok()
        });
        Pool {
            state: Mutex::new(State {
                implicit_free: true,
                tokens: Vec::new(),
                waiters: 0,
                requested: 0,
            }),
            ready: Condvar::new(),
            helper,
        }
    })
}

/// Connects to cargo's jobserver; call before spawning threads or opening files.
pub(crate) fn init() {
    pool();
}

pub(crate) enum Slot {
    Implicit,
    Token(#[allow(dead_code)] jobserver::Acquired),
    Unlimited,
}

impl Drop for Slot {
    fn drop(&mut self) {
        if matches!(self, Slot::Implicit) {
            let pool = pool();
            pool.state.lock().unwrap().implicit_free = true;
            pool.ready.notify_one();
        }
    }
}

/// Blocks until this process may run one more compiler job.
pub(crate) fn acquire() -> Slot {
    let pool = pool();
    let mut state = pool.state.lock().unwrap();
    state.waiters += 1;
    let slot = loop {
        if state.implicit_free {
            state.implicit_free = false;
            break Slot::Implicit;
        }
        let Some(helper) = &pool.helper else {
            break Slot::Unlimited;
        };
        if let Some(token) = state.tokens.pop() {
            break Slot::Token(token);
        }
        if state.requested < state.waiters {
            helper.request_token();
            state.requested += 1;
        }
        state = pool.ready.wait(state).unwrap();
    };
    state.waiters -= 1;
    slot
}
