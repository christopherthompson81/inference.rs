//! Job slots shared with cargo's jobserver, so kernel crates building in parallel stay within `cargo -j`.

use std::sync::{Condvar, Mutex, OnceLock};

struct State {
    // Cargo already counts this build script as one running job; that slot is ours without asking.
    implicit_free: bool,
    tokens: Vec<jobserver::Acquired>,
    waiters: usize,
    requested: usize,
    // The jobserver failed; stop limiting rather than stall the build.
    broken: bool,
}

struct Pool {
    state: Mutex<State>,
    ready: Condvar,
    helper: Option<jobserver::HelperThread>,
}

static POOL: OnceLock<Pool> = OnceLock::new();

fn on_token(token: std::io::Result<jobserver::Acquired>) {
    let pool = POOL.get().expect("helper runs after the pool is set");
    let mut state = pool.state.lock().unwrap();
    state.requested -= 1;
    match token {
        // A token nobody waits for any more goes straight back to cargo when dropped.
        Ok(token) if state.waiters > 0 => state.tokens.push(token),
        Ok(_) => return,
        Err(_) => state.broken = true,
    }
    pool.ready.notify_one();
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        // SAFETY: the fds come from cargo's CARGO_MAKEFLAGS; check_pipe rejects any that were not inherited as pipes.
        let client = unsafe { jobserver::Client::from_env_ext(true) }.client.ok();
        // Without a helper thread there is no way to wait for tokens, so the build runs unlimited as upstream does.
        let helper = client.and_then(|client| client.into_helper_thread(on_token).ok());
        Pool {
            state: Mutex::new(State {
                implicit_free: true,
                tokens: Vec::new(),
                waiters: 0,
                requested: 0,
                broken: false,
            }),
            ready: Condvar::new(),
            helper,
        }
    })
}

/// Connects to cargo's jobserver.
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
        // An arrived token is used before the implicit slot, or it could sit unreturned for the rest of the build.
        if let Some(token) = state.tokens.pop() {
            break Slot::Token(token);
        }
        if state.implicit_free {
            state.implicit_free = false;
            break Slot::Implicit;
        }
        let Some(helper) = pool.helper.as_ref().filter(|_| !state.broken) else {
            break Slot::Unlimited;
        };
        if state.requested < state.waiters {
            helper.request_token();
            state.requested += 1;
        }
        state = pool.ready.wait(state).unwrap();
    };
    state.waiters -= 1;
    if state.waiters == 0 {
        state.tokens.clear();
    }
    slot
}
