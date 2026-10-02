use std::time::{Duration, Instant};

use inference_core::{NormalRequest, Request, RequestMessage, SamplingParams};
use tokio::sync::mpsc::{channel, error::TrySendError};

use super::tiny_engine;

// Well under the 10 s drop timeout, which is what a Terminate stuck behind the queue used to cost.
const PROMPT_DROP_LIMIT: Duration = Duration::from_secs(5);
const QUEUED_MAX_TOKENS: usize = 64;
const PROMPT: &str = "hello";

#[tokio::test(flavor = "multi_thread")]
async fn dropping_an_engine_with_a_full_request_queue_stops_it_promptly() -> anyhow::Result<()> {
    let (_dir, engine) = tiny_engine().await?;
    let sender = engine.state().get_sender(None)?;
    // Held so the engine does not skip the queued requests as abandoned.
    let mut receivers = Vec::new();
    loop {
        let (tx, rx) = channel(1);
        let mut sampling = SamplingParams::deterministic();
        sampling.max_len = Some(QUEUED_MAX_TOKENS);
        let message = RequestMessage::Completion {
            text: PROMPT.to_string(),
            echo_prompt: false,
            best_of: None,
        };
        let request = NormalRequest::new_simple(message, sampling, tx, 0, None, None);
        match sender.try_send(Request::Normal(Box::new(request))) {
            Ok(()) => receivers.push(rx),
            Err(TrySendError::Full(_)) => break,
            Err(TrySendError::Closed(_)) => {
                anyhow::bail!("the engine stopped while the queue filled")
            }
        }
    }
    drop(sender);

    let started = Instant::now();
    drop(engine);
    let elapsed = started.elapsed();
    assert!(
        elapsed < PROMPT_DROP_LIMIT,
        "drop took {elapsed:?} with {} queued requests",
        receivers.len()
    );
    Ok(())
}
