//! What the CUDA-graph dispatcher did during a test, read from a debugging metrics recorder.
#![allow(dead_code)]

use inference::{Model, RequestBuilder, TextMessageRole, TextMessages};
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

/// The greedy (token, logprob) per step.
pub type Trace = Vec<(u32, f32)>;
/// Each prompt's trace, per round of concurrent requests.
pub type Rounds = Vec<Vec<Trace>>;

const GRAPHS_ENV: &str = "INFERENCE_RS_CUDA_GRAPHS";
pub const EVENTS: &str = "inference_cuda_graph_events_total";
pub const DISPATCH: &str = "inference_cuda_graph_dispatch_total";
// different lengths, so a concurrent batch decodes rows at different positions
pub const GRAPH_PROMPTS: [&str; 5] = [
    "hello",
    "the quick brown fox jumps",
    "a",
    "one two three four five six seven",
    "x y",
];
// rounds of concurrent requests, so the decode batches fill different graph buckets
const GRAPH_BATCHES: [usize; 3] = [1, 3, 5];
// bf16 logprobs move with the batch a row decodes in and with graph padding
const BF16_LOGPROB_TOLERANCE: f32 = 0.06;

/// The greedy (token, logprob) per step of `max_len` steps for one prompt.
pub async fn prompt_trace(model: &Model, prompt: &str, max_len: usize) -> anyhow::Result<Trace> {
    let request =
        RequestBuilder::from(TextMessages::new().add_message(TextMessageRole::User, prompt))
            .set_sampler_max_len(max_len)
            .set_sampler_topk(1)
            .return_logprobs(true)
            .set_sampler_topn_logprobs(1);
    let response = model.send_chat_request(request).await?;
    let trace: Trace = response.choices[0]
        .logprobs
        .as_ref()
        .and_then(|lp| lp.content.as_ref())
        .map(|toks| {
            toks.iter()
                .map(|t| (t.top_logprobs[0].token, t.top_logprobs[0].logprob))
                .collect()
        })
        .unwrap_or_default();
    anyhow::ensure!(!trace.is_empty(), "the model generated nothing");
    Ok(trace)
}

/// Each prompt's trace, per round of concurrent requests.
pub async fn rounds(model: &Model, max_len: usize) -> anyhow::Result<Rounds> {
    let mut rounds = Vec::new();
    for batch in GRAPH_BATCHES {
        rounds.push(
            futures::future::try_join_all(
                GRAPH_PROMPTS[..batch]
                    .iter()
                    .map(|prompt| prompt_trace(model, prompt, max_len)),
            )
            .await?,
        );
    }
    Ok(rounds)
}

/// Every round decodes each prompt to its `expected` trace: the same ids, each step's logprob within tolerance.
pub fn assert_traces(rounds: &[Vec<Trace>], expected: &[&[(u32, f32)]]) {
    for (round, batch) in rounds.iter().zip(GRAPH_BATCHES) {
        assert_eq!(round.len(), batch, "{rounds:?}");
        for (trace, expected) in round.iter().zip(expected) {
            assert_trace(trace, expected, rounds);
        }
    }
}

/// `trace` has `expected`'s ids, each step's logprob within tolerance; `context` is printed on a mismatch.
pub fn assert_trace(
    trace: &[(u32, f32)],
    expected: &[(u32, f32)],
    context: &(impl std::fmt::Debug + ?Sized),
) {
    let ids = |t: &[(u32, f32)]| t.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    assert_eq!(ids(trace), ids(expected), "decode moved: {context:?}");
    assert!(
        trace
            .iter()
            .zip(expected)
            .all(|(got, want)| (got.1 - want.1).abs() < BF16_LOGPROB_TOLERANCE),
        "logprob moved: {context:?}"
    );
}

/// No decode step fell back to eager and no capture failed.
pub fn assert_no_fallback(counters: &Counters) {
    let failures = counters.total(EVENTS, &[("outcome", "failure")])
        + counters.total(EVENTS, &[("event", "eager_fallback")])
        + counters.total(DISPATCH, &[("mode", "eager")]);
    assert_eq!(failures, 0, "a decode step fell back: {counters:?}");
}

/// Decode graphs were captured and replayed, with no fallback.
pub fn assert_replayed(counters: &Counters) {
    let events = |event| {
        counters.total(
            EVENTS,
            &[
                ("component", "target"),
                ("event", event),
                ("outcome", "success"),
            ],
        )
    };
    assert!(
        events("capture") > 0 && events("replay") > 0,
        "no graph captured and replayed: {counters:?}"
    );
    assert_no_fallback(counters);
}

/// Every dispatch, prefill included, stopped at the disabled check, and no graph ran.
pub fn assert_disabled(counters: &Counters) {
    let disabled = counters.total(DISPATCH, &[("reason", "disabled")]);
    assert!(
        disabled > 0
            && disabled == counters.total(DISPATCH, &[])
            && counters.total(EVENTS, &[]) == 0,
        "graphs ran while disabled: {counters:?}"
    );
}

/// Graphs on: `load_rounds` replays captured graphs and decodes to `expected`.
pub async fn assert_rounds_replay(
    load_rounds: impl AsyncFnOnce() -> anyhow::Result<Rounds>,
    expected: &[&[(u32, f32)]],
) -> anyhow::Result<()> {
    let snapshotter = recorder();
    let rounds = load_rounds().await?;
    assert_replayed(&Counters::take(&snapshotter));
    assert_traces(&rounds, expected);
    Ok(())
}

/// Graphs off: `load_rounds` runs no graph and decodes to the same `expected`.
pub async fn assert_rounds_without_graphs(
    load_rounds: impl AsyncFnOnce() -> anyhow::Result<Rounds>,
    expected: &[&[(u32, f32)]],
) -> anyhow::Result<()> {
    disable_graphs();
    let snapshotter = recorder();
    let rounds = load_rounds().await?;
    assert_disabled(&Counters::take(&snapshotter));
    assert_traces(&rounds, expected);
    Ok(())
}

/// Installs the process's metrics recorder; nextest runs each test in its own process, so each gets a fresh one.
pub fn recorder() -> Snapshotter {
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().expect("one recorder per test process");
    snapshotter
}

/// Turns decode graphs off for this test process; call before the first model loads, which reads it once.
pub fn disable_graphs() {
    // SAFETY: no other thread reads the environment yet
    unsafe { std::env::set_var(GRAPHS_ENV, "0") };
}

/// A snapshot drains the counters, so take one per check.
pub struct Counters(Vec<(String, Labels, u64)>);

type Labels = Vec<(String, String)>;

impl Counters {
    pub fn take(snapshotter: &Snapshotter) -> Self {
        Self(
            snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .filter_map(|(key, .., value)| {
                    let DebugValue::Counter(n) = value else {
                        return None;
                    };
                    let key = key.key();
                    let labels = key
                        .labels()
                        .map(|l| (l.key().to_string(), l.value().to_string()))
                        .collect();
                    Some((key.name().to_string(), labels, n))
                })
                .collect(),
        )
    }

    /// The total of counter `name` over the label sets that include every `(key, value)` in `labels`.
    pub fn total(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        self.0
            .iter()
            .filter(|(n, l, _)| {
                n == name
                    && labels
                        .iter()
                        .all(|(k, v)| l.iter().any(|(lk, lv)| lk == k && lv == v))
            })
            .map(|(.., n)| n)
            .sum()
    }
}

impl std::fmt::Debug for Counters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.iter().filter(|(.., n)| *n > 0))
            .finish()
    }
}
