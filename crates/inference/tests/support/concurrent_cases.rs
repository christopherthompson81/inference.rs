//! A module's model tests as concurrent cases of one test. Each test process pays ~0.12 s of CUDA context creation
//! that the driver serializes across processes, so cases share one process's context and run side by side, as many
//! at once as their measured VRAM fits a per-process budget (docs/investigations/test_time_investigation.md).
#![allow(dead_code)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;

// MiB of VRAM the running cases of one process may claim together
pub const BUDGET_MIB: u32 = 4096;
// a substring of case names: only those run, for iterating on one case
const CASE_FILTER_ENV: &str = "INFERENCE_TEST_CASE";

pub type CaseFuture = Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>;

/// A case and the peak VRAM its own test process used alone, in MiB (0 for a CPU-only case).
pub struct Case {
    pub name: &'static str,
    pub vram_mib: u32,
    pub run: fn() -> CaseFuture,
}

/// Runs every case as its own task (a panic fails only that case), then fails naming each failing case.
pub async fn run(cases: Vec<Case>) {
    let only = std::env::var(CASE_FILTER_ENV).ok();
    let budget = Arc::new(Semaphore::new(BUDGET_MIB as usize));
    let mut handles = Vec::new();
    for case in cases {
        if only.as_deref().is_some_and(|o| !case.name.contains(o)) {
            continue;
        }
        let budget = budget.clone();
        let task = tokio::spawn(async move {
            let _claim = budget
                .acquire_many_owned(case.vram_mib.min(BUDGET_MIB))
                .await
                .expect("the budget is never closed");
            let start = Instant::now();
            ((case.run)().await, start.elapsed())
        });
        handles.push((case.name, task));
    }
    let mut failed = Vec::new();
    for (name, task) in handles {
        match task.await {
            Ok((Ok(()), took)) => eprintln!("case {name}: ok in {took:.2?}"),
            Ok((Err(e), took)) => {
                eprintln!("case {name}: FAILED in {took:.2?}: {e:?}");
                failed.push(name);
            }
            // the panic hook has printed it
            Err(e) => {
                eprintln!("case {name}: FAILED: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "failing cases: {}", failed.join(", "));
}

/// `concurrent_cases!(test_name, [case_a: mib, case_b: mib, ...])`: one test running each `async fn() ->
/// anyhow::Result<()>` case concurrently, each claiming its VRAM from the budget.
macro_rules! concurrent_cases {
    ($test:ident, [$($case:ident: $mib:expr),+ $(,)?]) => {
        #[tokio::test(flavor = "multi_thread")]
        async fn $test() {
            concurrent_cases::run(vec![$(concurrent_cases::Case {
                name: stringify!($case),
                vram_mib: $mib,
                run: || Box::pin($case()),
            }),+])
            .await
        }
    };
}
