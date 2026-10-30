//! GPU test cases run in one test process. Each GPU process spends ~0.12 s in context creation, which the driver
//! serializes, so a suite's wall follows its count of GPU processes (docs/investigations/test_time_investigation.md).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

// a substring of case names: only those run, for iterating on one case
pub const CASE_FILTER_ENV: &str = "INFERENCE_TEST_CASE";

pub type Case<'a> = (&'a str, &'a dyn Fn() -> Result<(), String>);

/// What a case returns: `()` (it fails by panicking) or a `Result`.
pub trait Outcome {
    fn into_outcome(self) -> Result<(), String>;
}

impl Outcome for () {
    fn into_outcome(self) -> Result<(), String> {
        Ok(())
    }
}

impl<E: std::fmt::Debug> Outcome for Result<(), E> {
    fn into_outcome(self) -> Result<(), String> {
        self.map_err(|e| format!("{e:?}"))
    }
}

/// Runs every case (a panic too fails only its own) and then fails naming each failing case.
pub fn run(cases: &[Case]) {
    let only = std::env::var(CASE_FILTER_ENV).ok();
    let mut failed = Vec::new();
    for (name, case) in cases {
        if only.as_deref().is_some_and(|o| !name.contains(o)) {
            continue;
        }
        let start = Instant::now();
        let outcome = match catch_unwind(AssertUnwindSafe(case)) {
            Ok(outcome) => outcome,
            // the panic hook has printed it
            Err(_) => Err("panicked".to_string()),
        };
        match outcome {
            Ok(()) => eprintln!("case {name}: ok in {:.2?}", start.elapsed()),
            Err(e) => {
                eprintln!("case {name}: FAILED in {:.2?}: {e}", start.elapsed());
                failed.push(*name);
            }
        }
    }
    assert!(failed.is_empty(), "failing cases: {}", failed.join(", "));
}

/// `gpu_cases!(test_name, [case_a, case_b, ...])`: one `#[test]` running each case, a `fn()` returning an [`Outcome`].
#[doc(hidden)]
#[macro_export]
macro_rules! gpu_cases {
    ($test:ident, [$($case:ident),+ $(,)?]) => {
        #[test]
        fn $test() {
            $crate::gpu_cases::run(&[$((stringify!($case), &|| $crate::gpu_cases::Outcome::into_outcome($case()))),+]);
        }
    };
}
