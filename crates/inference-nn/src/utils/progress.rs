use indicatif::{
    MultiProgress, ProgressBar, ProgressBarIter, ProgressDrawTarget, ProgressIterator,
    ProgressStyle,
};
use inference_quant::get_immediate_isq;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::iter::Iterator;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

static PROGRESS_SUPPRESS_COUNT: AtomicUsize = AtomicUsize::new(0);
static PROGRESS_SINKS: Mutex<Vec<(u64, Arc<ProgressSink>)>> = Mutex::new(Vec::new());
static NEXT_SINK: AtomicU64 = AtomicU64::new(0);
const PERCENT: u64 = 100;

/// What a progress stage counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgressUnit {
    Items,
    Bytes,
}

/// A step of a model load: `done` of `total` `unit`s of `stage` (`Loading repeating layers`).
#[derive(Clone, Debug, serde::Serialize)]
pub struct ProgressEvent<'a> {
    pub stage: &'a str,
    pub done: u64,
    pub total: u64,
    pub unit: ProgressUnit,
}

/// Receives the progress of every model load in the process, from whatever thread the load runs on.
pub type ProgressSink = dyn Fn(&ProgressEvent<'_>) + Send + Sync;

/// Keeps a sink receiving progress until dropped.
pub struct ProgressSinkGuard(u64);

impl Drop for ProgressSinkGuard {
    fn drop(&mut self) {
        let mut sinks = PROGRESS_SINKS
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        sinks.retain(|(id, _)| *id != self.0);
    }
}

/// Sends the progress of every model load in the process to `sink` while the guard lives.
pub fn add_progress_sink(sink: Arc<ProgressSink>) -> ProgressSinkGuard {
    let id = NEXT_SINK.fetch_add(1, Ordering::Relaxed);
    PROGRESS_SINKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push((id, sink));
    ProgressSinkGuard(id)
}

/// Whether any sink is listening, so a load reports progress even when it draws none.
pub fn progress_reported() -> bool {
    !PROGRESS_SINKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_empty()
}

fn emit(event: &ProgressEvent<'_>) {
    // a sink must not run under the lock: it may take long, or a load may start from it
    let sinks: Vec<Arc<ProgressSink>> = PROGRESS_SINKS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|(_, sink)| sink.clone())
        .collect();
    for sink in sinks {
        sink(event);
    }
}

/// Reports one stage to the sinks: at its start, at each whole percent, and at its end.
#[derive(Clone)]
pub struct StageReporter(Arc<StageState>);

struct StageState {
    stage: String,
    total: u64,
    unit: ProgressUnit,
    done: AtomicU64,
    // the last percent reported, under a lock so parallel steps reach the sinks in order
    reported: Mutex<Option<u64>>,
}

impl StageReporter {
    /// `None` when no sink listens, so an unobserved load counts nothing.
    pub fn new(stage: impl Into<String>, total: u64, unit: ProgressUnit) -> Option<Self> {
        if !progress_reported() {
            return None;
        }
        let reporter = Self(Arc::new(StageState {
            stage: stage.into(),
            total,
            unit,
            done: AtomicU64::new(0),
            reported: Mutex::new(None),
        }));
        reporter.report(0);
        Some(reporter)
    }

    pub fn inc(&self, n: u64) {
        let done = self.0.done.fetch_add(n, Ordering::Relaxed) + n;
        self.report(done);
    }

    /// Counts from zero again without reporting, as when a download restarts from a byte offset it then reports.
    pub fn restart(&self) {
        self.0.done.store(0, Ordering::Relaxed);
    }

    fn report(&self, done: u64) {
        let percent = match self.0.total {
            0 => PERCENT,
            total => done.min(total) * PERCENT / total,
        };
        let mut reported = self
            .0
            .reported
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if reported.is_some_and(|last| last >= percent) {
            return;
        }
        *reported = Some(percent);
        emit(&ProgressEvent {
            stage: &self.0.stage,
            done,
            total: self.0.total,
            unit: self.0.unit,
        });
    }
}

/// An iterator that reports each item to a stage once the next one is asked for, so the last counts when it is done.
pub struct Reported<I> {
    iter: I,
    reporter: Option<StageReporter>,
    pending: bool,
}

impl<I> Reported<I> {
    pub fn new(iter: I, reporter: Option<StageReporter>) -> Self {
        Self {
            iter,
            reporter,
            pending: false,
        }
    }
}

impl<I: Iterator> Iterator for Reported<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if std::mem::take(&mut self.pending)
            && let Some(reporter) = &self.reporter
        {
            reporter.inc(1);
        }
        let item = self.iter.next();
        self.pending = item.is_some();
        item
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// RAII guard that suppresses progress bar drawing while it is alive.
pub struct ProgressScopeGuard {
    suppressed: bool,
}

impl ProgressScopeGuard {
    pub fn new(silent: bool) -> Self {
        if silent {
            PROGRESS_SUPPRESS_COUNT.fetch_add(1, Ordering::SeqCst);
        }
        Self { suppressed: silent }
    }
}

impl Drop for ProgressScopeGuard {
    fn drop(&mut self) {
        if self.suppressed {
            PROGRESS_SUPPRESS_COUNT.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

#[inline]
pub fn progress_suppressed() -> bool {
    PROGRESS_SUPPRESS_COUNT.load(Ordering::SeqCst) > 0
}

#[inline]
pub fn configure_progress_bar(bar: &ProgressBar) {
    if progress_suppressed() {
        bar.set_draw_target(ProgressDrawTarget::hidden());
    }
}

/// A `MultiProgress` for a load that may be `silent`: drawn unless silent, and made at all when a sink listens.
pub fn multi_progress_unless_silent(silent: bool) -> Option<MultiProgress> {
    (!silent || progress_reported()).then(|| {
        let multi = new_multi_progress();
        if silent {
            multi.set_draw_target(ProgressDrawTarget::hidden());
        }
        multi
    })
}

pub fn new_multi_progress() -> MultiProgress {
    let multi = MultiProgress::new();
    if progress_suppressed() {
        multi.set_draw_target(ProgressDrawTarget::hidden());
    }
    multi
}

/// `iter.with_progress(is_silent)` shows a progress bar over the iterator unless silent (or suppressed).
pub trait IterWithProgress<'a, T>: Iterator<Item = T> + 'a {
    fn with_progress(self, is_silent: bool) -> Box<dyn Iterator<Item = T> + 'a>
    where
        Self: Sized,
    {
        if is_silent {
            return Box::new(self);
        }
        let bar = match self.size_hint() {
            (lower, Some(upper)) if lower == upper => ProgressBar::new(upper as u64),
            _ => ProgressBar::no_length(),
        };
        configure_progress_bar(&bar);
        Box::new(self.progress_with(bar))
    }
}

impl<'a, T: Iterator + 'a> IterWithProgress<'a, T::Item> for T {}

/// Nice progress bar with over an iterator and a message.
/// COLOR is one of r,g,b
pub struct NiceProgressBar<
    'a,
    T: ExactSizeIterator,
    const COLOR: char = 'b',
    L: AsRef<str> = &'static str,
>(pub T, pub L, pub &'a MultiProgress);

impl<T: ExactSizeIterator, const COLOR: char, L: AsRef<str>> IntoIterator
    for NiceProgressBar<'_, T, COLOR, L>
{
    type IntoIter = Reported<ProgressBarIter<T>>;
    type Item = T::Item;

    fn into_iter(self) -> Self::IntoIter {
        let color = match COLOR {
            'b' => "blue",
            'g' => "green",
            'r' => "red",
            other => panic!("Color char `{other}` not supported"),
        };
        let bar = ProgressBar::new(self.0.len() as u64);
        configure_progress_bar(&bar);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{}: [{{elapsed_precise}}] [{{bar:40.{color}/{color}}}] {{pos}}/{{len}} ({{eta}})",
                    self.1.as_ref()
                ))
                .unwrap()
                .progress_chars("#>-"),
        );

        // Add to the multi progress
        self.2.add(bar.clone());

        let reporter =
            StageReporter::new(self.1.as_ref(), self.0.len() as u64, ProgressUnit::Items);
        Reported::new(self.0.progress_with(bar), reporter)
    }
}

/// Parallel iterator with progress reporting.
pub struct ParProgress<I> {
    iter: I,
    bar: ProgressBar,
    reporter: Option<StageReporter>,
}

impl<I> ParallelIterator for ParProgress<I>
where
    I: ParallelIterator,
    I::Item: Send,
{
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let reporter = self.reporter.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            if let Some(reporter) = &reporter {
                reporter.inc(1);
            }
            item
        });
        iter.drive_unindexed(consumer)
    }
}

impl<I> IndexedParallelIterator for ParProgress<I>
where
    I: IndexedParallelIterator,
    I::Item: Send,
{
    fn len(&self) -> usize {
        self.iter.len()
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let bar = self.bar.clone();
        let reporter = self.reporter.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            if let Some(reporter) = &reporter {
                reporter.inc(1);
            }
            item
        });
        iter.drive(consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        let bar = self.bar.clone();
        let reporter = self.reporter.clone();
        let iter = self.iter.map(move |item| {
            bar.inc(1);
            if let Some(reporter) = &reporter {
                reporter.inc(1);
            }
            item
        });
        iter.with_producer(callback)
    }
}

impl<'a, T, const COLOR: char, L: AsRef<str>> IntoParallelIterator
    for NiceProgressBar<'a, T, COLOR, L>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
{
    type Iter = ParProgress<T::Iter>;
    type Item = <T as IntoParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        let color = match COLOR {
            'b' => "blue",
            'g' => "green",
            'r' => "red",
            other => panic!("Color char `{other}` not supported"),
        };
        let bar = ProgressBar::new(self.0.len() as u64);
        configure_progress_bar(&bar);
        bar.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{}: [{{elapsed_precise}}] [{{bar:40.{color}/{color}}}] {{pos}}/{{len}} ({{eta}})",
                    self.1.as_ref()
                ))
                .unwrap()
                .progress_chars("#>-"),
        );
        self.2.add(bar.clone());
        let reporter =
            StageReporter::new(self.1.as_ref(), self.0.len() as u64, ProgressUnit::Items);
        ParProgress {
            iter: self.0.into_par_iter(),
            bar,
            reporter,
        }
    }
}

impl<'a, T, const COLOR: char, L: AsRef<str>> NiceProgressBar<'a, T, COLOR, L>
where
    T: ExactSizeIterator + IntoParallelIterator + Send + Sync + 'a,
    <T as IntoParallelIterator>::Item: Send + 'a,
    T::Iter: ParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + IndexedParallelIterator<Item = <T as IntoParallelIterator>::Item>
        + Send,
    T: IntoParallelIterator<Item = <T as Iterator>::Item>,
{
    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `is_parallel`: If true, uses Rayon parallel iteration; otherwise uses sequential iteration.
    /// - `f`: A closure to apply to each item.
    pub fn run<F, U>(self, _is_parallel: bool, f: F) -> inference_tensor::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> inference_tensor::Result<U> + Sync + Send,
        U: Send,
    {
        // if is_parallel {
        //     self.into_par_iter().map(f).collect()
        // } else {
        //     self.into_iter().map(f).collect()
        // }
        self.into_iter().map(f).collect()
    }

    /// Applies the given closure over the items, optionally in parallel, and collects the results.
    ///
    /// - `f`: A closure to apply to each item.
    pub fn par_iter_if_isq<F, U>(self, f: F) -> inference_tensor::Result<Vec<U>>
    where
        F: Fn(<T as IntoParallelIterator>::Item) -> inference_tensor::Result<U> + Sync + Send,
        U: Send,
    {
        self.run(get_immediate_isq().is_some_and(|x| x.ty.is_some()), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the events a sink hears while `body` runs, as (stage, done, total)
    fn heard(body: impl FnOnce()) -> Vec<(String, u64, u64)> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let sink = events.clone();
        let guard = add_progress_sink(Arc::new(move |event: &ProgressEvent<'_>| {
            sink.lock()
                .unwrap()
                .push((event.stage.to_string(), event.done, event.total));
        }));
        body();
        drop(guard);
        Arc::try_unwrap(events).unwrap().into_inner().unwrap()
    }

    #[test]
    fn a_stage_reports_its_start_each_percent_and_its_end() {
        let events = heard(|| {
            let multi = new_multi_progress();
            for _ in NiceProgressBar::<_, 'b', _>(0..250, "Loading things".to_string(), &multi) {}
        });
        assert_eq!(events.first().map(|e| e.1), Some(0), "{events:?}");
        assert_eq!(
            events.last().map(|e| (e.1, e.2)),
            Some((250, 250)),
            "{events:?}"
        );
        assert!(events.len() <= 101, "{} events", events.len());
        assert!(
            events.windows(2).all(|pair| pair[0].1 < pair[1].1),
            "{events:?}"
        );
    }

    #[test]
    fn an_item_counts_once_the_next_is_asked_for() {
        let events = heard(|| {
            let multi = new_multi_progress();
            let mut items = NiceProgressBar::<_, 'b'>(0..2, "Loading", &multi).into_iter();
            items.next();
        });
        assert_eq!(
            events,
            [("Loading".to_string(), 0, 2)],
            "the first item was not done yet"
        );
    }

    #[test]
    fn a_restarted_count_never_reports_backwards() {
        let events = heard(|| {
            let reporter = StageReporter::new("Downloading a", 100, ProgressUnit::Bytes).unwrap();
            reporter.inc(40);
            reporter.restart();
            reporter.inc(40);
            reporter.inc(60);
        });
        let done: Vec<u64> = events.iter().map(|e| e.1).collect();
        assert_eq!(done, [0, 40, 100]);
    }

    #[test]
    fn an_empty_stage_is_done_from_the_start_and_nothing_counts_unheard() {
        let events = heard(|| {
            StageReporter::new("Downloading empty", 0, ProgressUnit::Bytes);
        });
        assert_eq!(events, [("Downloading empty".to_string(), 0, 0)]);
        assert!(StageReporter::new("unheard", 10, ProgressUnit::Items).is_none());
    }
}
