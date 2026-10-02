#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::info;

/// Cumulative speculative verification counters since load (or the last reset after warmup).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Sequences verified: one per sequence per step that ran speculative verification.
    pub drafts: usize,
    pub draft_tokens_proposed: usize,
    pub draft_tokens_accepted: usize,
    /// Accepted draft tokens at each 0-based proposal position.
    pub accepted_per_position: Vec<usize>,
}

#[derive(Default)]
struct PrefixCacheStats {
    hits: usize,
    total_sequences: usize,
}

pub struct IntervalLogger {
    enable_logging: Arc<AtomicBool>,
    prefix_cache_stats: Arc<Mutex<PrefixCacheStats>>,
    tokens_processed: Arc<AtomicUsize>,
    prefill_tokens_processed: Arc<AtomicUsize>,
    decode_tokens_processed: Arc<AtomicUsize>,
    num_running: Arc<AtomicUsize>,
    num_waiting: Arc<AtomicUsize>,
    sequence_capacity: Arc<AtomicUsize>,
    encoder_cache_hits: Option<Arc<AtomicUsize>>,
    encoder_cache_misses: Option<Arc<AtomicUsize>>,
    speculative: Arc<Mutex<SpeculativeStats>>,
    shutdown_tx: Sender<()>,
    worker: Option<JoinHandle<()>>,
    #[cfg(test)]
    worker_exited: Arc<AtomicBool>,
}

impl IntervalLogger {
    /// Starts an interval logger. Call `begin_logging` to begin the logging process.
    pub fn new(
        interval: Duration,
        encoder_cache_counters: Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)>,
    ) -> Self {
        let prefix_cache_stats = Arc::new(Mutex::new(PrefixCacheStats::default()));
        let tokens_processed = Arc::new(AtomicUsize::new(0));
        let prefill_tokens_processed = Arc::new(AtomicUsize::new(0));
        let decode_tokens_processed = Arc::new(AtomicUsize::new(0));
        let enable_logging = Arc::new(AtomicBool::new(false));
        let num_running = Arc::new(AtomicUsize::new(0));
        let num_waiting = Arc::new(AtomicUsize::new(0));
        let sequence_capacity = Arc::new(AtomicUsize::new(0));
        let speculative = Arc::new(Mutex::new(SpeculativeStats::default()));

        let t_prefix_cache_stats = prefix_cache_stats.clone();
        let t_tokens_processed = tokens_processed.clone();
        let t_prefill_tokens_processed = prefill_tokens_processed.clone();
        let t_decode_tokens_processed = decode_tokens_processed.clone();
        let t_enable_logging = enable_logging.clone();
        let t_num_running = num_running.clone();
        let t_num_waiting = num_waiting.clone();
        let t_sequence_capacity = sequence_capacity.clone();
        let t_speculative = speculative.clone();
        let (encoder_cache_hits, encoder_cache_misses) = match encoder_cache_counters {
            Some((h, m)) => (Some(h), Some(m)),
            None => (None, None),
        };
        let t_enc_hits = encoder_cache_hits.clone();
        let t_enc_misses = encoder_cache_misses.clone();
        #[cfg(test)]
        let worker_exited = Arc::new(AtomicBool::new(false));
        #[cfg(test)]
        let t_worker_exited = worker_exited.clone();
        let (shutdown_tx, shutdown_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let mut last_speculative = SpeculativeStats::default();
            while let Err(RecvTimeoutError::Timeout) = shutdown_rx.recv_timeout(interval) {
                let num_running = t_num_running.load(Ordering::Relaxed);
                let num_waiting = t_num_waiting.load(Ordering::Relaxed);
                metrics::gauge!("inference_sequences_running").set(num_running as f64);
                metrics::gauge!("inference_sequences_waiting").set(num_waiting as f64);
                metrics::gauge!("inference_sequences_capacity")
                    .set(t_sequence_capacity.load(Ordering::Relaxed) as f64);

                if !t_enable_logging.load(Ordering::Relaxed) {
                    continue;
                }

                let (prefix_cache_hits, total_new_seqs) = {
                    let stats = t_prefix_cache_stats.lock().unwrap();
                    (stats.hits, stats.total_sequences)
                };
                if let (Some(hits), Some(misses)) = (&t_enc_hits, &t_enc_misses) {
                    metrics::counter!("inference_encoder_cache_hits_total")
                        .absolute(hits.load(Ordering::Relaxed) as u64);
                    metrics::counter!("inference_encoder_cache_misses_total")
                        .absolute(misses.load(Ordering::Relaxed) as u64);
                }
                let tokens_processed = t_tokens_processed.swap(0, Ordering::Relaxed);
                let prefill_tokens_processed =
                    t_prefill_tokens_processed.swap(0, Ordering::Relaxed);
                let decode_tokens_processed = t_decode_tokens_processed.swap(0, Ordering::Relaxed);
                let speculative = t_speculative.lock().unwrap().clone();
                // counters below the last snapshot mean a reset in between, so the interval counts from zero
                if speculative.drafts < last_speculative.drafts {
                    last_speculative = SpeculativeStats::default();
                }
                let delta = |now: usize, last: usize| now.saturating_sub(last);
                let spec_drafts = delta(speculative.drafts, last_speculative.drafts);
                let spec_draft_tokens = delta(
                    speculative.draft_tokens_proposed,
                    last_speculative.draft_tokens_proposed,
                );
                let spec_accepted_tokens = delta(
                    speculative.draft_tokens_accepted,
                    last_speculative.draft_tokens_accepted,
                );
                last_speculative = speculative;

                if total_new_seqs != 0 && tokens_processed != 0 {
                    let enc_cache_info =
                        if let (Some(hits), Some(misses)) = (&t_enc_hits, &t_enc_misses) {
                            let h = hits.load(Ordering::Relaxed);
                            let m = misses.load(Ordering::Relaxed);
                            let total = h + m;
                            if total > 0 {
                                format!(
                                    ", Encoder cache hitrate {:.2}%",
                                    100. * h as f64 / total as f64
                                )
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };
                    let spec_info = if spec_draft_tokens > 0 {
                        // vLLM-style rates: accept rate over proposed draft tokens,
                        // mean acceptance length includes the bonus token.
                        let accept_rate =
                            100. * spec_accepted_tokens as f64 / spec_draft_tokens as f64;
                        let mean_len = 1. + spec_accepted_tokens as f64 / spec_drafts.max(1) as f64;
                        format!(", MTP accept {accept_rate:.1}% (len {mean_len:.2})")
                    } else {
                        String::new()
                    };

                    // Throughput = tokens processed during this interval / interval duration.
                    // The counter is atomically swapped to 0 each interval, so the metric
                    // reflects only the current window and is not cumulative.
                    info!(
                        "Throughput (T/s) {:.2} (prefill {:.2}, decode {:.2}), Prefix cache hitrate {:.2}%{enc_cache_info}{spec_info}, {num_running} running, {num_waiting} waiting",
                        tokens_processed as f64 / interval.as_secs_f64(),
                        prefill_tokens_processed as f64 / interval.as_secs_f64(),
                        decode_tokens_processed as f64 / interval.as_secs_f64(),
                        100. * prefix_cache_hits as f64 / total_new_seqs as f64,
                    );
                }
            }
            #[cfg(test)]
            t_worker_exited.store(true, Ordering::Release);
        });

        Self {
            prefix_cache_stats,
            tokens_processed,
            prefill_tokens_processed,
            decode_tokens_processed,
            enable_logging,
            num_running,
            num_waiting,
            sequence_capacity,
            encoder_cache_hits,
            encoder_cache_misses,
            speculative,
            shutdown_tx,
            worker: Some(worker),
            #[cfg(test)]
            worker_exited,
        }
    }

    pub fn enable_logging(&self) {
        self.enable_logging.store(true, Ordering::Relaxed);
    }

    /// Reset all counters to zero. Call after warmup/dummy runs to get clean stats.
    pub fn reset(&self) {
        *self.prefix_cache_stats.lock().unwrap() = PrefixCacheStats::default();
        self.tokens_processed.store(0, Ordering::Relaxed);
        self.prefill_tokens_processed.store(0, Ordering::Relaxed);
        self.decode_tokens_processed.store(0, Ordering::Relaxed);
        self.num_running.store(0, Ordering::Relaxed);
        self.num_waiting.store(0, Ordering::Relaxed);
        if let Some(ref hits) = self.encoder_cache_hits {
            hits.store(0, Ordering::Relaxed);
        }
        if let Some(ref misses) = self.encoder_cache_misses {
            misses.store(0, Ordering::Relaxed);
        }
        *self.speculative.lock().unwrap() = SpeculativeStats::default();
    }

    /// Count prompt (prefill) tokens through the pipeline. Also advances the
    /// combined `inference_tokens_processed_total` counter, which always equals
    /// prefill plus decode tokens.
    pub fn add_prefill_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        self.prefill_tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        metrics::counter!("inference_tokens_processed_total").increment(num_tokens as u64);
        metrics::counter!("inference_prefill_tokens_processed_total").increment(num_tokens as u64);
    }

    /// Count generated (decode) tokens through the pipeline. With speculative
    /// decoding only verified tokens are counted. Also advances the combined
    /// `inference_tokens_processed_total` counter.
    pub fn add_decode_tokens_processed(&self, num_tokens: usize) {
        self.tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        self.decode_tokens_processed
            .fetch_add(num_tokens, Ordering::Relaxed);
        metrics::counter!("inference_tokens_processed_total").increment(num_tokens as u64);
        metrics::counter!("inference_decode_tokens_processed_total").increment(num_tokens as u64);
    }

    /// Record one speculative verification batch (across all its sequences).
    pub fn add_speculative_stats(
        &self,
        num_drafts: usize,
        num_draft_tokens: usize,
        num_accepted_tokens: usize,
        accepted_per_pos: &[usize],
    ) {
        if num_drafts == 0 {
            return;
        }
        {
            let mut stats = self.speculative.lock().unwrap();
            stats.drafts += num_drafts;
            stats.draft_tokens_proposed += num_draft_tokens;
            stats.draft_tokens_accepted += num_accepted_tokens;
            if stats.accepted_per_position.len() < accepted_per_pos.len() {
                stats
                    .accepted_per_position
                    .resize(accepted_per_pos.len(), 0);
            }
            for (total, count) in stats.accepted_per_position.iter_mut().zip(accepted_per_pos) {
                *total += count;
            }
        }
        metrics::counter!("inference_speculative_drafts_total").increment(num_drafts as u64);
        metrics::counter!("inference_speculative_draft_tokens_proposed_total")
            .increment(num_draft_tokens as u64);
        metrics::counter!("inference_speculative_draft_tokens_accepted_total")
            .increment(num_accepted_tokens as u64);
        for (position, count) in accepted_per_pos.iter().enumerate() {
            if *count > 0 {
                metrics::counter!(
                    "inference_speculative_draft_tokens_accepted_per_pos_total",
                    "position" => position.to_string()
                )
                .increment(*count as u64);
            }
        }
    }

    pub fn add_new_sequence(&self) {
        self.prefix_cache_stats.lock().unwrap().total_sequences += 1;
        metrics::counter!("inference_prefix_cache_lookups_total").increment(1);
    }

    pub fn add_prefix_cache_hit(&self) {
        self.prefix_cache_stats.lock().unwrap().hits += 1;
        metrics::counter!("inference_prefix_cache_hits_total").increment(1);
    }

    pub fn set_num_running(&self, running: usize) {
        self.num_running.store(running, Ordering::Relaxed);
        metrics::gauge!("inference_sequences_running").set(running as f64);
    }

    pub fn set_num_waiting(&self, waiting: usize) {
        self.num_waiting.store(waiting, Ordering::Relaxed);
        metrics::gauge!("inference_sequences_waiting").set(waiting as f64);
    }

    pub fn set_sequence_capacity(&self, capacity: usize) {
        self.sequence_capacity.store(capacity, Ordering::Relaxed);
        metrics::gauge!("inference_sequences_capacity").set(capacity as f64);
    }

    /// Return cumulative prefix cache (hits, total_sequences).
    pub fn prefix_cache_stats(&self) -> (usize, usize) {
        let stats = self.prefix_cache_stats.lock().unwrap();
        (stats.hits, stats.total_sequences)
    }

    pub fn speculative_stats(&self) -> SpeculativeStats {
        self.speculative.lock().unwrap().clone()
    }

    /// Return cumulative encoder cache (hits, misses), or `None` if no encoder cache exists.
    pub fn encoder_cache_stats(&self) -> Option<(usize, usize)> {
        match (&self.encoder_cache_hits, &self.encoder_cache_misses) {
            (Some(h), Some(m)) => Some((h.load(Ordering::Relaxed), m.load(Ordering::Relaxed))),
            _ => None,
        }
    }
}

impl Drop for IntervalLogger {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INACTIVE_LOGGER_INTERVAL: Duration = Duration::from_secs(3600);
    const TEST_SEQUENCE_CAPACITY: usize = 16;

    #[test]
    fn sequence_capacity_is_retained_for_recurring_publication() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);

        logger.set_sequence_capacity(TEST_SEQUENCE_CAPACITY);

        assert_eq!(
            logger.sequence_capacity.load(Ordering::Relaxed),
            TEST_SEQUENCE_CAPACITY
        );
    }

    #[test]
    fn speculative_stats_accumulate_until_reset() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
        logger.add_speculative_stats(2, 4, 3, &[2, 1]);
        logger.add_speculative_stats(1, 3, 1, &[1, 0, 0]);
        assert_eq!(
            logger.speculative_stats(),
            SpeculativeStats {
                drafts: 3,
                draft_tokens_proposed: 7,
                draft_tokens_accepted: 4,
                accepted_per_position: vec![3, 1, 0],
            }
        );
        logger.reset();
        assert_eq!(logger.speculative_stats(), SpeculativeStats::default());
    }

    #[test]
    fn drop_wakes_and_joins_worker() {
        let logger = IntervalLogger::new(INACTIVE_LOGGER_INTERVAL, None);
        let worker_exited = logger.worker_exited.clone();

        drop(logger);

        assert!(worker_exited.load(Ordering::Acquire));
    }
}
