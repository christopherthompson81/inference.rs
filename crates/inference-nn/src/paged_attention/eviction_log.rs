//! Why recently evicted prefix-cache entries went, so a lookup that misses can tell no space from no match.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;

/// A block's lease was revoked to make room for an allocation.
pub const BLOCK_PRESSURE: &str = "block_pressure";
/// A free cached block was taken for new data.
pub const BLOCK_REUSED: &str = "block_reused";
/// Keys the logs keep before forgetting the oldest.
pub const EVICTION_LOG_CAPACITY: usize = 1 << 16;

/// The cause of each recent eviction, oldest forgotten first; a key keeps its first cause until it is forgotten.
pub struct EvictionLog<K> {
    capacity: usize,
    order: VecDeque<(K, u64)>,
    causes: HashMap<K, (&'static str, u64)>,
    next: u64,
}

impl<K: Copy + Eq + Hash> EvictionLog<K> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
            causes: HashMap::new(),
            next: 0,
        }
    }

    pub fn record(&mut self, key: K, cause: &'static str) {
        if self.causes.contains_key(&key) {
            return;
        }
        self.causes.insert(key, (cause, self.next));
        self.order.push_back((key, self.next));
        self.next += 1;
        while self.causes.len() > self.capacity {
            let Some((old, stamp)) = self.order.pop_front() else {
                break;
            };
            if self.causes.get(&old).is_some_and(|(_, s)| *s == stamp) {
                self.causes.remove(&old);
            }
        }
    }

    /// Clears a key that is cached again, so a later eviction records its own cause.
    pub fn forget(&mut self, key: &K) {
        self.causes.remove(key);
    }

    pub fn cause(&self, key: &K) -> Option<&'static str> {
        self.causes.get(key).map(|(cause, _)| *cause)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_keeps_its_first_cause_until_forgotten_and_old_keys_go() {
        let mut log = EvictionLog::new(2);
        log.record(1u64, BLOCK_PRESSURE);
        log.record(1, BLOCK_REUSED);
        assert_eq!(log.cause(&1), Some(BLOCK_PRESSURE));
        log.forget(&1);
        log.record(1, BLOCK_REUSED);
        assert_eq!(log.cause(&1), Some(BLOCK_REUSED));
        log.record(2, BLOCK_REUSED);
        log.record(3, BLOCK_REUSED);
        assert_eq!(log.cause(&1), None);
        assert_eq!(log.cause(&3), Some(BLOCK_REUSED));
    }
}
