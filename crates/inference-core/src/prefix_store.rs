//! Paged prefix cache entries in host memory and on disk, so a long agent session's history survives GPU eviction and,
//! on disk, restarts.
//!
//! A block (one paged KV block of every attention layer, plus its token ids) is stored once under its chain hash; an
//! entry is the recurrent state (and DFlash checkpoint) at a block boundary plus its chain of block hashes. With both
//! tiers, memory is in front: a write goes to both and a disk hit is copied into memory. On disk the key file is
//! written last, so an entry with one is complete. An entry expires its TTL after its last use (on disk the key
//! file's modification time); expired entries go with their unshared blocks.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, mpsc},
    time::{Duration, SystemTime},
};

use inference_tensor::{DType, Device, Result, Tensor};

use crate::kv_cache::{PagedAuxiliaryPrefixData, RecurrentSnapshotState};

const BLOCKS_DIR: &str = "blocks";
const ENTRIES_DIR: &str = "entries";
const KEY_FILE: &str = "key.bin";
const STATE_FILE: &str = "state.safetensors";
const NAMESPACE_FILE: &str = "namespace.txt";
// What every store's namespace opens with. Bump it when a model changes what its cached state means without changing
// its shapes (QSA's pooled keys moving to a block's last cell): stores under another header are removed on open
const LAYOUT_HEADER: &str = "layout 1";
// the entry's TTL in whole seconds; an entry without one takes the store's default
const TTL_FILE: &str = "ttl.txt";
// how often the writer sweeps expired entries when no write wakes it
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);
const PARTIAL_SUFFIX: &str = ".partial";
const SAFETENSORS_EXT: &str = "safetensors";
const TOKENS: &str = "tokens";
const AUX_KEY: &str = "auxiliary.key";
const AUX_VALUE: &str = "auxiliary.value";
// origin, valid start and next committed positions of the DFlash checkpoint
const AUX_POSITIONS: &str = "auxiliary.positions";
const HASH_BYTES: usize = 8;

/// One paged block on the host: its chain hash, tokens, and each attention layer's K and V rows as `[1, ...]`.
#[derive(Clone, Debug)]
pub struct StoredBlock {
    pub hash: u64,
    pub tokens: Vec<u32>,
    pub layers: Vec<(usize, Tensor, Tensor)>,
}

/// A prefix entry to persist: its block chain, the blocks the store lacks, and the state at its end.
pub struct PrefixWrite {
    pub key: Vec<u64>,
    pub blocks: Vec<StoredBlock>,
    pub recurrent: Vec<RecurrentSnapshotState>,
    pub auxiliary: Option<PagedAuxiliaryPrefixData>,
    /// How long the entry outlives its last use; `None` takes the store's default.
    pub ttl: Option<Duration>,
}

enum StoreOp {
    Write(PrefixWrite),
    // a hit on a stored entry: its last use is now, under this TTL
    Touch(Vec<u64>, Option<Duration>),
}

/// A prefix entry read back, its blocks in chain order.
pub struct LoadedPrefix {
    pub blocks: Vec<StoredBlock>,
    pub recurrent: Vec<RecurrentSnapshotState>,
    pub auxiliary: Option<PagedAuxiliaryPrefixData>,
}

struct StoredEntry {
    key: Vec<u64>,
    bytes: u64,
    last_used: SystemTime,
    ttl: Duration,
}

impl StoredEntry {
    fn expired(&self, now: SystemTime) -> bool {
        now.duration_since(self.last_used)
            .is_ok_and(|idle| idle > self.ttl)
    }
}

#[derive(Default)]
struct StoreIndex {
    blocks: HashMap<u64, u64>,
    entries: HashMap<String, StoredEntry>,
}

impl StoreIndex {
    fn bytes(&self) -> u64 {
        self.blocks.values().sum::<u64>() + self.entries.values().map(|e| e.bytes).sum::<u64>()
    }
}

/// Where a host prefix cache lives (memory, a directory, or both), how much each may hold, how long an entry outlives
/// its last use, and the model it serves.
#[derive(Clone, Debug)]
pub struct PrefixStoreSettings {
    pub dir: Option<PathBuf>,
    pub capacity_bytes: u64,
    /// Host memory for entries in front of `dir`, or on their own without one; 0 keeps none in memory.
    pub memory_bytes: u64,
    /// For entries whose requests name no TTL of their own.
    pub default_ttl: Duration,
    /// Tokens between a long prompt's recurrent checkpoints written here, so a request diverging early in it redoes at
    /// most this many; 0 keeps only the final ones.
    pub checkpoint_tokens: usize,
    /// Filled from the model's loader when the engine starts; entries are kept apart per identity.
    pub model_identity: String,
}

// Weight files a local path names: the file itself, or a directory's own weight files
const WEIGHT_EXTENSIONS: [&str; 3] = ["safetensors", "gguf", "bin"];

/// Size and modification time of the weight files at `path` (a file, or a directory's weight files), one line each.
pub fn local_weight_stamps(path: &Path) -> Vec<String> {
    let stamp = |file: &Path| {
        fs::metadata(file).ok().map(|meta| {
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            format!("{} {} {modified}", file.display(), meta.len())
        })
    };
    if path.is_file() {
        return stamp(path).into_iter().collect();
    }
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut stamps = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|file| {
            file.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| WEIGHT_EXTENSIONS.contains(&ext))
        })
        .filter_map(|file| stamp(&file))
        .collect::<Vec<_>>();
    stamps.sort();
    stamps
}

struct DiskStore {
    root: PathBuf,
    index: Arc<Mutex<StoreIndex>>,
    writes: Mutex<mpsc::Sender<StoreOp>>,
}

fn entry_id(key: &[u64]) -> String {
    format!(
        "{:016x}-{}",
        key.last().copied().unwrap_or_default(),
        key.len()
    )
}

fn block_path(root: &Path, hash: u64) -> PathBuf {
    let name = format!("{hash:016x}");
    root.join(BLOCKS_DIR)
        .join(&name[..2])
        .join(format!("{name}.{SAFETENSORS_EXT}"))
}

fn io(error: std::io::Error) -> inference_tensor::Error {
    inference_tensor::Error::msg(format!("prefix store: {error}"))
}

fn file_bytes(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or_default()
}

// Written beside its destination and renamed over it, so readers never see half a file
fn save_atomic(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let partial = path.with_extension(format!("{SAFETENSORS_EXT}{PARTIAL_SUFFIX}"));
    inference_tensor::safetensors::save(tensors, &partial)?;
    fs::rename(&partial, path).map_err(io)
}

fn read_ttl(dir: &Path) -> Option<Duration> {
    let text = fs::read_to_string(dir.join(TTL_FILE)).ok()?;
    text.trim().parse().ok().map(Duration::from_secs)
}

fn write_ttl(dir: &Path, ttl: Duration) {
    let _ = fs::write(dir.join(TTL_FILE), ttl.as_secs().to_string());
}

fn read_key(path: &Path) -> Option<Vec<u64>> {
    let bytes = fs::read(path).ok()?;
    let (hashes, rest) = bytes.as_chunks::<HASH_BYTES>();
    if hashes.is_empty() || !rest.is_empty() {
        return None;
    }
    Some(
        hashes
            .iter()
            .map(|hash| u64::from_le_bytes(*hash))
            .collect(),
    )
}

impl DiskStore {
    // stores under `dir` whose namespace opens with another layout header; no store of this build can reach them
    fn remove_other_layouts(dir: &Path) -> Result<()> {
        let Ok(stores) = fs::read_dir(dir) else {
            return Ok(());
        };
        for store in stores.flatten() {
            let Ok(namespace) = fs::read_to_string(store.path().join(NAMESPACE_FILE)) else {
                continue;
            };
            if namespace.lines().next() != Some(LAYOUT_HEADER) {
                fs::remove_dir_all(store.path()).map_err(io)?;
            }
        }
        Ok(())
    }

    /// The store for one model setup (`namespace`: weights, dtypes, cache layout) under `dir`, holding at most
    /// `capacity_bytes`; entries a crash left half written, and expired ones, are removed.
    fn open(
        dir: &Path,
        namespace: &str,
        capacity_bytes: u64,
        default_ttl: Duration,
    ) -> Result<Self> {
        use std::hash::{Hash, Hasher};
        let namespace = format!("{LAYOUT_HEADER}\n{namespace}");
        let mut hasher = siphasher::sip::SipHasher13::new();
        namespace.hash(&mut hasher);
        let root = dir.join(format!("{:016x}", hasher.finish()));
        Self::remove_other_layouts(dir)?;
        fs::create_dir_all(root.join(BLOCKS_DIR)).map_err(io)?;
        fs::create_dir_all(root.join(ENTRIES_DIR)).map_err(io)?;
        fs::write(root.join(NAMESPACE_FILE), namespace).map_err(io)?;
        let index = Arc::new(Mutex::new(Self::scan(&root, default_ttl)?));
        let (sender, receiver) = mpsc::channel::<StoreOp>();
        let writer = WriterCtx {
            root: root.clone(),
            capacity_bytes,
            default_ttl,
            index: Arc::clone(&index),
        };
        writer.expire(&mut index.lock().expect("prefix store index poisoned"));
        std::thread::Builder::new()
            .name("prefix-store-writer".to_string())
            .spawn(move || {
                loop {
                    match receiver.recv_timeout(SWEEP_INTERVAL) {
                        Ok(StoreOp::Write(write)) => {
                            if let Err(error) = writer.write(write) {
                                tracing::warn!("Prefix store write failed: {error}");
                            }
                        }
                        Ok(StoreOp::Touch(key, ttl)) => writer.touch(&key, ttl),
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    writer.expire(&mut writer.index.lock().expect("prefix store index poisoned"));
                }
            })
            .map_err(io)?;
        let stored = index
            .lock()
            .expect("prefix store index poisoned")
            .entries
            .len();
        tracing::info!(root = %root.display(), entries = stored, "Prefix cache disk store opened");
        Ok(Self {
            root,
            index,
            writes: Mutex::new(sender),
        })
    }

    fn scan(root: &Path, default_ttl: Duration) -> Result<StoreIndex> {
        let mut index = StoreIndex::default();
        for shard in fs::read_dir(root.join(BLOCKS_DIR)).map_err(io)?.flatten() {
            for file in fs::read_dir(shard.path()).map_err(io)?.flatten() {
                let path = file.path();
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                if name.ends_with(PARTIAL_SUFFIX) {
                    let _ = fs::remove_file(&path);
                } else if let Some(hash) = name
                    .strip_suffix(&format!(".{SAFETENSORS_EXT}"))
                    .and_then(|hex| u64::from_str_radix(hex, 16).ok())
                {
                    index.blocks.insert(hash, file_bytes(&path));
                }
            }
        }
        for entry in fs::read_dir(root.join(ENTRIES_DIR)).map_err(io)?.flatten() {
            let dir = entry.path();
            let id = entry.file_name().to_string_lossy().into_owned();
            let key_path = dir.join(KEY_FILE);
            match read_key(&key_path) {
                Some(key) if dir.join(STATE_FILE).exists() => {
                    let last_used = fs::metadata(&key_path)
                        .and_then(|m| m.modified())
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    let bytes = file_bytes(&key_path) + file_bytes(&dir.join(STATE_FILE));
                    index.entries.insert(
                        id,
                        StoredEntry {
                            key,
                            bytes,
                            last_used,
                            ttl: read_ttl(&dir).unwrap_or(default_ttl),
                        },
                    );
                }
                _ => {
                    let _ = fs::remove_dir_all(&dir);
                }
            }
        }
        Ok(index)
    }

    /// Whether the store has the block with this chain hash.
    fn has_block(&self, hash: u64) -> bool {
        self.index
            .lock()
            .expect("prefix store index poisoned")
            .blocks
            .contains_key(&hash)
    }

    /// Whether an entry with exactly this block chain is stored.
    fn has_entry(&self, key: &[u64]) -> bool {
        self.index
            .lock()
            .expect("prefix store index poisoned")
            .entries
            .contains_key(&entry_id(key))
    }

    /// Queues an entry for the writer thread.
    fn write(&self, write: PrefixWrite) {
        self.send(StoreOp::Write(write));
    }

    /// Marks a stored entry used now, under `ttl` (`None`: the store's default), on the writer thread.
    fn touch(&self, key: &[u64], ttl: Option<Duration>) {
        self.send(StoreOp::Touch(key.to_vec(), ttl));
    }

    fn send(&self, op: StoreOp) {
        let _ = self
            .writes
            .lock()
            .expect("prefix store queue poisoned")
            .send(op);
    }

    /// The longest stored entry whose chain starts `hashes` and is longer than `min_blocks`: its block count.
    fn longest_entry(&self, hashes: &[u64], min_blocks: usize) -> Option<usize> {
        let index = self.index.lock().expect("prefix store index poisoned");
        index
            .entries
            .values()
            .filter(|entry| entry.key.len() > min_blocks && hashes.starts_with(&entry.key))
            .map(|entry| entry.key.len())
            .max()
    }

    /// The entry for the first `n_blocks` of `hashes`, its blocks checked against `tokens`, marked used now under
    /// `ttl`; `None` (and the entry dropped) when a block is missing or holds other tokens.
    fn load(
        &self,
        hashes: &[u64],
        tokens: &[u32],
        block_size: usize,
        ttl: Option<Duration>,
    ) -> Result<Option<LoadedPrefix>> {
        let key = hashes.to_vec();
        let id = entry_id(&key);
        let dir = self.root.join(ENTRIES_DIR).join(&id);
        let loaded = self.read_entry(&dir, &key, tokens, block_size);
        let mut index = self.index.lock().expect("prefix store index poisoned");
        match loaded {
            Ok(Some(entry)) => {
                drop(index);
                self.touch(&key, ttl);
                Ok(Some(entry))
            }
            Ok(None) | Err(_) => {
                index.entries.remove(&id);
                let _ = fs::remove_dir_all(&dir);
                loaded
            }
        }
    }

    fn read_entry(
        &self,
        dir: &Path,
        key: &[u64],
        tokens: &[u32],
        block_size: usize,
    ) -> Result<Option<LoadedPrefix>> {
        if read_key(&dir.join(KEY_FILE)).as_deref() != Some(key) {
            return Ok(None);
        }
        let mut blocks = Vec::with_capacity(key.len());
        for (i, &hash) in key.iter().enumerate() {
            let path = block_path(&self.root, hash);
            let Ok(mut tensors) = inference_tensor::safetensors::load(&path, &Device::Cpu) else {
                return Ok(None);
            };
            let stored = tensors
                .remove(TOKENS)
                .map(|t| t.to_vec1::<u32>())
                .transpose()?;
            if stored.as_deref() != tokens.get(i * block_size..(i + 1) * block_size) {
                return Ok(None);
            }
            let mut layers = HashMap::<usize, (Option<Tensor>, Option<Tensor>)>::new();
            for (name, tensor) in tensors {
                let Some((kind, layer)) = name.split_once('.') else {
                    continue;
                };
                let Ok(layer) = layer.parse::<usize>() else {
                    continue;
                };
                let slot = layers.entry(layer).or_default();
                match kind {
                    "k" => slot.0 = Some(tensor),
                    "v" => slot.1 = Some(tensor),
                    _ => {}
                }
            }
            let mut layers = layers
                .into_iter()
                .map(|(layer, (k, v))| Some((layer, k?, v?)))
                .collect::<Option<Vec<_>>>()
                .unwrap_or_default();
            layers.sort_by_key(|(layer, _, _)| *layer);
            blocks.push(StoredBlock {
                hash,
                tokens: stored.unwrap_or_default(),
                layers,
            });
        }
        let mut state = inference_tensor::safetensors::load(dir.join(STATE_FILE), &Device::Cpu)?;
        let mut recurrent = Vec::new();
        while let (Some(conv), Some(rec)) = (
            state.remove(&format!("conv.{}", recurrent.len())),
            state.remove(&format!("recurrent.{}", recurrent.len())),
        ) {
            recurrent.push(RecurrentSnapshotState {
                conv,
                recurrent: rec,
            });
        }
        let auxiliary = match (
            state.remove(AUX_KEY),
            state.remove(AUX_VALUE),
            state.remove(AUX_POSITIONS),
        ) {
            (Some(key), Some(value), Some(positions)) => {
                let positions = positions
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(usize::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>();
                let Some(&[origin_pos, valid_start_pos, next_committed_pos]) =
                    positions.as_deref().ok()
                else {
                    return Ok(None);
                };
                Some(PagedAuxiliaryPrefixData {
                    key,
                    value,
                    origin_pos,
                    valid_start_pos,
                    next_committed_pos,
                })
            }
            _ => None,
        };
        Ok(Some(LoadedPrefix {
            blocks,
            recurrent,
            auxiliary,
        }))
    }
}

fn tensor_bytes(tensor: &Tensor) -> u64 {
    (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64
}

fn block_bytes(block: &StoredBlock) -> u64 {
    let tokens = (block.tokens.len() * std::mem::size_of::<u32>()) as u64;
    let rows = block
        .layers
        .iter()
        .map(|(_, k, v)| tensor_bytes(k) + tensor_bytes(v));
    tokens + rows.sum::<u64>()
}

fn state_bytes(write: &PrefixWrite) -> u64 {
    let recurrent = write
        .recurrent
        .iter()
        .map(|r| tensor_bytes(&r.conv) + tensor_bytes(&r.recurrent));
    let auxiliary = write
        .auxiliary
        .iter()
        .map(|a| tensor_bytes(&a.key) + tensor_bytes(&a.value));
    recurrent.sum::<u64>() + auxiliary.sum::<u64>()
}

struct MemoryEntry {
    key: Vec<u64>,
    recurrent: Vec<RecurrentSnapshotState>,
    auxiliary: Option<PagedAuxiliaryPrefixData>,
    bytes: u64,
    last_used: SystemTime,
    ttl: Duration,
}

impl MemoryEntry {
    fn expired(&self, now: SystemTime) -> bool {
        now.duration_since(self.last_used)
            .is_ok_and(|idle| idle > self.ttl)
    }
}

#[derive(Default)]
struct MemoryIndex {
    blocks: HashMap<u64, (StoredBlock, u64)>,
    entries: HashMap<String, MemoryEntry>,
}

impl MemoryIndex {
    fn bytes(&self) -> u64 {
        self.blocks.values().map(|(_, bytes)| bytes).sum::<u64>()
            + self.entries.values().map(|e| e.bytes).sum::<u64>()
    }

    fn live(&self, id: &str, now: SystemTime) -> Option<&MemoryEntry> {
        self.entries.get(id).filter(|entry| !entry.expired(now))
    }

    // The entry, then the blocks no entry still names
    fn remove_entry(&mut self, id: &str) {
        self.entries.remove(id);
        let named = self
            .entries
            .values()
            .flat_map(|entry| entry.key.iter().copied())
            .collect::<HashSet<_>>();
        self.blocks.retain(|hash, _| named.contains(hash));
    }

    fn expire(&mut self, now: SystemTime) {
        let expired = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.expired(now))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        for id in &expired {
            self.remove_entry(id);
        }
    }

    // Least recently used entries go until the store fits, always keeping the newest
    fn collect(&mut self, capacity_bytes: u64) {
        let mut by_age = self
            .entries
            .iter()
            .map(|(id, entry)| (entry.last_used, id.clone()))
            .collect::<Vec<_>>();
        by_age.sort();
        for (_, id) in by_age {
            if self.bytes() <= capacity_bytes || self.entries.len() <= 1 {
                break;
            }
            self.remove_entry(&id);
        }
    }
}

/// Entries in host memory, kept as a disk store keeps them, without files.
struct MemoryStore {
    capacity_bytes: u64,
    default_ttl: Duration,
    index: Mutex<MemoryIndex>,
}

impl MemoryStore {
    fn index(&self) -> std::sync::MutexGuard<'_, MemoryIndex> {
        self.index.lock().expect("prefix memory store poisoned")
    }

    fn has_block(&self, hash: u64) -> bool {
        self.index().blocks.contains_key(&hash)
    }

    fn has_entry(&self, key: &[u64]) -> bool {
        self.index()
            .live(&entry_id(key), SystemTime::now())
            .is_some()
    }

    fn longest_entry(&self, hashes: &[u64], min_blocks: usize) -> Option<usize> {
        let now = SystemTime::now();
        self.index()
            .entries
            .values()
            .filter(|entry| {
                !entry.expired(now)
                    && entry.key.len() > min_blocks
                    && hashes.starts_with(&entry.key)
            })
            .map(|entry| entry.key.len())
            .max()
    }

    // Holds `write`, returning its whole chain's blocks: those it carried and those already held
    fn insert(&self, write: &PrefixWrite) -> Vec<StoredBlock> {
        let now = SystemTime::now();
        let mut index = self.index();
        index.expire(now);
        for block in &write.blocks {
            index
                .blocks
                .entry(block.hash)
                .or_insert_with(|| (block.clone(), block_bytes(block)));
        }
        let chain = write
            .key
            .iter()
            .filter_map(|hash| index.blocks.get(hash).map(|(block, _)| block.clone()))
            .collect();
        index.entries.insert(
            entry_id(&write.key),
            MemoryEntry {
                key: write.key.clone(),
                recurrent: write.recurrent.clone(),
                auxiliary: write.auxiliary.clone(),
                bytes: state_bytes(write),
                last_used: now,
                ttl: write.ttl.unwrap_or(self.default_ttl),
            },
        );
        index.collect(self.capacity_bytes);
        chain
    }

    fn touch(&self, key: &[u64], ttl: Option<Duration>) {
        if let Some(entry) = self.index().entries.get_mut(&entry_id(key)) {
            entry.last_used = SystemTime::now();
            entry.ttl = ttl.unwrap_or(self.default_ttl);
        }
    }

    // As `DiskStore::load`: `None`, and the entry dropped, when a block is missing or holds other tokens
    fn load(
        &self,
        hashes: &[u64],
        tokens: &[u32],
        block_size: usize,
        ttl: Option<Duration>,
    ) -> Option<LoadedPrefix> {
        let now = SystemTime::now();
        let id = entry_id(hashes);
        let mut index = self.index();
        index.expire(now);
        let entry = index.live(&id, now).filter(|entry| entry.key == hashes)?;
        let (recurrent, auxiliary) = (entry.recurrent.clone(), entry.auxiliary.clone());
        let blocks = hashes
            .iter()
            .enumerate()
            .map(|(i, hash)| {
                let (block, _) = index.blocks.get(hash)?;
                (Some(block.tokens.as_slice()) == tokens.get(i * block_size..(i + 1) * block_size))
                    .then(|| block.clone())
            })
            .collect::<Option<Vec<_>>>();
        let Some(blocks) = blocks else {
            index.remove_entry(&id);
            return None;
        };
        drop(index);
        self.touch(hashes, ttl);
        Some(LoadedPrefix {
            blocks,
            recurrent,
            auxiliary,
        })
    }
}

/// The prefix cache's host tiers: entries in memory, in a directory, or both.
pub struct PrefixStore {
    memory: Option<MemoryStore>,
    disk: Option<DiskStore>,
}

impl PrefixStore {
    /// The store for one model setup (`namespace`: weights, dtypes, cache layout): `memory_bytes` of host memory and,
    /// with `dir`, up to `disk_bytes` under it; entries a crash left half written, and expired ones, are removed.
    pub fn open(
        dir: Option<&Path>,
        namespace: &str,
        disk_bytes: u64,
        memory_bytes: u64,
        default_ttl: Duration,
    ) -> Result<Arc<Self>> {
        let disk = dir
            .map(|dir| DiskStore::open(dir, namespace, disk_bytes, default_ttl))
            .transpose()?;
        let memory = (memory_bytes > 0).then(|| {
            tracing::info!(bytes = memory_bytes, "Prefix cache memory store opened");
            MemoryStore {
                capacity_bytes: memory_bytes,
                default_ttl,
                index: Mutex::default(),
            }
        });
        Ok(Arc::new(Self { memory, disk }))
    }

    /// Whether a write may leave out this block: memory holds it, or the disk does when there is no memory tier
    /// (writes go to memory first, and it hands the disk any block the disk lacks).
    pub fn has_block(&self, hash: u64) -> bool {
        match (&self.memory, &self.disk) {
            (Some(memory), _) => memory.has_block(hash),
            (None, Some(disk)) => disk.has_block(hash),
            (None, None) => false,
        }
    }

    /// Whether an entry with exactly this block chain is stored in either tier.
    pub fn has_entry(&self, key: &[u64]) -> bool {
        self.memory.as_ref().is_some_and(|m| m.has_entry(key))
            || self.disk.as_ref().is_some_and(|d| d.has_entry(key))
    }

    /// Stores an entry: in memory at once, and on disk by the writer thread with every block the disk lacks.
    pub fn write(&self, write: PrefixWrite) {
        let Some(memory) = &self.memory else {
            if let Some(disk) = &self.disk {
                disk.write(write);
            }
            return;
        };
        let chain = memory.insert(&write);
        if let Some(disk) = &self.disk {
            let blocks = chain
                .into_iter()
                .filter(|block| !disk.has_block(block.hash))
                .collect();
            disk.write(PrefixWrite { blocks, ..write });
        }
    }

    /// Marks a stored entry used now, under `ttl` (`None`: the store's default), in both tiers.
    pub fn touch(&self, key: &[u64], ttl: Option<Duration>) {
        if let Some(memory) = &self.memory {
            memory.touch(key, ttl);
        }
        if let Some(disk) = &self.disk {
            disk.touch(key, ttl);
        }
    }

    /// The longest stored entry whose chain starts `hashes` and is longer than `min_blocks`: its block count.
    pub fn longest_entry(&self, hashes: &[u64], min_blocks: usize) -> Option<usize> {
        let memory = self
            .memory
            .as_ref()
            .and_then(|m| m.longest_entry(hashes, min_blocks));
        let disk = self
            .disk
            .as_ref()
            .and_then(|d| d.longest_entry(hashes, min_blocks));
        memory.max(disk)
    }

    /// The entry for the first `n_blocks` of `hashes`, its blocks checked against `tokens`, marked used now under
    /// `ttl`: from memory, else from disk (and then copied into memory); `None` when neither tier holds it whole.
    pub fn load(
        &self,
        hashes: &[u64],
        tokens: &[u32],
        block_size: usize,
        ttl: Option<Duration>,
    ) -> Result<Option<LoadedPrefix>> {
        if let Some(loaded) = self
            .memory
            .as_ref()
            .and_then(|m| m.load(hashes, tokens, block_size, ttl))
        {
            if let Some(disk) = &self.disk {
                disk.touch(hashes, ttl);
            }
            return Ok(Some(loaded));
        }
        let Some(disk) = &self.disk else {
            return Ok(None);
        };
        let loaded = disk.load(hashes, tokens, block_size, ttl)?;
        if let (Some(memory), Some(loaded)) = (&self.memory, &loaded) {
            memory.insert(&PrefixWrite {
                key: hashes.to_vec(),
                blocks: loaded.blocks.clone(),
                recurrent: loaded.recurrent.clone(),
                auxiliary: loaded.auxiliary.clone(),
                ttl,
            });
        }
        Ok(loaded)
    }
}

struct WriterCtx {
    root: PathBuf,
    capacity_bytes: u64,
    default_ttl: Duration,
    index: Arc<Mutex<StoreIndex>>,
}

impl WriterCtx {
    fn write(&self, write: PrefixWrite) -> Result<()> {
        for block in &write.blocks {
            let path = block_path(&self.root, block.hash);
            if path.exists() {
                continue;
            }
            fs::create_dir_all(path.parent().expect("block files sit in a shard")).map_err(io)?;
            let mut tensors = HashMap::new();
            let n = block.tokens.len();
            tensors.insert(
                TOKENS.to_string(),
                Tensor::from_vec(block.tokens.clone(), n, &Device::Cpu)?,
            );
            for (layer, k, v) in &block.layers {
                tensors.insert(format!("k.{layer}"), k.clone());
                tensors.insert(format!("v.{layer}"), v.clone());
            }
            save_atomic(&tensors, &path)?;
            self.index
                .lock()
                .expect("prefix store index poisoned")
                .blocks
                .insert(block.hash, file_bytes(&path));
        }
        let id = entry_id(&write.key);
        let dir = self.root.join(ENTRIES_DIR).join(&id);
        fs::create_dir_all(&dir).map_err(io)?;
        let mut state = HashMap::new();
        for (i, layer) in write.recurrent.iter().enumerate() {
            state.insert(format!("conv.{i}"), layer.conv.clone());
            state.insert(format!("recurrent.{i}"), layer.recurrent.clone());
        }
        if let Some(aux) = &write.auxiliary {
            state.insert(AUX_KEY.to_string(), aux.key.clone());
            state.insert(AUX_VALUE.to_string(), aux.value.clone());
            let positions =
                [aux.origin_pos, aux.valid_start_pos, aux.next_committed_pos].map(|p| p as i64);
            state.insert(
                AUX_POSITIONS.to_string(),
                Tensor::new(&positions, &Device::Cpu)?.to_dtype(DType::I64)?,
            );
        }
        save_atomic(&state, &dir.join(STATE_FILE))?;
        let key_bytes = write
            .key
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect::<Vec<_>>();
        let key_partial = dir.join(format!("{KEY_FILE}{PARTIAL_SUFFIX}"));
        fs::write(&key_partial, key_bytes).map_err(io)?;
        fs::rename(&key_partial, dir.join(KEY_FILE)).map_err(io)?;
        let ttl = write.ttl.unwrap_or(self.default_ttl);
        write_ttl(&dir, ttl);
        let bytes = file_bytes(&dir.join(KEY_FILE)) + file_bytes(&dir.join(STATE_FILE));
        let mut index = self.index.lock().expect("prefix store index poisoned");
        index.entries.insert(
            id,
            StoredEntry {
                key: write.key,
                bytes,
                last_used: SystemTime::now(),
                ttl,
            },
        );
        self.collect(&mut index);
        Ok(())
    }

    fn touch(&self, key: &[u64], ttl: Option<Duration>) {
        let id = entry_id(key);
        let dir = self.root.join(ENTRIES_DIR).join(&id);
        let ttl = ttl.unwrap_or(self.default_ttl);
        let mut index = self.index.lock().expect("prefix store index poisoned");
        let Some(entry) = index.entries.get_mut(&id) else {
            return;
        };
        entry.last_used = SystemTime::now();
        if entry.ttl != ttl {
            entry.ttl = ttl;
            write_ttl(&dir, ttl);
        }
        let _ = fs::File::options()
            .write(true)
            .open(dir.join(KEY_FILE))
            .and_then(|f| f.set_modified(SystemTime::now()));
    }

    // Least recently used entries go until the store fits
    fn collect(&self, index: &mut StoreIndex) {
        if index.bytes() <= self.capacity_bytes {
            return;
        }
        let mut by_age = index
            .entries
            .iter()
            .map(|(id, entry)| (entry.last_used, id.clone()))
            .collect::<Vec<_>>();
        by_age.sort();
        for (_, id) in by_age {
            if index.bytes() <= self.capacity_bytes || index.entries.len() <= 1 {
                break;
            }
            self.remove_entry(index, &id);
        }
    }

    fn expire(&self, index: &mut StoreIndex) {
        let now = SystemTime::now();
        let expired = index
            .entries
            .iter()
            .filter(|(_, entry)| entry.expired(now))
            .map(|(id, _)| id.clone())
            .collect::<Vec<_>>();
        if expired.is_empty() {
            return;
        }
        for id in &expired {
            self.remove_entry(index, id);
        }
        tracing::debug!(entries = expired.len(), "Expired prefix store entries");
    }

    // The entry, then the blocks no entry still names
    fn remove_entry(&self, index: &mut StoreIndex, id: &str) {
        index.entries.remove(id);
        let _ = fs::remove_dir_all(self.root.join(ENTRIES_DIR).join(id));
        let named = index
            .entries
            .values()
            .flat_map(|entry| entry.key.iter().copied())
            .collect::<HashSet<_>>();
        let orphans = index
            .blocks
            .keys()
            .filter(|hash| !named.contains(hash))
            .copied()
            .collect::<Vec<_>>();
        for hash in orphans {
            index.blocks.remove(&hash);
            let _ = fs::remove_file(block_path(&self.root, hash));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = 4;
    const CAPACITY: u64 = 1 << 30;
    const NAMESPACE: &str = "test model / bf16 / q8";
    // the writer thread finishes queued writes well within this
    const WRITE_WAIT: std::time::Duration = std::time::Duration::from_millis(20);
    const WRITE_POLLS: usize = 500;
    const TTL: Duration = Duration::from_secs(3600);
    const SHORT_TTL: Duration = Duration::from_secs(300);

    fn block(hash: u64, first_token: u32) -> Result<StoredBlock> {
        let row =
            |fill: f32| Tensor::full(fill, (1, 2, BLOCK, 8), &Device::Cpu)?.to_dtype(DType::BF16);
        Ok(StoredBlock {
            hash,
            tokens: (first_token..first_token + u32::try_from(BLOCK)?).collect(),
            layers: vec![(3, row(1.)?, row(2.)?), (7, row(3.)?, row(4.)?)],
        })
    }

    fn write(store: &PrefixStore, key: &[u64]) -> Result<()> {
        write_ttl(store, key, None)
    }

    fn write_ttl(store: &PrefixStore, key: &[u64], ttl: Option<Duration>) -> Result<()> {
        let blocks = key
            .iter()
            .enumerate()
            .map(|(i, &hash)| block(hash, u32::try_from(i * BLOCK)?))
            .collect::<Result<Vec<_>>>()?;
        let state = |fill: f32| Tensor::full(fill, (1, 3, 5), &Device::Cpu);
        store.write(PrefixWrite {
            key: key.to_vec(),
            blocks,
            recurrent: vec![RecurrentSnapshotState {
                conv: state(5.)?,
                recurrent: state(6.)?,
            }],
            auxiliary: Some(PagedAuxiliaryPrefixData {
                key: state(7.)?,
                value: state(8.)?,
                origin_pos: 1,
                valid_start_pos: 2,
                next_committed_pos: 3,
            }),
            ttl,
        });
        for _ in 0..WRITE_POLLS {
            if store.has_entry(key) {
                return Ok(());
            }
            std::thread::sleep(WRITE_WAIT);
        }
        inference_tensor::bail!("the writer never stored {key:?}")
    }

    impl PrefixStore {
        fn disk(&self) -> &DiskStore {
            self.disk.as_ref().expect("a disk tier")
        }
    }

    fn tokens(blocks: usize) -> Vec<u32> {
        (0..u32::try_from(blocks * BLOCK).expect("a small test prompt")).collect()
    }

    #[test]
    fn an_entry_reads_back_and_survives_reopening() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let key = [11, 22, 33];
        {
            let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
            write(&store, &key)?;
        }
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        assert_eq!(store.longest_entry(&[11, 22, 33, 44], 0), Some(3));
        assert_eq!(store.longest_entry(&[11, 22, 33, 44], 3), None);
        assert_eq!(store.longest_entry(&[11, 99, 33], 0), None);
        let loaded = store
            .load(&key, &tokens(3), BLOCK, None)?
            .expect("stored entry");
        assert_eq!(loaded.blocks.len(), 3);
        assert_eq!(loaded.blocks[1].tokens, (4..8).collect::<Vec<u32>>());
        let layers = loaded.blocks[1]
            .layers
            .iter()
            .map(|(l, _, _)| *l)
            .collect::<Vec<_>>();
        assert_eq!(layers, [3, 7]);
        assert_eq!(loaded.blocks[1].layers[1].2.dtype(), DType::BF16);
        assert_eq!(loaded.recurrent.len(), 1);
        let aux = loaded.auxiliary.expect("auxiliary state");
        assert_eq!(
            (aux.origin_pos, aux.valid_start_pos, aux.next_committed_pos),
            (1, 2, 3)
        );
        // another setup never sees this one's entries
        let other = PrefixStore::open(Some(dir.path()), "another model", CAPACITY, 0, TTL)?;
        assert_eq!(other.longest_entry(&key, 0), None);
        Ok(())
    }

    #[test]
    fn a_store_under_another_layout_is_removed_on_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let current = {
            let store = PrefixStore::open(Some(dir.path()), "other model", CAPACITY, 0, TTL)?;
            write(&store, &[1, 2])?;
            store.disk().root.clone()
        };
        let older = dir.path().join("00000000000000aa");
        fs::create_dir_all(older.join(ENTRIES_DIR)).map_err(io)?;
        fs::write(older.join(NAMESPACE_FILE), NAMESPACE).map_err(io)?;
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        assert!(!older.exists());
        assert!(current.join(NAMESPACE_FILE).exists());
        assert_eq!(store.longest_entry(&[1, 2], 0), None);
        Ok(())
    }

    #[test]
    fn a_block_with_other_tokens_drops_the_entry() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        write(&store, &[1, 2])?;
        let mut other = tokens(2);
        other[5] += 1;
        assert!(store.load(&[1, 2], &other, BLOCK, None)?.is_none());
        assert_eq!(store.longest_entry(&[1, 2], 0), None);
        Ok(())
    }

    #[test]
    fn a_full_store_drops_the_oldest_entry_and_its_unshared_blocks() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let probe = PrefixStore::open(Some(dir.path()), "probe", CAPACITY, 0, TTL)?;
        write(&probe, &[1, 2])?;
        let entry_bytes = probe.disk().index.lock().unwrap().bytes();
        // room for two two-block entries that share their first block, not a third
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, entry_bytes * 2, 0, TTL)?;
        write(&store, &[1, 2])?;
        write(&store, &[1, 3])?;
        write(&store, &[1, 4])?;
        assert_eq!(store.longest_entry(&[1, 2], 0), None);
        assert_eq!(store.longest_entry(&[1, 3], 0), Some(2));
        assert_eq!(store.longest_entry(&[1, 4], 0), Some(2));
        assert!(store.has_block(1) && !store.has_block(2));
        Ok(())
    }

    #[test]
    fn a_half_written_entry_is_removed_on_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = {
            let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
            write(&store, &[5, 6])?;
            store.disk().root.clone()
        };
        let entry = root.join(ENTRIES_DIR).join(entry_id(&[5, 6]));
        fs::remove_file(entry.join(KEY_FILE)).map_err(io)?;
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        assert_eq!(store.longest_entry(&[5, 6], 0), None);
        assert!(!entry.exists());
        Ok(())
    }

    fn wait_for(mut done: impl FnMut() -> bool) -> Result<()> {
        for _ in 0..WRITE_POLLS {
            if done() {
                return Ok(());
            }
            std::thread::sleep(WRITE_WAIT);
        }
        inference_tensor::bail!("the writer never got there")
    }

    #[test]
    fn an_entry_past_its_ttl_is_swept_with_its_unshared_blocks() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        write(&store, &[1, 2])?;
        write(&store, &[1, 3])?;
        // a hit under a zero TTL leaves the entry expired at once; the writer sweeps after every operation
        store.touch(&[1, 2], Some(Duration::ZERO));
        wait_for(|| !store.has_entry(&[1, 2]))?;
        assert!(store.has_entry(&[1, 3]));
        assert!(store.has_block(1) && !store.has_block(2) && store.has_block(3));
        Ok(())
    }

    #[test]
    fn an_entry_keeps_its_ttl_across_reopening() -> Result<()> {
        let dir = tempfile::tempdir()?;
        {
            let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
            write_ttl(&store, &[7, 8], Some(SHORT_TTL))?;
        }
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        let ttl = store.disk().index.lock().unwrap().entries[&entry_id(&[7, 8])].ttl;
        assert_eq!(ttl, SHORT_TTL);
        Ok(())
    }

    #[test]
    fn an_entry_idle_past_its_ttl_is_removed_on_open() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let root = {
            let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
            write_ttl(&store, &[5, 6], Some(SHORT_TTL))?;
            store.disk().root.clone()
        };
        let entry = root.join(ENTRIES_DIR).join(entry_id(&[5, 6]));
        let idle = SystemTime::now() - 2 * SHORT_TTL;
        fs::File::options()
            .write(true)
            .open(entry.join(KEY_FILE))
            .and_then(|f| f.set_modified(idle))
            .map_err(io)?;
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        assert_eq!(store.longest_entry(&[5, 6], 0), None);
        assert!(!entry.exists() && !store.has_block(5));
        Ok(())
    }

    #[test]
    fn an_entry_in_memory_reads_back_without_a_directory() -> Result<()> {
        let store = PrefixStore::open(None, NAMESPACE, 0, CAPACITY, TTL)?;
        write(&store, &[11, 22])?;
        assert!(store.has_block(11) && store.has_block(22));
        assert_eq!(store.longest_entry(&[11, 22, 33], 0), Some(2));
        let loaded = store
            .load(&[11, 22], &tokens(2), BLOCK, None)?
            .expect("held entry");
        assert_eq!(loaded.blocks[1].tokens, (4..8).collect::<Vec<u32>>());
        assert!(loaded.auxiliary.is_some());
        let mut other = tokens(2);
        other[5] += 1;
        assert!(store.load(&[11, 22], &other, BLOCK, None)?.is_none());
        assert_eq!(store.longest_entry(&[11, 22], 0), None);
        Ok(())
    }

    #[test]
    fn a_full_memory_store_drops_the_oldest_entry_and_its_unshared_blocks() -> Result<()> {
        let probe = PrefixStore::open(None, NAMESPACE, 0, CAPACITY, TTL)?;
        write(&probe, &[1, 2])?;
        let entry_bytes = probe
            .memory
            .as_ref()
            .expect("a memory tier")
            .index()
            .bytes();
        let store = PrefixStore::open(None, NAMESPACE, 0, entry_bytes * 2, TTL)?;
        write(&store, &[1, 2])?;
        write(&store, &[1, 3])?;
        write(&store, &[1, 4])?;
        assert_eq!(store.longest_entry(&[1, 2], 0), None);
        assert_eq!(store.longest_entry(&[1, 3], 0), Some(2));
        assert!(store.has_block(1) && !store.has_block(2));
        Ok(())
    }

    // a write carries only the blocks memory lacks, as the engine sends them; the disk still gets the whole chain
    #[test]
    fn with_both_tiers_writes_reach_disk_whole_and_a_disk_hit_returns_to_memory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let three = || -> Result<Vec<u32>> {
            let mut chain = tokens(1);
            chain.extend(
                (BLOCK..2 * BLOCK)
                    .map(u32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            );
            Ok(chain)
        };
        {
            let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, CAPACITY, TTL)?;
            let memory = store.memory.as_ref().expect("a memory tier");
            memory.insert(&PrefixWrite {
                key: vec![1, 2],
                blocks: vec![block(1, 0)?, block(2, u32::try_from(BLOCK)?)?],
                recurrent: Vec::new(),
                auxiliary: None,
                ttl: None,
            });
            assert!(store.has_block(1) && !store.disk().has_block(1));
            store.write(PrefixWrite {
                key: vec![1, 3],
                blocks: vec![block(3, u32::try_from(BLOCK)?)?],
                recurrent: Vec::new(),
                auxiliary: None,
                ttl: None,
            });
            wait_for(|| store.disk().has_entry(&[1, 3]))?;
        }
        let on_disk = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, 0, TTL)?;
        assert!(on_disk.load(&[1, 3], &three()?, BLOCK, None)?.is_some());
        drop(on_disk);
        let store = PrefixStore::open(Some(dir.path()), NAMESPACE, CAPACITY, CAPACITY, TTL)?;
        let memory = store.memory.as_ref().expect("a memory tier");
        assert!(!memory.has_entry(&[1, 3]));
        assert!(store.load(&[1, 3], &three()?, BLOCK, None)?.is_some());
        assert!(memory.has_entry(&[1, 3]));
        Ok(())
    }
}
