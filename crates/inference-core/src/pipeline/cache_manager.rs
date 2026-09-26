use candle_core::{Result, Tensor};

use crate::{
    kv_cache::*,
    pipeline::{CacheManagerMixin, MetadataMixin},
    sequence::Sequence,
};

pub trait CacheManager<T: CacheManagerMixin + MetadataMixin + ?Sized> {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()>;
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> Result<()>;
}

pub struct NormalCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for NormalCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        let mut new_k_cache = Vec::new();
        let mut new_v_cache = Vec::new();

        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            // Preallocate combined k and v caches across all sequences, avoiding Tensor::cat copies
            let batch_len = seqs.len();
            // Use the first sequence as template
            let (first_k, first_v) = {
                let src_cache = if modify_draft_cache {
                    seqs[0].normal_draft_cache()
                } else {
                    seqs[0].normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    new_k_cache.push(None);
                    new_v_cache.push(None);
                    continue;
                };
                match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Shared { .. } => {
                        new_k_cache.push(None);
                        new_v_cache.push(None);
                        continue;
                    }
                }
            };
            // Build dims for batched cache
            let mut dims_k = first_k.dims().to_vec();
            let mut dims_v = first_v.dims().to_vec();
            dims_k[0] *= batch_len;
            dims_v[0] *= batch_len;
            let batch_k = Tensor::zeros(dims_k.clone(), first_k.dtype(), first_k.device()).unwrap();
            let batch_v = Tensor::zeros(dims_v.clone(), first_v.dtype(), first_v.device()).unwrap();
            // Fill each sequence's cache slice
            for (i, seq) in seqs.iter_mut().enumerate() {
                let src_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let Some(cache) = src_cache.get(layer).unwrap().as_ref() else {
                    continue;
                };
                let (src_k, src_v) = match cache {
                    KvCache::Normal { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Rotating { k, v } => {
                        (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                    }
                    KvCache::Shared { .. } => continue,
                };
                let offset = i * first_k.dims()[0];
                batch_k.slice_set(&src_k, 0, offset).unwrap();
                batch_v.slice_set(&src_v, 0, offset).unwrap();
            }
            new_k_cache.push(Some(batch_k));
            new_v_cache.push(Some(batch_v));
        }

        let seq0_cache = if modify_draft_cache {
            &*seqs[0].normal_draft_cache()
        } else {
            &*seqs[0].normal_cache()
        };

        let mut caches = Vec::new();
        for (layer_idx, (k_cache, v_cache)) in new_k_cache.into_iter().zip(new_v_cache).enumerate()
        {
            // Use this for the various parameters. Assumes all seqs are from one model.
            let Some(cache_ref) = seq0_cache[layer_idx].as_ref() else {
                let mut cache = pipeline.cache().normal().0[layer_idx].clone();
                cache.reset();
                caches.push(cache);
                continue;
            };
            match cache_ref {
                KvCache::Normal { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_capsl = old_k.capacity_seq_len;

                    caches.push(KvCache::Normal {
                        k: SingleCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                        v: SingleCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                        },
                    });
                }
                KvCache::Rotating { k: old_k, .. } => {
                    let template_cache_dim = old_k.dim;
                    let template_cache_csl = old_k.current_seq_len;
                    let template_cache_msl = old_k.max_seq_len;
                    let template_cache_capsl = old_k.capacity_seq_len;
                    let template_cache_wpos = old_k.write_pos;

                    caches.push(KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                            write_pos: template_cache_wpos,
                            last_append_result: None,
                        },
                        v: RotatingCache {
                            all_data: v_cache.map(|x| x.contiguous().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: template_cache_csl,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: template_cache_capsl,
                            write_pos: template_cache_wpos,
                            last_append_result: None,
                        },
                    });
                }
                KvCache::Shared { owner } => {
                    caches.push(KvCache::Shared { owner: *owner });
                }
            }
        }
        *pipeline.cache().normal() = NormalCache(caches);
        Ok(())
    }
    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let all_cache = pipeline.cache().normal();
        for layer in 0..pipeline.get_metadata().num_hidden_layers {
            let cache = all_cache.0.get(layer).unwrap();
            if let KvCache::Shared { owner } = cache {
                for seq in seqs.iter_mut() {
                    let output_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    output_cache[layer] = Some(KvCache::Shared { owner: *owner });
                }
                continue;
            }
            // This case for llama 3.2 vision cross attn
            if cache.k().unwrap().is_none() {
                continue;
            }

            let (k_cache, v_cache) = match cache {
                KvCache::Normal { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::Rotating { k, v } => {
                    (k.all_data.clone().unwrap(), v.all_data.clone().unwrap())
                }
                KvCache::Shared { .. } => unreachable!(),
            };

            let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(k_caches.len(), seqs.len());
            let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
            debug_assert_eq!(v_caches.len(), seqs.len());

            for (seq_i, seq) in seqs.iter_mut().enumerate() {
                let output_cache = if modify_draft_cache {
                    seq.normal_draft_cache()
                } else {
                    seq.normal_cache()
                };
                let seq_cache = &mut output_cache[layer];
                let k = k_caches.get(seq_i).unwrap().clone();
                let v = v_caches.get(seq_i).unwrap().clone();

                match cache {
                    KvCache::Normal {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Normal {
                            k: SingleCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                capacity_seq_len: cache_k.capacity_seq_len,
                            },
                            v: SingleCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                            },
                        });
                    }
                    KvCache::Rotating {
                        k: cache_k,
                        v: cache_v,
                    } => {
                        *seq_cache = Some(KvCache::Rotating {
                            k: RotatingCache {
                                all_data: Some(k),
                                dim: cache_k.dim,
                                current_seq_len: cache_k.current_seq_len,
                                max_seq_len: cache_k.max_seq_len,
                                capacity_seq_len: cache_k.capacity_seq_len,
                                write_pos: cache_k.write_pos,
                                last_append_result: None,
                            },
                            v: RotatingCache {
                                all_data: Some(v),
                                dim: cache_v.dim,
                                current_seq_len: cache_v.current_seq_len,
                                max_seq_len: cache_v.max_seq_len,
                                capacity_seq_len: cache_v.capacity_seq_len,
                                write_pos: cache_v.write_pos,
                                last_append_result: None,
                            },
                        });
                    }
                    KvCache::Shared { .. } => unreachable!(),
                }
            }
        }
    }
    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        _modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> Result<()> {
        if seqs.iter().any(|seq| seq.preallocated_cache().is_none()) {
            for layer in pipeline.cache().normal().0.iter_mut() {
                layer.reset();
            }
            return Ok(());
        }

        let layer_devices = pipeline.device_mapper().map(|device_mapper| {
            let total_layers = pipeline.cache().normal().0.len();
            let mut layer_devices = Vec::with_capacity(total_layers);
            for layer in 0..total_layers {
                let device = device_mapper
                    .device_for(layer, false)
                    .cloned()
                    .expect("Internal bug, layer out of range!");
                layer_devices.push(device);
            }
            layer_devices
        });

        let old_caches = pipeline.cache().normal().0.clone();

        for (layer_idx, layer) in pipeline.cache().normal().0.iter_mut().enumerate() {
            if !load_preallocated_cache {
                layer.reset();
                continue;
            }

            match &old_caches[layer_idx] {
                KvCache::Rotating { k, .. } => {
                    *layer = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: None,
                            dim: k.dim,
                            current_seq_len: 0,
                            max_seq_len: k.max_seq_len,
                            capacity_seq_len: k.capacity_seq_len,
                            write_pos: 0,
                            last_append_result: None,
                        },
                        v: RotatingCache {
                            all_data: None,
                            dim: k.dim,
                            current_seq_len: 0,
                            max_seq_len: k.max_seq_len,
                            capacity_seq_len: k.capacity_seq_len,
                            write_pos: 0,
                            last_append_result: None,
                        },
                    };
                    continue;
                }
                KvCache::Shared { owner } => {
                    *layer = KvCache::Shared { owner: *owner };
                    continue;
                }
                KvCache::Normal { .. } => {}
            }

            let mut k_caches = Vec::new();
            let mut v_caches = Vec::new();
            let mut missing_preallocated = false;
            for seq in seqs.iter_mut() {
                let Some((mut k_preallocated_cache, mut v_preallocated_cache)) = seq
                    .preallocated_cache()
                    .and_then(|cache| cache.get(layer_idx))
                    .cloned()
                    .flatten()
                else {
                    missing_preallocated = true;
                    break;
                };
                if let Some(layer_devices) = &layer_devices {
                    let layer_dev = &layer_devices[layer_idx];
                    k_preallocated_cache = k_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                    v_preallocated_cache = v_preallocated_cache
                        .to_device(layer_dev)
                        .expect("Could not prepare cache");
                }
                k_caches.push(k_preallocated_cache);
                v_caches.push(v_preallocated_cache);
            }
            if missing_preallocated {
                layer.reset();
                continue;
            }
            let k_cache = if k_caches.len() > 1 {
                Tensor::cat(&k_caches, 0).unwrap()
            } else {
                k_caches[0].clone()
            };
            let v_cache = if v_caches.len() > 1 {
                Tensor::cat(&v_caches, 0).unwrap()
            } else {
                v_caches[0].clone()
            };

            // Use this for the various parameters. Assumes all seqs are from one model.
            match &old_caches[layer_idx] {
                KvCache::Normal { k, .. } => {
                    let template_cache_dim = k.dim;
                    let template_cache_msl = k.max_seq_len;

                    let cache = KvCache::Normal {
                        k: SingleCache {
                            all_data: Some(k_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                        v: SingleCache {
                            all_data: Some(v_cache.zeros_like().unwrap()),
                            dim: template_cache_dim,
                            current_seq_len: 0,
                            max_seq_len: template_cache_msl,
                            capacity_seq_len: k_cache.dims()[template_cache_dim],
                        },
                    };
                    *layer = cache;
                }
                KvCache::Rotating { .. } | KvCache::Shared { .. } => unreachable!(),
            }
        }
        Ok(())
    }
}

pub struct FullCacheManager;

enum SeqCache {
    Normal,
    XLora,
    Draft,
}

fn clone_in_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    src: SeqCache,
) {
    let mut new_cache = Vec::new();
    'outer: for layer in 0..num_hidden_layers {
        let mut k_vec = Vec::new();
        let mut v_vec = Vec::new();
        for seq in &mut *seqs {
            let src_cache = match src {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let cache = src_cache.get(layer).unwrap();
            // This case for llama 3.2 vision cross attn
            if cache.is_none() {
                new_cache.push(None);
                continue 'outer;
            }
            let cache = cache
                .as_ref()
                .expect("Not handling completions in `clone_in_cache`.");
            k_vec.push(cache.0.clone());
            v_vec.push(cache.1.clone());
        }
        new_cache.push(Some((
            if k_vec.len() > 1 {
                Tensor::cat(&k_vec, 0).unwrap()
            } else {
                k_vec[0].clone()
            },
            if v_vec.len() > 1 {
                Tensor::cat(&v_vec, 0).unwrap()
            } else {
                v_vec[0].clone()
            },
        )));
    }
    *cache = new_cache;
}

fn clone_out_cache(
    num_hidden_layers: usize,
    cache: &mut LayerCaches,
    seqs: &mut [&mut crate::sequence::Sequence],
    target: SeqCache,
) {
    for layer in 0..num_hidden_layers {
        let cache = cache.get(layer).unwrap();
        // This case for llama 3.2 vision cross attn
        if cache.is_none() {
            continue;
        }

        let k_cache = cache.as_ref().unwrap().0.clone();
        let v_cache = cache.as_ref().unwrap().1.clone();

        let k_caches = k_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(k_caches.len(), seqs.len());
        let v_caches = v_cache.chunk(seqs.len(), 0).unwrap();
        debug_assert_eq!(v_caches.len(), seqs.len());

        for (seq_i, seq) in seqs.iter_mut().enumerate() {
            let output_cache = match target {
                SeqCache::Normal => seq.cache(),
                SeqCache::XLora => seq.xlora_cache(),
                SeqCache::Draft => seq.draft_cache(),
            };
            let seq_cache = &mut output_cache[layer];
            let k = k_caches.get(seq_i).unwrap().clone();
            let v = v_caches.get(seq_i).unwrap().clone();
            *seq_cache = Some((k, v));
        }
    }
}

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for FullCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        if modify_draft_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return Ok(());
        }
        clone_in_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_in_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            pipeline
                .cache()
                .full()
                .get_scalings_cache()
                .clone_from(seqs[0].scaling_cache());
        }
        Ok(())
    }

    fn clone_out_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) {
        if modify_draft_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().lock(),
                seqs,
                SeqCache::Draft,
            );
            return;
        }
        clone_out_cache(
            pipeline.get_metadata().num_hidden_layers,
            &mut pipeline.cache().full().lock(),
            seqs,
            SeqCache::Normal,
        );
        if pipeline.get_metadata().is_xlora && !pipeline.get_metadata().no_kv_cache {
            clone_out_cache(
                pipeline.get_metadata().num_hidden_layers,
                &mut pipeline.cache().full().xlora_lock(),
                seqs,
                SeqCache::XLora,
            );
        }
        if pipeline.get_metadata().is_xlora {
            seqs[0]
                .scaling_cache()
                .clone_from(&pipeline.cache().full().get_scalings_cache());
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        _seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> Result<()> {
        let mut new_cache = Vec::new();
        for _ in 0..pipeline.get_metadata().num_hidden_layers {
            new_cache.push(None);
        }
        pipeline.cache().full().lock().clone_from(&new_cache);
        if modify_draft_cache {
            pipeline.cache().full().draft_lock().clone_from(&new_cache);
        }
        if pipeline.cache().full().is_xlora() {
            *pipeline.cache().full().xlora_lock() = new_cache;
        }
        Ok(())
    }
}

/// Cache manager for hybrid models (attention + recurrent layers).
///
/// This implements vLLM-style continuous batching:
/// - Attention layers: Standard KV cache batching (cat on clone_in, chunk on clone_out)
/// - Recurrent layers: Pool-based state management with indexed access
///
/// Each sequence has a `recurrent_state_idx` pointing to its slot in the
/// state pool. The forward pass builds a `state_indices` tensor from these
/// indices and uses gather/scatter operations.
pub struct HybridCacheManager;

impl<T: CacheManagerMixin + MetadataMixin + ?Sized> CacheManager<T> for HybridCacheManager {
    fn clone_in_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut crate::sequence::Sequence],
        modify_draft_cache: bool,
    ) -> Result<()> {
        let mut hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();
        let sequence_slots = seqs
            .iter()
            .map(|seq| {
                seq.recurrent_state_idx()
                    .map(|slot_idx| (*seq.id(), slot_idx))
                    .ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "hybrid sequence {} has no recurrent state slot",
                            seq.id()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        hybrid_cache.install_sequence_state_indices(&sequence_slots)?;

        // For attention layers, we still need to batch KV caches
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get_mut(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                // Batch KV caches from sequences (same as NormalCacheManager)
                let mut k_tensors = Vec::new();
                let mut v_tensors = Vec::new();
                let mut template_cache: Option<KvCache> = None;

                for seq in seqs.iter_mut() {
                    let seq_cache = if modify_draft_cache {
                        seq.normal_draft_cache()
                    } else {
                        seq.normal_cache()
                    };
                    if let Some(Some(ref kv)) = seq_cache.get(layer_idx) {
                        if template_cache.is_none() {
                            template_cache = Some(kv.clone());
                        }
                        if let (Ok(Some(k)), Ok(Some(v))) = (kv.k(), kv.v()) {
                            k_tensors.push(k);
                            v_tensors.push(v);
                        }
                    }
                }

                if !k_tensors.is_empty() {
                    // cat/clone of narrow'd views may be non-contiguous;
                    // all_data must be contiguous for slice_set in SingleCache::append.
                    let batched_k = if k_tensors.len() > 1 {
                        Tensor::cat(&k_tensors, 0)?
                    } else {
                        k_tensors[0].contiguous()?
                    };
                    let batched_v = if v_tensors.len() > 1 {
                        Tensor::cat(&v_tensors, 0)?
                    } else {
                        v_tensors[0].contiguous()?
                    };

                    if let Some(ref template) = template_cache {
                        match (template, kv_cache) {
                            (KvCache::Normal { k: tk, .. }, KvCache::Normal { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            (KvCache::Rotating { k: tk, .. }, KvCache::Rotating { k, v }) => {
                                k.all_data = Some(batched_k);
                                k.current_seq_len = tk.current_seq_len;
                                k.capacity_seq_len = tk.current_seq_len;
                                v.all_data = Some(batched_v);
                                v.current_seq_len = tk.current_seq_len;
                                v.capacity_seq_len = tk.current_seq_len;
                            }
                            _ => {}
                        }
                    }
                }
            }
            // For recurrent layers: No copying needed!
            // The pool is accessed directly via state_indices during forward.
        }
        Ok(())
    }

    fn clone_out_cache(&self, pipeline: &T, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        let hybrid_cache = pipeline.cache().hybrid();
        let num_layers = hybrid_cache.num_layers();
        let num_seqs = seqs.len();

        // For attention layers, split batched KV caches back to sequences
        for layer_idx in 0..num_layers {
            let layer_cache = hybrid_cache.caches.get(layer_idx).unwrap();

            if let HybridLayerCache::Attention(kv_cache) = layer_cache {
                if let (Ok(Some(k)), Ok(Some(v))) = (kv_cache.k(), kv_cache.v()) {
                    let k_chunks = k.chunk(num_seqs, 0).unwrap();
                    let v_chunks = v.chunk(num_seqs, 0).unwrap();

                    for (seq_idx, seq) in seqs.iter_mut().enumerate() {
                        let seq_k = k_chunks.get(seq_idx).unwrap().clone();
                        let seq_v = v_chunks.get(seq_idx).unwrap().clone();

                        let seq_cache = if modify_draft_cache {
                            seq.normal_draft_cache()
                        } else {
                            seq.normal_cache()
                        };

                        // Initialize cache if needed
                        if seq_cache.get(layer_idx).is_none() || seq_cache[layer_idx].is_none() {
                            while seq_cache.len() <= layer_idx {
                                seq_cache.push(None);
                            }
                            seq_cache[layer_idx] = Some(kv_cache.clone());
                        }

                        if let Some(ref mut seq_kv) = seq_cache[layer_idx] {
                            match (kv_cache, seq_kv) {
                                (KvCache::Normal { k: src_k, .. }, KvCache::Normal { k, v }) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                (
                                    KvCache::Rotating { k: src_k, .. },
                                    KvCache::Rotating { k, v },
                                ) => {
                                    k.all_data = Some(seq_k);
                                    k.current_seq_len = src_k.current_seq_len;
                                    k.capacity_seq_len = src_k.current_seq_len;
                                    v.all_data = Some(seq_v);
                                    v.current_seq_len = src_k.current_seq_len;
                                    v.capacity_seq_len = src_k.current_seq_len;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            // For recurrent layers: No splitting needed!
            // The pool was updated in-place during forward via scatter operations.
        }
    }

    fn set_none_cache(
        &self,
        pipeline: &T,
        seqs: &mut [&mut Sequence],
        modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) -> Result<()> {
        let sequence_slots = seqs
            .iter()
            .map(|seq| {
                seq.recurrent_state_idx()
                    .map(|slot_idx| (*seq.id(), slot_idx))
                    .ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "hybrid sequence {} has no recurrent state slot",
                            seq.id()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut hybrid_cache = pipeline.cache().hybrid();
        hybrid_cache.validate_sequence_slots(&sequence_slots)?;
        for seq in seqs.iter_mut() {
            let seq_cache = if modify_draft_cache {
                seq.normal_draft_cache()
            } else {
                seq.normal_cache()
            };
            for kv in seq_cache.iter_mut().flatten() {
                kv.reset();
            }
        }
        let slots = sequence_slots
            .iter()
            .map(|&(_, slot_idx)| slot_idx)
            .collect::<Vec<_>>();
        hybrid_cache.reset_attention_and_slots(&slots)?;
        hybrid_cache.install_sequence_state_indices(&sequence_slots)?;
        Ok(())
    }
}
