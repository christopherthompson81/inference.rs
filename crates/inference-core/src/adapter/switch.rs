//! Adapters a checkpoint carries in its own weights and switches between per token (IBM's granite-switch): each
//! adapter has a control token, and every token runs the adapter of the control tokens before it.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use inference_quant::{
    LoraAdapterWeights, LoraExecution, LoraExecutionArena, LoraLayerRegistry, LoraSiteHandle,
    LoraWeights, ShardedVarBuilder,
};
use inference_tensor::{Result, Tensor};

// llama.cpp's default router gain when the GGUF leaves it out
pub const DEFAULT_ROUTER_GAIN: f64 = 15.0;
// the embedded factors add `B A x` unscaled
const DELTA_SCALE: f64 = 1.0;
const BLOCK_PREFIX: &str = "model.layers.";
// our projections against the GGUF's factor stems; gate and up share Granite's fused `input_linear`
const ATTENTION_SITES: [(&str, &str); 4] = [
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
];
const FUSED_GATE_UP_SITE: &str = "shared_mlp.input_linear";
const GATE_UP_STEMS: [&str; 2] = ["ffn_gate", "ffn_up"];
const DOWN_SITE: &str = "shared_mlp.output_linear";
const DOWN_STEM: &str = "ffn_down";

#[derive(Clone, Copy, Debug)]
struct ControlToken {
    slot: u32,
    substitute: u32,
}

/// Which adapter each token runs, from the control tokens before it, as llama.cpp's router layer finds it.
#[derive(Debug)]
pub struct SwitchRouting {
    tokens: HashMap<u32, ControlToken>,
    adapters: u32,
    // a non-control token's weight against a control token's under the router's softmax, e^(-2 gain)
    leak: f64,
}

impl SwitchRouting {
    /// Adapter `i`'s control token `activate[i]` runs slot `i + 1` and reads as `substitute[i]`'s embedding.
    pub fn new(activate: &[u32], substitute: &[u32], gain: f64) -> anyhow::Result<Self> {
        if activate.len() != substitute.len() {
            anyhow::bail!(
                "{} control tokens but {} substitutes",
                activate.len(),
                substitute.len()
            );
        }
        let tokens = activate
            .iter()
            .zip(substitute)
            .zip(1u32..)
            .map(|((&token, &substitute), slot)| (token, ControlToken { slot, substitute }))
            .collect();
        Ok(Self {
            tokens,
            adapters: u32::try_from(activate.len())?,
            leak: (-2.0 * gain).exp(),
        })
    }

    pub fn adapters(&self) -> u32 {
        self.adapters
    }

    /// The token the model embeds for `token`: a control token reads as its substitute.
    pub fn substitute(&self, token: u32) -> u32 {
        self.tokens
            .get(&token)
            .map_or(token, |control| control.substitute)
    }

    pub fn is_control(&self, token: u32) -> bool {
        self.tokens.contains_key(&token)
    }

    /// Positions `start..end` of `tokens`, each with its slot: the router's softmax average of the slots of the
    /// control tokens up to it (other tokens weigh e^(-2 gain) as much and carry slot 0), rounded; 0 with none.
    pub fn slots(&self, tokens: &[u32], start: usize, end: usize) -> Vec<u32> {
        let (mut sum, mut controls) = (0f64, 0f64);
        let mut slots = Vec::with_capacity(end.saturating_sub(start));
        for (position, token) in tokens.iter().take(end).enumerate() {
            if let Some(control) = self.tokens.get(token) {
                sum += f64::from(control.slot);
                controls += 1.0;
            }
            if position < start {
                continue;
            }
            let others = (position + 1) as f64 - controls;
            // in f32 as the router's softmax runs: the leak then vanishes beside a control token, so a tie rounds up
            let slot = match controls > 0.0 {
                true => (sum as f32 / (controls + others * self.leak) as f32).round(),
                false => 0.0,
            };
            slots.push(slot.clamp(0.0, self.adapters as f32) as u32);
        }
        slots
    }
}

/// The checkpoint's adapters as runtime LoRA weights over the model's sites, one per slot after the base.
pub struct SwitchAdapters {
    routing: Arc<SwitchRouting>,
    registry: Arc<LoraLayerRegistry>,
    arena: Arc<LoraExecutionArena>,
    adapters: Vec<Arc<LoraAdapterWeights>>,
}

impl std::fmt::Debug for SwitchAdapters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SwitchAdapters")
            .field("adapters", &self.adapters.len())
            .finish_non_exhaustive()
    }
}

// one slot's factors from the stacked `{slots, rank, in}` A and `{slots, out, rank}` B
fn slot_factors(
    vb: &ShardedVarBuilder,
    stem: &str,
    slot: usize,
    site: &LoraSiteHandle,
) -> Result<(Tensor, Tensor)> {
    let get = |suffix: &str| -> Result<Tensor> {
        vb.get_unchecked_dtype(&format!("{stem}.{suffix}"), site.activation_dtype())?
            .narrow(0, slot, 1)?
            .squeeze(0)?
            .to_device(site.device())
    };
    Ok((get("lora_a")?, get("lora_b")?))
}

// the GGUF factor stems a site takes, `blk.N.<stem>`, or none for a site the adapters leave alone
fn site_stems(path: &str) -> Option<(usize, Vec<&'static str>)> {
    let rest = path.strip_prefix(BLOCK_PREFIX)?;
    let (layer, site) = rest.split_once('.')?;
    let layer = layer.parse().ok()?;
    let stems = match site {
        FUSED_GATE_UP_SITE => GATE_UP_STEMS.to_vec(),
        DOWN_SITE => vec![DOWN_STEM],
        _ => vec![ATTENTION_SITES.iter().find(|(ours, _)| *ours == site)?.1],
    };
    Some((layer, stems))
}

impl SwitchAdapters {
    /// Reads each slot's factors for every site `registry` holds from the GGUF's `blk.N.*.lora_a` / `lora_b`.
    pub fn load(
        routing: Arc<SwitchRouting>,
        registry: Arc<LoraLayerRegistry>,
        vb: &ShardedVarBuilder,
    ) -> Result<Self> {
        let sites = registry.finalize()?;
        let mut adapters = Vec::with_capacity(routing.adapters() as usize);
        for slot in 1..=routing.adapters() as usize {
            let mut weights = Vec::with_capacity(sites.len());
            for site in &sites {
                let Some((layer, stems)) = site_stems(site.key().path()) else {
                    continue;
                };
                let block = vb.pp(format!("blk.{layer}"));
                let factors = stems
                    .iter()
                    .map(|stem| slot_factors(&block, stem, slot, site))
                    .collect::<Result<Vec<_>>>()?;
                let (a, b) = match factors.as_slice() {
                    [(a, b)] => (a.clone(), b.clone()),
                    // fused outputs: A stacks the ranks, B is block diagonal so each output reads its own
                    [(a_gate, b_gate), (a_up, b_up)] => {
                        let (gate_out, rank) = b_gate.dims2()?;
                        let (up_out, _) = b_up.dims2()?;
                        let zeros =
                            |rows| Tensor::zeros((rows, rank), b_gate.dtype(), b_gate.device());
                        let b = Tensor::cat(
                            &[
                                Tensor::cat(&[b_gate, &zeros(gate_out)?], 1)?,
                                Tensor::cat(&[&zeros(up_out)?, b_up], 1)?,
                            ],
                            0,
                        )?;
                        (Tensor::cat(&[a_gate, a_up], 0)?, b)
                    }
                    _ => unreachable!("a site takes one or two factor stems"),
                };
                weights.push((site.clone(), LoraWeights::new(a, b, DELTA_SCALE)?));
            }
            adapters.push(Arc::new(LoraAdapterWeights::new(
                registry.runtime_id(),
                weights,
            )?));
        }
        Ok(Self {
            routing,
            registry,
            arena: Arc::new(LoraExecutionArena::new()),
            adapters,
        })
    }

    pub fn routing(&self) -> &Arc<SwitchRouting> {
        &self.routing
    }

    /// The execution for one step whose flattened token rows run `row_slots` (0 the base model); `None` when every
    /// row runs the base.
    pub fn execution(&self, row_slots: &[u32]) -> Result<Option<Arc<LoraExecution>>> {
        let slots = row_slots
            .iter()
            .copied()
            .filter(|&slot| slot > 0)
            .collect::<BTreeSet<_>>();
        if slots.is_empty() {
            return Ok(None);
        }
        let mut execution = LoraExecution::new_with_arena(
            self.registry.runtime_id(),
            row_slots
                .iter()
                .map(|&slot| (slot > 0).then_some(slot))
                .collect(),
            self.arena.clone(),
        );
        for slot in slots {
            execution.insert_adapter(slot, self.adapters[slot as usize - 1].clone())?;
        }
        Ok(Some(Arc::new(execution)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONTROL_A: u32 = 900;
    const CONTROL_B: u32 = 901;

    fn routing() -> SwitchRouting {
        SwitchRouting::new(&[CONTROL_A, CONTROL_B], &[7, 8], DEFAULT_ROUTER_GAIN).unwrap()
    }

    #[test]
    fn a_token_runs_the_adapter_of_the_control_token_before_it() {
        let routing = routing();
        let tokens = [1, 2, CONTROL_B, 3, 4];
        assert_eq!(routing.slots(&tokens, 0, 5), vec![0, 0, 2, 2, 2]);
        // a chunk reads the control tokens of the chunks before it
        assert_eq!(routing.slots(&tokens, 3, 5), vec![2, 2]);
        assert_eq!(routing.substitute(CONTROL_B), 8);
        assert_eq!(routing.substitute(3), 3);
    }

    // two visible control tokens share the router's weight: slots 1 and 2 average to 1.5, which rounds up
    #[test]
    fn several_control_tokens_round_their_mean_slot() {
        let routing = routing();
        assert_eq!(
            routing.slots(&[CONTROL_A, 5, CONTROL_B, 6], 0, 4),
            vec![1, 1, 2, 2]
        );
    }
}
