//! K-EXAONE (`ExaoneMoeForCausalLM`): EXAONE 4's attention (q/k norms, sliding layers beside NoPE global ones) in
//! pre-norm layers whose feed-forward is DeepSeek V3's MoE (sigmoid routing with a selection bias, a shared expert,
//! dense leading layers), with an MTP layer at the checkpoint's root.

use inference_tensor::Result;
use serde::Serialize;

use crate::{
    decoder::{DecoderSpec, MoeRouting, MoeSpec, MtpSpec, NormNames, SharedExpertSpec, exaone4},
    moe::ExpertProjNames,
};

const MLP: &str = "mlp";
const SHARED_EXPERTS: &str = "shared_experts";
const DENSE: &str = "dense";
const FULL_ATTENTION: &str = "full_attention";
// DeepSeek V3's MTP under EXAONE's names; the head's layer is pre-norm like the stack's
const MTP: MtpSpec = MtpSpec {
    embedding_norm: "mtp.pre_fc_norm_embedding",
    hidden_norm: "mtp.pre_fc_norm_hidden",
    projection: "mtp.fc",
    block: "mtp.layers.0",
    final_norm: Some("mtp.norm"),
    dense_block: true,
    mask_first_position: false,
    at_root: true,
};

#[derive(Debug, Clone, Serialize, inference_json::FromJson)]
pub struct Config {
    #[serde(flatten)]
    pub base: exaone4::Config,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub num_shared_experts: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub first_k_dense_replace: usize,
    /// `dense` or `sparse` per layer, over `first_k_dense_replace`.
    pub mlp_layer_types: Option<Vec<String>>,
    #[serde(default)]
    pub n_group: Option<usize>,
    /// The MTP layer's attention, `full_attention` (NoPE) as the stack's global layers.
    pub mtp_layer_types: Option<Vec<String>>,
    /// Injected by the loader when `--mtp` asks for the checkpoint's MTP layer.
    #[serde(default, rename = "_inference_mtp")]
    pub mtp: bool,
}

impl Config {
    fn is_dense(&self, layer: usize) -> bool {
        match &self.mlp_layer_types {
            Some(types) => types.get(layer).is_some_and(|kind| kind == DENSE),
            None => layer < self.first_k_dense_replace,
        }
    }

    pub fn decoder_spec(&self) -> Result<DecoderSpec> {
        if self.n_group.is_some_and(|groups| groups > 1) {
            inference_tensor::bail!(
                "K-EXAONE with grouped expert routing (`n_group` > 1) is not supported"
            );
        }
        let layers = self.base.num_hidden_layers;
        let mut spec = self.base.decoder_spec()?;
        spec.norm_names = NormNames::PRE;
        spec.moe = Some(MoeSpec {
            num_experts: self.num_experts,
            intermediate_size: self.moe_intermediate_size,
            routing: MoeRouting::SigmoidBiasedTopK {
                k: self.num_experts_per_tok,
                renormalize: self.norm_topk_prob,
                scale: self.routed_scaling_factor,
            },
            quantized_router: false,
            expert_names: ExpertProjNames::DEFAULT,
            name: MLP,
            dense_layers: (0..layers).filter(|&layer| self.is_dense(layer)).collect(),
            cuda_decode_graphs: true,
            shared_expert: (self.num_shared_experts > 0).then_some(SharedExpertSpec {
                intermediate_size: self.moe_intermediate_size * self.num_shared_experts,
                name: SHARED_EXPERTS,
                gated: false,
            }),
        });
        if self.mtp {
            if self.base.num_nextn_predict_layers == 0 {
                inference_tensor::bail!(
                    "`--mtp` asked for K-EXAONE's MTP layer, which this config has none of"
                );
            }
            spec.mtp = Some(MTP);
            // the head's paged layer follows the stack's, NoPE when it is a global one
            let global = self
                .mtp_layer_types
                .as_ref()
                .and_then(|types| types.first())
                .is_some_and(|kind| kind == FULL_ATTENTION);
            if global && self.base.sliding_window.is_some() {
                spec.no_rope_layers.push(layers);
            }
        }
        Ok(spec)
    }
}

pub type Model = crate::decoder::CausalLm;
