//! DeepSeek-OCR-2's query encoder (`Qwen2Decoder2Encoder`): Qwen2 layers over a view's SAM tokens followed by as
//! many learned queries, the tokens attending each other both ways and each query the tokens and the queries before
//! it; the queries' outputs are the view's features.

use std::collections::HashMap;
use std::sync::Mutex;

use inference_quant::ShardedVarBuilder;
use inference_tensor::{DType, Device, Result, Tensor};

use crate::{
    attention::{AttentionMask, FlashParams},
    decoder::{DecoderSpec, DecoderStack, EMBEDS_ONLY, LayerMasks, RopeKind, load_all_layers},
    device_map::DeviceMappedMask,
    layers::Activation,
    model::{ModelForwardContext, NormalLoadingMetadata},
    paged_attention::AttentionImplementation,
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::{Config, QueryEncoderWidth};

const RMS_NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

pub struct QueryEncoder {
    stack: DecoderStack,
    // per view token count, the view's side and its queries (`query_768`, `query_1024`)
    queries: HashMap<usize, (usize, Tensor)>,
    masks: Mutex<HashMap<usize, Tensor>>,
}

impl QueryEncoder {
    /// `vb` is the checkpoint's `model.qwen2_model`; one query table per view side in `sides`.
    pub fn load(vb: ShardedVarBuilder, width: &QueryEncoderWidth, sides: &[usize]) -> Result<Self> {
        let tokens = |side: usize| Config::grid(side).pow(2);
        let max_tokens = sides.iter().map(|&side| tokens(side)).max().unwrap_or(0);
        let spec = DecoderSpec {
            vocab_size: EMBEDS_ONLY,
            hidden_size: width.dim,
            intermediate_size: width.intermediate_size,
            num_heads: width.heads,
            num_kv_heads: width.kv_heads,
            head_dim: width.dim / width.heads,
            hidden_act: Activation::Silu,
            rms_norm_eps: RMS_NORM_EPS,
            rope: RopeKind::Default { theta: ROPE_THETA },
            // the view's tokens, then as many queries
            max_position_embeddings: 2 * max_tokens,
            qkv_bias: true,
            layer_windows: vec![None; width.layers],
            ..Default::default()
        };
        let device = vb.device().clone();
        let stack = {
            let _all = load_all_layers();
            DecoderStack::new(
                &spec,
                vb.pp("model").pp("model"),
                None,
                true,
                NormalLoadingMetadata::single_device(&device),
                &AttentionImplementation::Eager,
            )?
        };
        let queries = sides
            .iter()
            .map(|&side| {
                let table = vb
                    .pp(format!("query_{side}"))
                    .get((tokens(side), width.dim), "weight")?;
                Ok((tokens(side), (side, table)))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            stack,
            queries,
            masks: Mutex::new(HashMap::new()),
        })
    }

    // tokens see every token, queries every token and the queries up to their own
    fn mask(&self, tokens: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let mut masks = self.masks.lock().expect("query mask cache poisoned");
        if let Some(mask) = masks.get(&tokens) {
            return Ok(mask.clone());
        }
        let len = 2 * tokens;
        let mask: Vec<f32> = (0..len)
            .flat_map(|q| {
                (0..len).map(move |k| {
                    let visible = if q < tokens { k < tokens } else { k <= q };
                    if visible { 0.0 } else { f32::NEG_INFINITY }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (len, len), device)?.to_dtype(dtype)?;
        masks.insert(tokens, mask.clone());
        Ok(mask)
    }

    /// (B, N, C) SAM tokens to the queries' (B, N, C) outputs.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, tokens, dim) = xs.dims3()?;
        let Some((_, queries)) = self.queries.get(&tokens) else {
            inference_tensor::bail!("DeepSeek-OCR-2 has no queries for a view of {tokens} tokens");
        };
        let queries = queries
            .to_dtype(xs.dtype())?
            .unsqueeze(0)?
            .broadcast_as((batch, tokens, dim))?;
        let xs = Tensor::cat(&[xs, &queries.contiguous()?], 1)?;
        let len = 2 * tokens;
        let mask = self.mask(tokens, xs.dtype(), xs.device())?;
        let masks = LayerMasks::new(
            Some(DeviceMappedMask::new(
                AttentionMask::Custom(mask),
                &*self.stack.mapper,
            )?),
            None,
            None,
        );
        let offsets = vec![0; batch];
        let context_lens = vec![(0, len); batch];
        let position_ids = vec![len; batch];
        let flash = FlashParams::empty(false);
        let mut ctx =
            ModelForwardContext::new(&offsets, &context_lens, &position_ids, None, &flash);
        self.stack
            .forward(xs, &masks, None, &mut ctx)?
            .narrow(1, tokens, tokens)
    }

    /// Tensors under the checkpoint's `model.qwen2_model` prefix.
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.stack.residual_uvb(&uvb.pp("model").pp("model"), true);
        for (side, table) in self.queries.values() {
            uvb.pp(format!("query_{side}"))
                .add_tensor("weight", table.clone());
        }
        uvb.to_safetensors()
    }
}
