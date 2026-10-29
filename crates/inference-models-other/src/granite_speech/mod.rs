//! IBM Granite Speech: a CTC Conformer encoder, a windowed BLIP-2 Q-Former projector, and a Granite text model whose
//! audio placeholders take the projected features.

pub mod config;
pub mod encoder;
pub mod features;
pub mod inputs_processor;

use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{Linear, Module};
use inference_tensor::{D, DType, Device, Result, Tensor};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    blip2_qformer::{QFormer, QFormerDims},
    granite::GraniteMoeHybrid,
    kv_cache::EitherCache,
    layers::linear_b,
    model::{IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata, NormalModel},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    utils::unvarbuilder::UnVarBuilder,
};
pub use config::Config;
use encoder::Encoder;

pub struct GraniteSpeechSpecificArgs {
    /// `(1, frames, features)` stacked log-mel features of each audio clip in the batch, in prompt order.
    pub audio_features: Vec<Tensor>,
}

/// transformers' `GraniteSpeechEncoderProjector`: a Q-Former reads each window of encoder frames with
/// `window / downsample` queries.
struct Projector {
    query: Tensor,
    qformer: QFormer,
    linear: Linear,
    window: usize,
}

impl Projector {
    fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let p = &cfg.projector_config;
        let queries = cfg.window_size / cfg.downsample_rate;
        Ok(Self {
            query: vb.get((1, queries, p.hidden_size), "query")?,
            qformer: QFormer::new(
                &QFormerDims {
                    hidden: p.hidden_size,
                    heads: p.num_attention_heads,
                    layers: p.num_hidden_layers,
                    intermediate: p.intermediate_size,
                    layer_norm_eps: p.layer_norm_eps,
                    encoder_hidden: p.encoder_hidden_size,
                },
                vb.pp("qformer"),
            )?,
            linear: linear_b(
                p.hidden_size,
                cfg.text_config.hidden_size,
                true,
                vb.pp("linear"),
            )?,
            window: cfg.window_size,
        })
    }

    /// `(1, frames, encoder)` to `(windows * queries, text hidden)`.
    fn forward(&self, encoded: &Tensor) -> Result<Tensor> {
        let (batch, frames, width) = encoded.dims3()?;
        let windows = frames.div_ceil(self.window);
        let encoded = encoded
            .pad_with_zeros(1, 0, windows * self.window - frames)?
            .reshape((batch * windows, self.window, width))?;
        let (_, queries, hidden) = self.query.dims3()?;
        let query = self
            .query
            .to_dtype(encoded.dtype())?
            .broadcast_as((batch * windows, queries, hidden))?
            .contiguous()?;
        let out = self.qformer.forward(&query, &encoded)?;
        self.linear
            .forward(&out.reshape((batch * windows * queries, hidden))?)
    }
}

pub struct GraniteSpeechModel {
    language_model: GraniteMoeHybrid,
    encoder: Encoder,
    projector: Projector,
    audio_token: u32,
}

impl GraniteSpeechModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if cfg.has_lora_adapter {
            inference_tensor::bail!(
                "this Granite Speech checkpoint switches a LoRA adapter on for audio inputs, which is not supported"
            );
        }
        let device = normal_loading_metadata.real_device.clone();
        let audio_vb = vb.clone().without_lora_registry().set_device(device);
        let encoder = Encoder::new(&cfg.encoder_config, audio_vb.pp("encoder"))?;
        let projector = Projector::new(cfg, audio_vb.pp("projector"))?;
        let text_vb = vb.pp("language_model");
        let lm_head = if cfg.text_config.tie_word_embeddings {
            text_vb.pp("model").pp("embed_tokens")
        } else {
            text_vb.pp("lm_head")
        };
        let language_model = GraniteMoeHybrid::new_inner(
            &cfg.text_config,
            text_vb.pp("model"),
            lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?
        // transformers applies the lm_head to the base Granite model's states, skipping its logits_scaling
        .without_logits_scaling();
        Ok(Self {
            language_model,
            encoder,
            projector,
            audio_token: cfg.audio_token_index,
        })
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        args: GraniteSpeechSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut embeds = self.language_model.token_embeddings(input_ids)?;
        if !args.audio_features.is_empty() {
            let projected = args
                .audio_features
                .iter()
                .map(|features| {
                    let encoded = self
                        .encoder
                        .forward(&features.to_dtype(self.projector.query.dtype())?)?;
                    self.projector.forward(&encoded)
                })
                .collect::<Result<Vec<_>>>()?;
            let projected = Tensor::cat(&projected, 0)?;
            let positions = input_ids
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?
                .into_iter()
                .enumerate()
                .filter(|&(_, id)| id == self.audio_token)
                .map(|(i, _)| u32::try_from(i).expect("a token position fits u32"))
                .collect::<Vec<_>>();
            if positions.len() != projected.dim(0)? {
                inference_tensor::bail!(
                    "Granite Speech audio features and audio tokens do not match, tokens: {}, features: {}",
                    positions.len(),
                    projected.dim(0)?
                );
            }
            let shape = embeds.shape().clone();
            let hidden = embeds.dim(D::Minus1)?;
            let flat = embeds.reshape(((), hidden))?;
            let positions = Tensor::new(positions.as_slice(), flat.device())?;
            let replace =
                (projected.to_dtype(flat.dtype())? - flat.index_select(&positions, 0)?)?;
            embeds = flat.index_add(&positions, &replace, 0)?.reshape(shape)?;
        }
        // transformers scales whatever embeddings it is given, the audio features with them
        let embeds = self.language_model.scale_embeddings(embeds)?;
        self.language_model
            .forward_embeds(input_ids, embeds, ctx, &|_, xs| Ok(xs))
    }
}

impl IsqModel for GraniteSpeechModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let text = uvb.pp("language_model");
        text.extend(self.language_model.residual_tensors_m(text.pp("model")));
        uvb.pp("encoder").extend(self.encoder.residual_tensors());
        let projector = uvb.pp("projector");
        projector.add_tensor("query", self.projector.query.clone());
        projector
            .pp("qformer")
            .extend(self.projector.qformer.residual_tensors());
        projector.pp("linear").add(&self.projector.linear);
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for GraniteSpeechModel {}
impl crate::model::BlockDiffusionMixin for GraniteSpeechModel {}
impl AnyMoeBaseModelMixin for GraniteSpeechModel {}

impl MultimodalModel for GraniteSpeechModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let args = *model_specific_args
            .downcast::<GraniteSpeechSpecificArgs>()
            .expect("Cannot downcast into `GraniteSpeechSpecificArgs`");
        self.forward_inner(input_ids, args, ctx)
    }

    fn device(&self) -> &Device {
        self.language_model.device()
    }

    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }

    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(GraniteSpeechSpecificArgs {
            audio_features: Vec::new(),
        })
    }

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
}
