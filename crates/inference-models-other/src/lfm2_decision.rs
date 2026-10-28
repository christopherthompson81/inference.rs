//! LiquidAI's d1-omni (llama.cpp's `lfm2-d1-omni` decision type): an LFM2 trunk read bidirectionally (centred
//! three-tap short convolutions, non-causal attention), then laya's decision head (a question-type embedding, a few
//! pre-norm blocks with biases, and a scorer over each option's `[MASK]` marker). It loads from llama.cpp's GGUF under
//! its tensor names. The logits are left uncalibrated: the decision profile applies the temperatures.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::Arc;

use inference_audio::nemo::{NemoMel, NemoMelConfig};
use inference_models_speech::parakeet::{Encoder as AudioEncoder, EncoderConfig};
use inference_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};
use inference_tensor::nn::{Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use inference_tensor::{D, DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::{AttentionMask, FlashParams, SdpaParams},
    decision::{DecisionMedia, DecisionRows},
    layers::{
        RmsNorm, RotaryEmbedding, Sdpa, apply_rotary_preselected_qk, dense_embedding, layer_norm,
    },
    lfm2_vl,
    media_inputs::{image_processor::ImagePreProcessor, preprocessor_config::PreProcessorConfig},
    model::{EmbeddingModel, IsqModel, NormalLoadingMetadata},
    modern_bert_decision::{self, Block},
    utils::unvarbuilder::UnVarBuilder,
};

const BLOCKS: &str = "blk";
const QUESTION_TYPES: usize = 3;
// the short convolutions' taps: the previous token, the token, the next one
const CONV_TAPS: usize = 3;
// the projector GGUF's tensors under LFM2-VL's names (`lfm2_vl_bindings`)
const VISION_TOWER: &str = "model.vision_tower";
const PROJECTOR: &str = "model.multi_modal_projector";
// PIL's bilinear filter id, which llama.cpp's LFM2 image preprocessor resizes with
const BILINEAR: usize = 2;
// the projector GGUF's audio tensors: the FastConformer, its adapter (`mlp.0` to `mlp.3`) and the residual block after
const AUDIO_ENCODER: &str = "a";
const AUDIO_ADAPTER: &str = "mm.a.mlp";
// llama.cpp's d1-omni audio front end: 16 kHz NeMo features, clips padded to half a second and cut at 30 s
const AUDIO_SAMPLE_RATE: u32 = 16000;
const AUDIO_N_FFT: usize = 512;
const AUDIO_WINDOW: usize = 400;
const AUDIO_HOP: usize = 160;
const AUDIO_PREEMPHASIS: f32 = 0.97;
const AUDIO_MIN_SAMPLES: usize = AUDIO_SAMPLE_RATE as usize / 2;
const AUDIO_MAX_SAMPLES: usize = 30 * AUDIO_SAMPLE_RATE as usize;
const AUDIO_NORM_EPS: f64 = 1e-5;
// the dw_striding pre-encode's module indices: conv, (relu), depthwise, pointwise, (relu), depthwise, pointwise
const PRE_ENCODE_CONVS: [usize; 5] = [0, 2, 3, 5, 6];
// llama.cpp's block tensor stems (the name up to its first dot) against our encoder's, transformers' Parakeet names
const AUDIO_BLOCK_NAMES: [(&str, &str); 20] = [
    ("ffn_norm", "norm_feed_forward1"),
    ("ffn_up", "feed_forward1.linear1"),
    ("ffn_down", "feed_forward1.linear2"),
    ("ln1", "norm_self_att"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_out", "self_attn.o_proj"),
    ("linear_pos", "self_attn.relative_k_proj"),
    ("norm_conv", "norm_conv"),
    ("conv_pw1", "conv.pointwise_conv1"),
    ("conv_dw", "conv.depthwise_conv"),
    ("conv_norm", "conv.norm"),
    ("conv_pw2", "conv.pointwise_conv2"),
    ("ffn_norm_1", "norm_feed_forward2"),
    ("ffn_up_1", "feed_forward2.linear1"),
    ("ffn_down_1", "feed_forward2.linear2"),
    ("ln2", "norm_out"),
    ("pos_bias_u", "self_attn.bias_u"),
    ("pos_bias_v", "self_attn.bias_v"),
];

/// The trunk and head sizes, from the GGUF's `lfm2.*` keys.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    /// per block, the trunk's then the head's; 0 marks a short-convolution block
    pub num_key_value_heads: Vec<usize>,
    pub intermediate_sizes: Vec<usize>,
    pub head_layers: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub layer_norm_eps: f64,
    /// the projector GGUF's vision tower, when one loads beside the model
    #[serde(default)]
    pub vision: Option<VisionConfig>,
    /// the projector GGUF's audio encoder, from its `clip.audio.*` keys
    #[serde(default)]
    pub audio: Option<EncoderConfig>,
}

/// LFM2-VL's tower and the projector's MLP width, from the projector GGUF's `clip.vision.*` keys.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VisionConfig {
    pub tower: lfm2_vl::config::VisionConfig,
    pub projector_hidden_size: usize,
    pub downsample_factor: usize,
}

impl Config {
    pub fn trunk_layers(&self) -> usize {
        self.num_key_value_heads.len() - self.head_layers
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// `out(c * conv(b * x))` over `[b | c | x] = in_proj(xs)`, the convolution centred on each token.
struct ShortConv {
    in_proj: Arc<dyn QuantMethod>,
    out_proj: Arc<dyn QuantMethod>,
    taps: [Tensor; CONV_TAPS],
}

impl ShortConv {
    /// The last of `media_len` media entries does not read the text on its right.
    fn forward(&self, xs: &Tensor, media_len: usize) -> Result<Tensor> {
        let (b_sz, seq_len, width) = xs.dims3()?;
        let bcx = self.in_proj.forward(xs)?;
        let part = |i: usize| bcx.narrow(D::Minus1, i * width, width);
        let bx = (part(0)? * part(2)?)?;
        let mut y = bx.broadcast_mul(&self.taps[1])?;
        if seq_len > 1 {
            // a tap outside the sequence reads zero
            let zero = Tensor::zeros((b_sz, 1, width), bx.dtype(), bx.device())?;
            let prev = Tensor::cat(&[&zero, &bx.narrow(1, 0, seq_len - 1)?], 1)?;
            let mut next = Tensor::cat(&[&bx.narrow(1, 1, seq_len - 1)?, &zero], 1)?;
            if media_len > 0 && media_len < seq_len {
                let mut keep = vec![1f32; seq_len];
                keep[media_len - 1] = 0.0;
                let keep =
                    Tensor::from_vec(keep, (1, seq_len, 1), bx.device())?.to_dtype(bx.dtype())?;
                next = next.broadcast_mul(&keep)?;
            }
            y = (y + prev.broadcast_mul(&self.taps[0])?)?;
            y = (y + next.broadcast_mul(&self.taps[2])?)?;
        }
        self.out_proj.forward(&(part(1)? * y)?)
    }
}

/// Grouped-query attention over the whole prompt, with RMS-normed q and k before NeoX rotary positions.
struct Attention {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    v: Arc<dyn QuantMethod>,
    o: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor, mask: &AttentionMask) -> Result<Tensor> {
        let (b_sz, seq_len, width) = xs.dims3()?;
        let heads = |proj: &Arc<dyn QuantMethod>, n: usize, norm: Option<&RmsNorm>| {
            let x = proj
                .forward(xs)?
                .reshape((b_sz, seq_len, n, self.head_dim))?;
            let x = match norm {
                Some(norm) => norm.forward(&x)?,
                None => x,
            };
            x.transpose(1, 2)?.contiguous()
        };
        let q = heads(&self.q, self.heads, Some(&self.q_norm))?;
        let k = heads(&self.k, self.kv_heads, Some(&self.k_norm))?;
        let v = heads(&self.v, self.kv_heads, None)?;
        let (cos, sin) = self.rotary.get_cos_sin()?;
        let (q, k) = apply_rotary_preselected_qk(
            &q,
            &k,
            &cos.narrow(0, 0, seq_len)?,
            &sin.narrow(0, 0, seq_len)?,
            true,
        )?;
        let params = SdpaParams {
            n_kv_groups: self.heads / self.kv_heads,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let attn =
            Sdpa.run_attention(&q, &k, &v, mask, Some(&FlashParams::empty(false)), &params)?;
        self.o
            .forward(&attn.transpose(1, 2)?.reshape((b_sz, seq_len, width))?)
    }
}

enum Mixer {
    Conv(ShortConv),
    Attention(Attention),
}

struct TrunkLayer {
    norm: RmsNorm,
    mixer: Mixer,
    ffn_norm: RmsNorm,
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl TrunkLayer {
    fn forward(&self, xs: &Tensor, mask: &AttentionMask, media_len: usize) -> Result<Tensor> {
        let normed = self.norm.forward(xs)?;
        let mixed = match &self.mixer {
            Mixer::Conv(conv) => conv.forward(&normed, media_len)?,
            Mixer::Attention(attn) => attn.forward(&normed, mask)?,
        };
        let xs = (xs + mixed)?;
        let normed = self.ffn_norm.forward(&xs)?;
        let gated = (self.gate.forward(&normed)?.silu()? * self.up.forward(&normed)?)?;
        xs + self.down.forward(&gated)?
    }
}

/// LFM2-VL's tower, projector and image processor; d1-omni's projector uses the exact GELU and no input norm.
struct Vision {
    tower: lfm2_vl::vision::VisionModel,
    projector: lfm2_vl::MultiModalProjector,
    processor: lfm2_vl::inputs_processor::Lfm2VlImageProcessor,
    preprocessor: PreProcessorConfig,
    device: Device,
}

/// The projector's FastConformer (as our Parakeet encoder, in f32 as llama.cpp runs it), its adapter into the
/// trunk's width and the residual block after it, with NeMo's features.
struct Audio {
    encoder: AudioEncoder,
    mel: NemoMel,
    adapter: [Linear; 2],
    adapter_norm: LayerNorm,
    residual: [Linear; 2],
    residual_norm: LayerNorm,
    device: Device,
}

/// The projector's `a.*` tensors under our encoder's names, in PyTorch's conv layouts.
fn audio_encoder_weights(vb: &ShardedVarBuilder, layers: usize) -> Result<HashMap<String, Tensor>> {
    let get = |name: &str| vb.get_unchecked_dtype(name, DType::F32);
    let mut weights = HashMap::new();
    for i in PRE_ENCODE_CONVS {
        let conv = format!("subsampling.layers.{i}");
        weights.insert(
            format!("{conv}.weight"),
            get(&format!("conv1d.{i}.weight"))?,
        );
        weights.insert(
            format!("{conv}.bias"),
            get(&format!("conv1d.{i}.bias"))?.flatten_all()?,
        );
    }
    for suffix in ["weight", "bias"] {
        weights.insert(
            format!("subsampling.linear.{suffix}"),
            get(&format!("pre_encode.out.{suffix}"))?,
        );
    }
    for layer in 0..layers {
        let block = vb.pp("blk").pp(layer);
        for (theirs, ours) in AUDIO_BLOCK_NAMES {
            let ours = format!("layers.{layer}.{ours}");
            if block.contains_tensor(theirs) {
                weights.insert(ours, block.get_unchecked_dtype(theirs, DType::F32)?);
                continue;
            }
            for suffix in ["weight", "bias"] {
                let name = format!("{theirs}.{suffix}");
                if !block.contains_tensor(&name) {
                    continue;
                }
                let t = block.get_unchecked_dtype(&name, DType::F32)?;
                // the 1x1 convs are stored as matrices and the depthwise one without its group axis
                let t = match (theirs, suffix) {
                    ("conv_pw1" | "conv_pw2", "weight") => t.unsqueeze(2)?,
                    ("conv_dw", "weight") => t.unsqueeze(1)?,
                    _ => t,
                };
                weights.insert(format!("{ours}.{suffix}"), t);
            }
        }
    }
    Ok(weights)
}

impl Audio {
    fn new(cfg: &EncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let weights = audio_encoder_weights(&vb.pp(AUDIO_ENCODER), cfg.num_hidden_layers)?;
        let encoder =
            AudioEncoder::new(cfg, VarBuilder::from_tensors(weights, DType::F32, &device))?;
        let mlp = vb.pp(AUDIO_ADAPTER);
        let get = |i: usize, suffix: &str| mlp.pp(i).get_unchecked_dtype(suffix, DType::F32);
        let linear = |i: usize| -> Result<Linear> {
            Ok(Linear::new(get(i, "weight")?, Some(get(i, "bias")?)))
        };
        let norm = |i: usize| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                get(i, "weight")?,
                get(i, "bias")?,
                AUDIO_NORM_EPS,
            ))
        };
        let mel = NemoMel::new(NemoMelConfig {
            sample_rate: AUDIO_SAMPLE_RATE,
            n_fft: AUDIO_N_FFT,
            win_length: AUDIO_WINDOW,
            hop_length: AUDIO_HOP,
            n_mels: cfg.num_mel_bins,
            preemphasis: Some(AUDIO_PREEMPHASIS),
            normalize: true,
        });
        Ok(Self {
            encoder,
            mel,
            adapter_norm: norm(0)?,
            adapter: [linear(1)?, linear(3)?],
            residual_norm: norm(4)?,
            residual: [linear(5)?, linear(6)?],
            device,
        })
    }

    /// One clip's `(rows, width)` embeddings: NeMo features over at most 30 s, the encoder, then the adapter.
    fn embeds(&self, audio: &inference_audio::AudioInput) -> Result<Tensor> {
        let mut pcm =
            inference_audio::mel::resample(&audio.to_mono(), audio.sample_rate, AUDIO_SAMPLE_RATE)
                .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        pcm.truncate(AUDIO_MAX_SAMPLES);
        if pcm.len() < AUDIO_MIN_SAMPLES {
            pcm.resize(AUDIO_MIN_SAMPLES, 0.0);
        }
        let (features, frames) = self
            .mel
            .features(&pcm)
            .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
        let n_mels = self.mel.config().n_mels;
        let features = Tensor::from_vec(features, (1, frames, n_mels), &self.device)?;
        let xs = self.encoder.forward(&features)?.squeeze(0)?;
        let mlp =
            |[up, down]: &[Linear; 2], xs: &Tensor| down.forward(&up.forward(xs)?.gelu_erf()?);
        let xs = mlp(&self.adapter, &self.adapter_norm.forward(&xs)?)?;
        &xs + mlp(&self.residual, &self.residual_norm.forward(&xs)?)?
    }
}

pub struct Model {
    vision: Option<Vision>,
    audio: Option<Audio>,
    embed_tokens: Embedding,
    layers: Vec<TrunkLayer>,
    norm: RmsNorm,
    type_emb: Embedding,
    blocks: Vec<Block>,
    scorer_norm: LayerNorm,
    scorer_hidden: Arc<dyn QuantMethod>,
    scorer_out: Arc<dyn QuantMethod>,
    max_positions: usize,
    device: Device,
}

fn linear(i: usize, o: usize, bias: bool, vb: ShardedVarBuilder) -> Result<Arc<dyn QuantMethod>> {
    inference_quant::linear_b(i, o, bias, &None, vb)
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = &normal_loading_metadata.mapper;
        let device = normal_loading_metadata.real_device.clone();
        let width = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let layers = (0..cfg.trunk_layers())
            .map(|i| {
                let vb = mapper.set_device(i, vb.pp(BLOCKS).pp(i), false);
                let kv_heads = cfg.num_key_value_heads[i];
                let mixer = match kv_heads {
                    0 => {
                        let conv = vb.pp("shortconv");
                        let weight = conv.pp("conv").get((width, CONV_TAPS), "weight")?;
                        let tap = |k: usize| weight.narrow(1, k, 1)?.squeeze(1);
                        Mixer::Conv(ShortConv {
                            in_proj: linear(width, 3 * width, false, conv.pp("in_proj"))?,
                            out_proj: linear(width, width, false, conv.pp("out_proj"))?,
                            taps: [tap(0)?, tap(1)?, tap(2)?],
                        })
                    }
                    kv_heads => Mixer::Attention(Attention {
                        q: linear(
                            width,
                            cfg.num_attention_heads * head_dim,
                            false,
                            vb.pp("attn_q"),
                        )?,
                        k: linear(width, kv_heads * head_dim, false, vb.pp("attn_k"))?,
                        v: linear(width, kv_heads * head_dim, false, vb.pp("attn_v"))?,
                        o: linear(
                            cfg.num_attention_heads * head_dim,
                            width,
                            false,
                            vb.pp("attn_output"),
                        )?,
                        q_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("attn_q_norm"))?,
                        k_norm: RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("attn_k_norm"))?,
                        rotary: Arc::new(RotaryEmbedding::new(
                            cfg.rope_theta,
                            head_dim,
                            cfg.max_position_embeddings,
                            vb.device(),
                            true,
                            vb.dtype(),
                        )?),
                        heads: cfg.num_attention_heads,
                        kv_heads,
                        head_dim,
                    }),
                };
                let ff = cfg.intermediate_sizes[i];
                Ok(TrunkLayer {
                    norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("attn_norm"))?,
                    mixer,
                    ffn_norm: RmsNorm::new(width, cfg.rms_norm_eps, vb.pp("ffn_norm"))?,
                    gate: linear(width, ff, false, vb.pp("ffn_gate"))?,
                    up: linear(width, ff, false, vb.pp("ffn_up"))?,
                    down: linear(ff, width, false, vb.pp("ffn_down"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let head = mapper.set_nm_device(vb.clone(), false);
        let norm = |vb: ShardedVarBuilder| {
            layer_norm(
                width,
                LayerNormConfig {
                    eps: cfg.layer_norm_eps,
                    remove_mean: true,
                    affine: true,
                },
                vb,
            )
        };
        let blocks = (cfg.trunk_layers()..cfg.num_key_value_heads.len())
            .map(|i| {
                let vb = head.pp(BLOCKS).pp(i);
                let ff = cfg.intermediate_sizes[i];
                Ok(Block::from_parts(
                    (norm(vb.pp("attn_norm"))?, norm(vb.pp("ffn_norm"))?),
                    (
                        linear(width, 3 * width, true, vb.pp("attn_qkv"))?,
                        linear(width, width, true, vb.pp("attn_output"))?,
                    ),
                    (
                        linear(width, ff, true, vb.pp("ffn_up"))?,
                        linear(ff, width, true, vb.pp("ffn_down"))?,
                    ),
                    width,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let cls = head.pp("cls");
        // a 1-D GGUF tensor, the scorer's last projection to one logit
        let scorer_out = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            inference_tensor::nn::Linear::new(
                cls.pp("output").get(width, "weight")?.unsqueeze(0)?,
                Some(cls.pp("output").get(1, "bias")?),
            ),
        ))?);
        let vision = cfg
            .vision
            .as_ref()
            .map(|vision| -> Result<Vision> {
                let shape = lfm2_vl::ProjectorShape {
                    downsample_factor: vision.downsample_factor,
                    use_layernorm: false,
                    hidden_size: vision.projector_hidden_size,
                    out_size: width,
                    bias: true,
                    activation: crate::layers::Activation::Gelu,
                };
                let tiling = lfm2_vl::config::Tiling {
                    downsample_factor: vision.downsample_factor,
                    ..Default::default()
                };
                // llama.cpp resizes bilinearly (PIL's 2), the pixels normalized to [-1, 1]
                let preprocessor = PreProcessorConfig {
                    resampling: Some(BILINEAR),
                    ..Default::default()
                };
                Ok(Vision {
                    tower: lfm2_vl::vision::VisionModel::new(
                        &vision.tower,
                        head.pp(VISION_TOWER).pp("vision_model"),
                    )?,
                    projector: lfm2_vl::MultiModalProjector::with_shape(
                        vision.tower.hidden_size,
                        &shape,
                        head.pp(PROJECTOR),
                    )?,
                    processor: lfm2_vl::inputs_processor::Lfm2VlImageProcessor::with_tiling(
                        &tiling,
                        &preprocessor,
                    ),
                    preprocessor,
                    device: head.device().clone(),
                })
            })
            .transpose()?;
        let audio = cfg
            .audio
            .as_ref()
            .map(|audio| Audio::new(audio, head.clone()))
            .transpose()?;
        Ok(Self {
            vision,
            audio,
            embed_tokens: dense_embedding(cfg.vocab_size, width, vb.pp("token_embd"), &None)?,
            layers,
            norm: RmsNorm::new(width, cfg.rms_norm_eps, head.pp("token_embd_norm"))?,
            type_emb: Embedding::new(
                head.pp("token_types")
                    .get((QUESTION_TYPES, width), "weight")?,
                width,
            ),
            blocks,
            scorer_norm: norm(cls.pp("norm"))?,
            scorer_hidden: linear(width, width, true, cls.clone())?,
            scorer_out,
            max_positions: cfg.max_position_embeddings,
            device,
        })
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let root = UnVarBuilder::new();
        root.pp("token_embd").add(&self.embed_tokens);
        root.pp("token_embd_norm").add(&self.norm);
        root.pp("token_types").add(&self.type_emb);
        root.pp("cls").pp("norm").add(&self.scorer_norm);
        root.to_safetensors()
    }
}

impl Model {
    /// A row's media as embeddings ahead of its text: its audio clip through the encoder and adapter, or each
    /// image's crops through the tower and projector.
    fn media_embeds(&self, media: &DecisionMedia, dtype: DType) -> Result<Tensor> {
        if let [audio] = media.audios.as_slice() {
            let Some(encoder) = &self.audio else {
                inference_tensor::bail!(
                    "d1-omni reads audio with its projector, which did not load"
                );
            };
            return encoder
                .embeds(audio)?
                .to_device(&self.device)?
                .to_dtype(dtype);
        }
        let Some(vision) = &self.vision else {
            inference_tensor::bail!("d1-omni reads images with its projector, which did not load");
        };
        let device = &vision.device;
        let processed = vision.processor.preprocess(
            media.images.clone(),
            Vec::new(),
            &vision.preprocessor,
            device,
            (media.images.len(), 1),
        )?;
        let (Some(mask), Some(shapes)) = (processed.pixel_attention_mask, processed.tgt_sizes)
        else {
            inference_tensor::bail!("the LFM2-VL processor gave no patch masks");
        };
        lfm2_vl::image_features(
            &vision.tower,
            &vision.projector,
            &processed.pixel_values.to_device(device)?,
            &mask.to_device(device)?,
            &shapes.to_device(device)?,
            self.embed_tokens.hidden_size(),
        )?
        .to_device(&self.device)?
        .to_dtype(dtype)
    }

    /// The trunk and the norm over `xs`, whose first `media_len` entries are media the text is not read by.
    fn trunk(&self, xs: Tensor, media_len: usize) -> Result<Tensor> {
        let mask = match media_len {
            0 => AttentionMask::None,
            m => AttentionMask::Custom(kind_mask(xs.dim(1)?, m, false, xs.dtype(), xs.device())?),
        };
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, &mask, media_len)?;
        }
        self.norm.forward(&xs)
    }

    /// The head blocks over the trunk's states with each row's question-type embedding added.
    fn head(&self, xs: &Tensor, kinds: &[u32], media_len: usize) -> Result<Tensor> {
        let kinds = Tensor::new(kinds, xs.device())?;
        let mut xs = xs.broadcast_add(&self.type_emb.forward(&kinds)?.unsqueeze(1)?)?;
        let mask = match media_len {
            0 => AttentionMask::None,
            m => AttentionMask::Custom(kind_mask(xs.dim(1)?, m, true, xs.dtype(), xs.device())?),
        };
        for block in &self.blocks {
            xs = block.forward_masked(&xs, &mask)?;
        }
        Ok(xs)
    }
}

/// The additive mask over `[media | text]`: media rows never read text; in the head, each kind reads only its own.
fn kind_mask(
    len: usize,
    media_len: usize,
    own_kind_only: bool,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0f32; len * len];
    for i in 0..len {
        for j in 0..len {
            let (row_media, col_media) = (i < media_len, j < media_len);
            let hidden = match own_kind_only {
                true => row_media != col_media,
                false => row_media && !col_media,
            };
            if hidden {
                mask[i * len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, len, len), device)?.to_dtype(dtype)
}

impl EmbeddingModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        extras: &crate::embedding::EmbeddingExtras,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Some(rows) = &extras.decision else {
            inference_tensor::bail!("a decision model answers decision questions only");
        };
        let text = self.embed_tokens.forward(input_ids)?;
        let xs = match rows.media.iter().any(Option::is_some) {
            false => {
                let xs = self.trunk(text, 0)?;
                self.head(&xs, &rows.kinds, 0)?
            }
            // a row with media runs alone, its media ahead of its text and its markers moved past them
            true => {
                let mut picked = Vec::with_capacity(rows.kinds.len());
                for (row, media) in rows.media.iter().enumerate() {
                    let text = text.narrow(0, row, 1)?;
                    let (xs, media_len) = match media {
                        Some(media) => {
                            let embeds = match media.embeds.get() {
                                Some(embeds) => embeds.to_device(&self.device)?,
                                None => {
                                    let embeds = self.media_embeds(media, text.dtype())?;
                                    // the request drops it off the engine thread, where a device free has no context
                                    let _ = media.embeds.set(embeds.to_device(&Device::Cpu)?);
                                    embeds
                                }
                            };
                            let embeds = embeds.unsqueeze(0)?;
                            let media_len = embeds.dim(1)?;
                            // llama.cpp cuts the text to the room the media leave; the API cuts it before it knows
                            if media_len + text.dim(1)? > self.max_positions {
                                inference_tensor::bail!(
                                    "the media take {media_len} of the {} positions, send fewer images",
                                    self.max_positions
                                );
                            }
                            (Tensor::cat(&[&embeds, &text], 1)?, media_len)
                        }
                        None => (text, 0),
                    };
                    let xs = self.trunk(xs, media_len)?;
                    let xs = self.head(&xs, &rows.kinds[row..=row], media_len)?;
                    let shift = u32::try_from(media_len)
                        .map_err(|e| inference_tensor::Error::Msg(e.to_string()))?;
                    let one = DecisionRows {
                        kinds: vec![rows.kinds[row]],
                        markers: vec![rows.markers[row].iter().map(|m| m + shift).collect()],
                        media: vec![None],
                    };
                    picked.push(modern_bert_decision::Model::markers(&xs, &one)?);
                }
                let most = picked
                    .iter()
                    .map(|p| p.dim(1))
                    .collect::<Result<Vec<_>>>()?;
                let most = most.into_iter().max().unwrap_or(0);
                let picked = picked
                    .into_iter()
                    .map(|p| {
                        let pad = most - p.dim(1)?;
                        match pad {
                            0 => Ok(p),
                            pad => Tensor::cat(
                                &[
                                    &p,
                                    &Tensor::zeros((1, pad, p.dim(2)?), p.dtype(), p.device())?,
                                ],
                                1,
                            ),
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
                let picked = Tensor::cat(&picked, 0)?;
                let hidden = self
                    .scorer_hidden
                    .forward(&self.scorer_norm.forward(&picked)?)?
                    .gelu_erf()?;
                return self
                    .scorer_out
                    .forward(&hidden)?
                    .squeeze(D::Minus1)?
                    .to_dtype(DType::F32);
            }
        };
        let picked = modern_bert_decision::Model::markers(&xs, rows)?;
        let hidden = self
            .scorer_hidden
            .forward(&self.scorer_norm.forward(&picked)?)?
            .gelu_erf()?;
        self.scorer_out
            .forward(&hidden)?
            .squeeze(D::Minus1)?
            .to_dtype(DType::F32)
    }
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AnyMoeBaseModelMixin for Model {}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::decision::DecisionRows;

    const WIDTH: usize = 8;
    const HEADS: usize = 2;
    const VOCAB: usize = 20;

    // a short-convolution block, an attention block (one KV head), then one head block
    fn config() -> Config {
        Config {
            vocab_size: VOCAB,
            hidden_size: WIDTH,
            num_attention_heads: HEADS,
            num_key_value_heads: vec![0, 1, HEADS],
            intermediate_sizes: vec![16, 16, 4 * WIDTH],
            head_layers: 1,
            rope_theta: 10_000.0,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            layer_norm_eps: 1e-5,
            vision: None,
            audio: None,
        }
    }

    fn weights(cfg: &Config) -> HashMap<String, Tensor> {
        let head_dim = WIDTH / HEADS;
        let mut shapes: Vec<(String, Vec<usize>)> = vec![
            ("token_embd.weight".into(), vec![VOCAB, WIDTH]),
            ("token_embd_norm.weight".into(), vec![WIDTH]),
            ("token_types.weight".into(), vec![3, WIDTH]),
            ("cls.norm.weight".into(), vec![WIDTH]),
            ("cls.norm.bias".into(), vec![WIDTH]),
            ("cls.weight".into(), vec![WIDTH, WIDTH]),
            ("cls.bias".into(), vec![WIDTH]),
            ("cls.output.weight".into(), vec![WIDTH]),
            ("cls.output.bias".into(), vec![1]),
        ];
        let mut add = |name: String, shape: Vec<usize>| shapes.push((name, shape));
        for (i, &kv) in cfg.num_key_value_heads.iter().enumerate() {
            let b = format!("blk.{i}");
            let ff = cfg.intermediate_sizes[i];
            if i >= cfg.trunk_layers() {
                for norm in ["attn_norm", "ffn_norm"] {
                    add(format!("{b}.{norm}.weight"), vec![WIDTH]);
                    add(format!("{b}.{norm}.bias"), vec![WIDTH]);
                }
                for (name, shape) in [
                    ("attn_qkv", [3 * WIDTH, WIDTH]),
                    ("attn_output", [WIDTH, WIDTH]),
                    ("ffn_up", [ff, WIDTH]),
                    ("ffn_down", [WIDTH, ff]),
                ] {
                    add(format!("{b}.{name}.weight"), shape.to_vec());
                    add(format!("{b}.{name}.bias"), vec![shape[0]]);
                }
                continue;
            }
            add(format!("{b}.attn_norm.weight"), vec![WIDTH]);
            add(format!("{b}.ffn_norm.weight"), vec![WIDTH]);
            add(format!("{b}.ffn_gate.weight"), vec![ff, WIDTH]);
            add(format!("{b}.ffn_up.weight"), vec![ff, WIDTH]);
            add(format!("{b}.ffn_down.weight"), vec![WIDTH, ff]);
            match kv {
                0 => {
                    add(format!("{b}.shortconv.conv.weight"), vec![WIDTH, CONV_TAPS]);
                    add(
                        format!("{b}.shortconv.in_proj.weight"),
                        vec![3 * WIDTH, WIDTH],
                    );
                    add(format!("{b}.shortconv.out_proj.weight"), vec![WIDTH, WIDTH]);
                }
                kv => {
                    add(format!("{b}.attn_q.weight"), vec![WIDTH, WIDTH]);
                    add(format!("{b}.attn_k.weight"), vec![kv * head_dim, WIDTH]);
                    add(format!("{b}.attn_v.weight"), vec![kv * head_dim, WIDTH]);
                    add(format!("{b}.attn_output.weight"), vec![WIDTH, WIDTH]);
                    add(format!("{b}.attn_q_norm.weight"), vec![head_dim]);
                    add(format!("{b}.attn_k_norm.weight"), vec![head_dim]);
                }
            }
        }
        shapes
            .into_iter()
            .enumerate()
            .map(|(seed, (name, shape))| {
                let n = shape.iter().product::<usize>();
                let values = (0..n)
                    .map(|i| ((i * 7 + seed * 13) as f32 * 0.37).sin() * 0.5)
                    .collect::<Vec<_>>();
                (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
            })
            .collect()
    }

    fn model() -> Model {
        let cfg = config();
        let vb = inference_quant::ShardedSafeTensors::wrap(weights(&cfg), DType::F32, Device::Cpu);
        let metadata = NormalLoadingMetadata {
            mapper: Box::new(crate::device_map::DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        };
        Model::new(&cfg, vb, metadata).unwrap()
    }

    fn score(model: &Model, tokens: &[u32]) -> f32 {
        let extras = crate::embedding::EmbeddingExtras {
            decision: Some(DecisionRows {
                kinds: vec![0],
                markers: vec![vec![1]],
                media: vec![None],
            }),
            ..Default::default()
        };
        let ids = Tensor::new(tokens, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        model
            .forward(&ids, &extras, &FlashParams::empty(false))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0]
    }

    // the trunk reads both ways: a marker's score moves with a token after it, which a causal trunk never sees
    #[test]
    fn a_marker_reads_the_tokens_after_it() {
        let model = model();
        let base = score(&model, &[3, 4, 5, 6, 7]);
        assert!(base.is_finite());
        assert_ne!(score(&model, &[3, 4, 5, 6, 9]), base);
        // and the same prompt scores the same
        assert_eq!(score(&model, &[3, 4, 5, 6, 7]), base);
    }

    // media ahead of the prompt never read the text, through either mixer, while the text reads them
    #[test]
    fn media_rows_do_not_read_the_text() {
        let model = model();
        let media = Tensor::from_vec(
            (0..2 * WIDTH).map(|i| (i as f32 * 0.21).cos()).collect(),
            (1, 2, WIDTH),
            &Device::Cpu,
        )
        .unwrap();
        let trunk = |tokens: &[u32], media: &Tensor| {
            let ids = Tensor::new(tokens, &Device::Cpu)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let text = model.embed_tokens.forward(&ids).unwrap();
            let xs = Tensor::cat(&[media, &text], 1).unwrap();
            let out = model.trunk(xs, 2).unwrap();
            let rows = |start, len| out.narrow(1, start, len).unwrap().flatten_all().unwrap();
            (
                rows(0, 2).to_vec1::<f32>().unwrap(),
                rows(2, tokens.len()).to_vec1::<f32>().unwrap(),
            )
        };
        let (media_rows, text_rows) = trunk(&[3, 4, 5], &media);
        let (other_media_rows, _) = trunk(&[9, 8, 7], &media);
        assert_eq!(media_rows, other_media_rows);
        let (_, other_text_rows) = trunk(&[3, 4, 5], &(&media * 2.0).unwrap());
        assert_ne!(text_rows, other_text_rows);
    }
}
