//! The 12 Hz codec's encoder, a Mimi (transformers' `MimiModel.encode`): a causal SEANet encoder, a causal
//! transformer, a stride-2 downsampler, and split residual vector quantization to the talker's codebooks.

use inference_nn::attention::{FlashParams, SdpaParams};
use inference_nn::layers::masker::{CausalMaskConfig, CausalMasker, NotACache};
use inference_nn::layers::{RotaryEmbedding, Sdpa, apply_rotary_preselected_qk};
use inference_quant::ShardedVarBuilder;
use inference_tensor::nn::{LayerNorm, Linear, Module};
use inference_tensor::{DType, Device, Result, Tensor};

use super::codec::{CausalConv, linear};
use super::config::CodecEncoderConfig;

const CLUSTER_USAGE_EPS: f64 = 1e-5;
// Mimi's 25 Hz transformer frames become the talker's 12.5 Hz ones
const DOWNSAMPLE_STRIDE: usize = 2;

struct ResnetBlock {
    conv1: CausalConv,
    conv2: CausalConv,
}

impl Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.conv1.forward(&xs.elu(1.0)?)?;
        let ys = self.conv2.forward(&ys.elu(1.0)?)?;
        xs + ys
    }
}

// the SEANet layers in order; ELUs sit before each downsampling conv and the last conv
enum Layer {
    Conv(CausalConv),
    Resnet(ResnetBlock),
    EluConv(CausalConv),
}

struct TransformerLayer {
    input_norm: LayerNorm,
    post_norm: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
    attn_scale: Tensor,
    mlp_scale: Tensor,
}

struct Rvq {
    input_proj: Tensor,
    // each codebook's centroids and their squared norms
    codebooks: Vec<(Tensor, Tensor)>,
}

impl Rvq {
    fn new(count: usize, cfg: &CodecEncoderConfig, vb: &ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.vector_quantization_hidden_dimension;
        let codebooks = (0..count)
            .map(|i| {
                let vb = vb.pp("layers").pp(i).pp("codebook");
                let usage = vb.get(cfg.codebook_size, "cluster_usage")?;
                let sum = vb.get((cfg.codebook_size, cfg.codebook_dim), "embed_sum")?;
                let embed = sum.broadcast_div(
                    &usage
                        .clamp(CLUSTER_USAGE_EPS, f64::INFINITY)?
                        .unsqueeze(1)?,
                )?;
                let norms = embed.sqr()?.sum(1)?;
                Ok((embed, norms))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            input_proj: vb
                .get((dim, cfg.hidden_size, 1), "input_proj.weight")?
                .squeeze(2)?,
            codebooks,
        })
    }

    // `(frames, hidden)` to each codebook's nearest centroids, `count` rows of `frames`
    fn encode(&self, xs: &Tensor, count: usize) -> Result<Vec<Vec<u32>>> {
        let mut residual = xs.matmul(&self.input_proj.t()?)?;
        let mut codes = Vec::with_capacity(count);
        for (embed, norms) in self.codebooks.iter().take(count) {
            // argmin |x - e|^2 = argmax 2 x.e - |e|^2
            let dots = (residual.matmul(&embed.t()?)? * 2.0)?;
            let indices = dots.broadcast_sub(&norms.unsqueeze(0)?)?.argmax(1)?;
            residual = (residual - embed.embedding(&indices)?)?;
            codes.push(indices.to_vec1::<u32>()?);
        }
        Ok(codes)
    }
}

pub struct CodecEncoder {
    layers: Vec<Layer>,
    transformer: Vec<TransformerLayer>,
    rotary: RotaryEmbedding,
    heads: usize,
    head_dim: usize,
    downsample: CausalConv,
    semantic: Rvq,
    acoustic: Rvq,
    semantic_count: usize,
    /// Samples per 12.5 Hz frame.
    pub downsample_rate: usize,
}

impl CodecEncoder {
    /// `vb` is the codec checkpoint's `encoder` prefix.
    pub fn new(cfg: &CodecEncoderConfig, vb: &ShardedVarBuilder, device: &Device) -> Result<Self> {
        let evb = vb.pp("encoder").pp("layers");
        let mut layers = Vec::new();
        let mut at = 0;
        let mut next = || {
            let vb = evb.pp(at);
            at += 1;
            vb
        };
        let conv = |in_ch, out_ch, kernel, stride, vb: &ShardedVarBuilder| {
            CausalConv::with(
                in_ch,
                out_ch,
                (kernel, stride, 1, 1),
                true,
                false,
                &vb.pp("conv"),
            )
        };
        let filters = cfg.num_filters;
        layers.push(Layer::Conv(conv(1, filters, cfg.kernel_size, 1, &next())?));
        let mut scale = 1;
        for &ratio in cfg.upsampling_ratios.iter().rev() {
            let dim = scale * filters;
            for j in 0..cfg.num_residual_layers {
                let vb = next().pp("block");
                let hidden = dim / cfg.compress;
                let dilation = cfg.dilation_growth_rate.pow(u32::try_from(j).unwrap_or(0));
                layers.push(Layer::Resnet(ResnetBlock {
                    conv1: CausalConv::with(
                        dim,
                        hidden,
                        (cfg.residual_kernel_size, 1, dilation, 1),
                        true,
                        false,
                        &vb.pp(1).pp("conv"),
                    )?,
                    conv2: conv(hidden, dim, 1, 1, &vb.pp(3))?,
                }));
            }
            next();
            layers.push(Layer::EluConv(conv(
                dim,
                2 * dim,
                2 * ratio,
                ratio,
                &next(),
            )?));
            scale *= 2;
        }
        next();
        layers.push(Layer::EluConv(conv(
            scale * filters,
            cfg.hidden_size,
            cfg.last_kernel_size,
            1,
            &next(),
        )?));

        let hidden = cfg.hidden_size;
        let inner = cfg.num_attention_heads * cfg.head_dim;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let norm = |vb: ShardedVarBuilder| -> Result<LayerNorm> {
            Ok(LayerNorm::new(
                vb.get(hidden, "weight")?,
                vb.get(hidden, "bias")?,
                cfg.norm_eps,
            ))
        };
        let transformer = (0..cfg.num_hidden_layers)
            .map(|i| {
                let vb = vb.pp("encoder_transformer").pp("layers").pp(i);
                let attn = vb.pp("self_attn");
                Ok(TransformerLayer {
                    input_norm: norm(vb.pp("input_layernorm"))?,
                    post_norm: norm(vb.pp("post_attention_layernorm"))?,
                    q: linear(hidden, inner, false, &attn.pp("q_proj"))?,
                    k: linear(hidden, kv, false, &attn.pp("k_proj"))?,
                    v: linear(hidden, kv, false, &attn.pp("v_proj"))?,
                    o: linear(inner, hidden, false, &attn.pp("o_proj"))?,
                    fc1: linear(
                        hidden,
                        cfg.intermediate_size,
                        false,
                        &vb.pp("mlp").pp("fc1"),
                    )?,
                    fc2: linear(
                        cfg.intermediate_size,
                        hidden,
                        false,
                        &vb.pp("mlp").pp("fc2"),
                    )?,
                    attn_scale: vb.pp("self_attn_layer_scale").get(hidden, "scale")?,
                    mlp_scale: vb.pp("mlp_layer_scale").get(hidden, "scale")?,
                })
            })
            .collect::<Result<_>>()?;
        #[allow(clippy::cast_possible_truncation)]
        let theta = cfg.rope_theta as f32;
        let q = vb.pp("quantizer");
        Ok(Self {
            layers,
            transformer,
            rotary: RotaryEmbedding::new(
                theta,
                cfg.head_dim,
                cfg.max_position_embeddings,
                device,
                true,
                vb.dtype(),
            )?,
            heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            downsample: CausalConv::with(
                hidden,
                hidden,
                (2 * DOWNSAMPLE_STRIDE, DOWNSAMPLE_STRIDE, 1, 1),
                false,
                true,
                &vb.pp("downsample").pp("conv"),
            )?,
            semantic: Rvq::new(
                cfg.num_semantic_quantizers,
                cfg,
                &q.pp("semantic_residual_vector_quantizer"),
            )?,
            acoustic: Rvq::new(
                cfg.num_quantizers - cfg.num_semantic_quantizers,
                cfg,
                &q.pp("acoustic_residual_vector_quantizer"),
            )?,
            semantic_count: cfg.num_semantic_quantizers,
            downsample_rate: cfg.upsampling_ratios.iter().product::<usize>() * DOWNSAMPLE_STRIDE,
        })
    }

    // the reference builds a full causal mask here: its `sliding_window` never reaches the attention
    fn transformer(&self, xs: Tensor) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let shape = Tensor::zeros((b, seq), DType::U32, xs.device())?;
        let mask = CausalMasker.make_causal_mask(
            &shape,
            &NotACache,
            xs.dtype(),
            &CausalMaskConfig {
                sliding_window: None,
                force_custom: true,
            },
        )?;
        let (cos, sin) = self.rotary.get_cos_sin()?;
        let (cos, sin) = (cos.narrow(0, 0, seq)?, sin.narrow(0, 0, seq)?);
        #[allow(clippy::cast_precision_loss)]
        let params = SdpaParams {
            n_kv_groups: 1,
            sliding_window: None,
            softcap: None,
            softmax_scale: 1.0 / (self.head_dim as f32).sqrt(),
            sinks: None,
            chunk: None,
        };
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((b, seq, (), self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let mut xs = xs;
        for layer in &self.transformer {
            let h = layer.input_norm.forward(&xs)?;
            let (q, k) = apply_rotary_preselected_qk(
                &heads(layer.q.forward(&h)?)?,
                &heads(layer.k.forward(&h)?)?,
                &cos,
                &sin,
                true,
            )?;
            let v = heads(layer.v.forward(&h)?)?;
            let attn =
                Sdpa.run_attention(&q, &k, &v, &mask, Some(&FlashParams::empty(false)), &params)?;
            let attn = attn
                .transpose(1, 2)?
                .reshape((b, seq, self.heads * self.head_dim))?;
            xs = (xs + layer.o.forward(&attn)?.broadcast_mul(&layer.attn_scale)?)?;
            let h = layer.post_norm.forward(&xs)?;
            let h = layer.fc2.forward(&layer.fc1.forward(&h)?.gelu_erf()?)?;
            xs = (xs + h.broadcast_mul(&layer.mlp_scale)?)?;
        }
        Ok(xs)
    }

    /// Mono 24 kHz samples to `quantizers` codebooks of codes, `(frames, quantizers)` row-major.
    pub fn encode(&self, pcm: &[f32], quantizers: usize, device: &Device) -> Result<Vec<Vec<u32>>> {
        let mut xs = Tensor::from_slice(pcm, (1, 1, pcm.len()), device)?;
        for layer in &self.layers {
            xs = match layer {
                Layer::Conv(conv) => conv.forward(&xs)?,
                Layer::Resnet(block) => block.forward(&xs)?,
                Layer::EluConv(conv) => conv.forward(&xs.elu(1.0)?)?,
            };
        }
        let xs = self.transformer(xs.transpose(1, 2)?)?;
        let xs = self.downsample.forward(&xs.transpose(1, 2)?)?;
        let frames = xs.squeeze(0)?.t()?.contiguous()?;
        let mut books = self.semantic.encode(&frames, self.semantic_count)?;
        books.extend(
            self.acoustic
                .encode(&frames, quantizers - self.semantic_count)?,
        );
        // the reference keeps ceil(samples / downsample_rate) frames
        let keep = pcm.len().div_ceil(self.downsample_rate).min(books[0].len());
        Ok((0..keep)
            .map(|t| books.iter().map(|book| book[t]).collect())
            .collect())
    }
}
