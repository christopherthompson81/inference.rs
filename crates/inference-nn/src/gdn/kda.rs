//! Kimi Delta Attention: GDN's delta rule with a decay per key channel. Its own q, k and v convs run as one over the
//! packed q|k|v, on GDN's conv and its hybrid-pool state; the recurrence and the gated output norm are
//! [`crate::ops::kda`].

use std::sync::Arc;

use inference_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use inference_tensor::{D, DType, Result, Tensor};

use super::config::{GdnDims, GdnVHeadLayout};
use super::{GdnLayerCache, RecurrentBatchKind};

// the convs over q, k and v, packed in this order
const CONVS: [&str; 3] = ["q_conv1d", "k_conv1d", "v_conv1d"];

/// A KDA layer's widths.
#[derive(Clone)]
pub struct KdaConfig {
    pub hidden_size: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub norm_eps: f64,
    pub quantization_config: Option<QuantizedConfig>,
}

impl KdaConfig {
    pub fn width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// The conv channels a sequence's state holds: q, k and v.
    pub fn conv_dim(&self) -> usize {
        3 * self.width()
    }

    fn dims(&self) -> GdnDims {
        let width = self.width();
        GdnDims {
            hidden_size: self.hidden_size,
            num_k_heads: self.heads,
            num_v_heads: self.heads,
            head_k_dim: self.head_dim,
            head_v_dim: self.head_dim,
            conv_kernel_size: self.conv_kernel,
            key_dim: width,
            value_dim: width,
            conv_dim: self.conv_dim(),
            v_per_group: 1,
            v_head_layout: GdnVHeadLayout::Grouped,
        }
    }
}

/// A low-rank projection, `up(down(x))`.
struct LowRank {
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
}

impl LowRank {
    fn load(cfg: &KdaConfig, name: char, vb: &ShardedVarBuilder) -> Result<Self> {
        let q = &cfg.quantization_config;
        Ok(Self {
            down: ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.head_dim,
                q,
                false,
                vb.pp(format!("{name}_a_proj")),
            )?,
            up: ReplicatedLayer::new(
                cfg.head_dim,
                cfg.width(),
                q,
                false,
                vb.pp(format!("{name}_b_proj")),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.up.forward(&self.down.forward(x)?)
    }
}

pub struct KimiDeltaAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    // `(3 width, 1, kernel)`: q's, k's and v's
    conv_weight: Tensor,
    forget: LowRank,
    gate: LowRank,
    b_proj: Arc<dyn QuantMethod>,
    dt_bias: Tensor,
    a_log: Tensor,
    o_norm: Tensor,
    o_proj: Arc<dyn QuantMethod>,
    cfg: KdaConfig,
}

impl KimiDeltaAttention {
    pub fn load(cfg: &KdaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let (h, width) = (cfg.hidden_size, cfg.width());
        let q = &cfg.quantization_config;
        let proj = |name: &str| ReplicatedLayer::new(h, width, q, false, vb.pp(name));
        let convs = CONVS
            .iter()
            .map(|name| {
                vb.pp(name)
                    .get_unchecked("weight")?
                    .reshape((width, 1, cfg.conv_kernel))
            })
            .collect::<Result<Vec<_>>>()?;
        let f32 = |t: Tensor| t.to_dtype(DType::F32);
        Ok(Self {
            q_proj: proj("q_proj")?,
            k_proj: proj("k_proj")?,
            v_proj: proj("v_proj")?,
            conv_weight: Tensor::cat(&convs, 0)?,
            forget: LowRank::load(cfg, 'f', &vb)?,
            gate: LowRank::load(cfg, 'g', &vb)?,
            b_proj: ReplicatedLayer::new(h, cfg.heads, q, false, vb.pp("b_proj"))?,
            dt_bias: f32(vb.get(width, "dt_bias")?)?,
            a_log: f32(vb.get_unchecked("A_log")?.flatten_all()?)?,
            o_norm: f32(vb.pp("o_norm").get(cfg.head_dim, "weight")?)?,
            o_proj: ReplicatedLayer::new(width, h, q, false, vb.pp("o_proj"))?,
            cfg: cfg.clone(),
        })
    }

    /// `x` normed `(b, s, hidden)` through the layer, `cache` the sequences' conv and recurrent state (advanced).
    pub fn forward(
        &self,
        x: &Tensor,
        cache: &mut GdnLayerCache,
        batch_kind: RecurrentBatchKind,
    ) -> Result<Tensor> {
        let width = self.cfg.width();
        let mixed = Tensor::cat(
            &[
                self.q_proj.forward(x)?,
                self.k_proj.forward(x)?,
                self.v_proj.forward(x)?,
            ],
            D::Minus1,
        )?;
        let convolved = super::backend::causal_conv1d(
            &mixed,
            &self.conv_weight,
            &self.cfg.dims(),
            cache,
            batch_kind,
        )?;
        let part = |i: usize| convolved.narrow(D::Minus1, i * width, width);
        let y = crate::ops::kda(
            crate::ops::Kda {
                q: &part(0)?,
                k: &part(1)?,
                v: &part(2)?,
                f: &self.forget.forward(x)?,
                gate: &self.gate.forward(x)?,
                b: &self.b_proj.forward(x)?,
                dt_bias: &self.dt_bias,
                a_log: &self.a_log,
                norm_w: &self.o_norm,
                norm_eps: self.cfg.norm_eps,
                heads: self.cfg.heads,
                head_dim: self.cfg.head_dim,
                slots: cache.slots.as_ref(),
            },
            &cache.recurrent_state,
        )?;
        self.o_proj.forward(&y)
    }

    /// The tensors ISQ leaves alone, under the layer's prefix.
    pub fn add_residual(&self, uvb: &crate::utils::unvarbuilder::UnVarBuilder) {
        let width = self.cfg.width();
        for (i, name) in CONVS.iter().enumerate() {
            if let Ok(conv) = self.conv_weight.narrow(0, i * width, width) {
                uvb.pp(name).add_tensor("weight", conv);
            }
        }
        uvb.add_tensor("dt_bias", self.dt_bias.clone());
        if let Ok(a_log) = self.a_log.reshape((1, 1, self.cfg.heads, 1)) {
            uvb.add_tensor("A_log", a_log);
        }
        uvb.pp("o_norm").add_tensor("weight", self.o_norm.clone());
    }
}
