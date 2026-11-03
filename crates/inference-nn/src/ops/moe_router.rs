use super::*;

#[derive(Debug, Clone, Copy)]
pub enum MoeRouterScoreFunction {
    Raw,
    Softmax,
    Sigmoid,
}

#[cfg(feature = "cuda")]
impl MoeRouterScoreFunction {
    const fn as_i32(self) -> i32 {
        match self {
            Self::Raw => 0,
            Self::Softmax => 1,
            Self::Sigmoid => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MoeRouterSelectedWeight {
    Score,
    Softmax,
    Sigmoid,
}

#[cfg(feature = "cuda")]
impl MoeRouterSelectedWeight {
    const fn as_i32(self) -> i32 {
        match self {
            Self::Score => 0,
            Self::Softmax => 1,
            Self::Sigmoid => 2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoeRouterTopKConfig {
    pub top_k: usize,
    pub score_function: MoeRouterScoreFunction,
    pub selected_weight: MoeRouterSelectedWeight,
    pub renormalize: bool,
    pub norm_min: f32,
    pub output_scale: f32,
    pub logit_clip: Option<(f32, f32)>,
}

pub fn moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    #[cfg(feature = "cuda")]
    if let Some(topk) =
        cuda_moe_router_topk_if_supported(logits, config, selection_bias, expert_scale)?
    {
        return Ok(topk);
    }
    if logits.device().is_cpu() {
        return cpu_moe_router_topk(logits, config, selection_bias, expert_scale);
    }
    tensor_moe_router_topk(logits, config, selection_bias, expert_scale)
}

fn tensor_moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    let logits = logits.to_dtype(DType::F32)?;
    let logits = match config.logit_clip {
        Some((min, max)) => logits.clamp(min as f64, max as f64)?,
        None => logits,
    };
    let scores = match config.score_function {
        MoeRouterScoreFunction::Raw => logits.clone(),
        MoeRouterScoreFunction::Softmax => inference_tensor::nn::ops::softmax_last_dim(&logits)?,
        MoeRouterScoreFunction::Sigmoid => inference_tensor::nn::ops::sigmoid(&logits)?,
    };
    let selection_scores = if let Some(selection_bias) = selection_bias {
        scores.broadcast_add(&selection_bias.to_dtype(DType::F32)?)?
    } else {
        scores.clone()
    };
    let indices = selection_scores
        .topk(config.top_k)?
        .indices
        .to_dtype(DType::U32)?;
    let selected_logits = match config.selected_weight {
        MoeRouterSelectedWeight::Score => None,
        MoeRouterSelectedWeight::Softmax | MoeRouterSelectedWeight::Sigmoid => {
            Some(logits.gather(&indices, D::Minus1)?)
        }
    };
    let mut values = match config.selected_weight {
        MoeRouterSelectedWeight::Score => scores.gather(&indices, D::Minus1)?,
        MoeRouterSelectedWeight::Softmax => {
            inference_tensor::nn::ops::softmax_last_dim(selected_logits.as_ref().unwrap())?
        }
        MoeRouterSelectedWeight::Sigmoid => {
            inference_tensor::nn::ops::sigmoid(selected_logits.as_ref().unwrap())?
        }
    };

    if config.renormalize {
        let denominator = values.sum_keepdim(D::Minus1)?;
        let denominator = if config.norm_min > 0.0 {
            let min = Tensor::full(config.norm_min, denominator.shape(), denominator.device())?;
            denominator.broadcast_maximum(&min)?
        } else {
            denominator
        };
        values = values.broadcast_div(&denominator)?;
    }
    if config.output_scale != 1.0 {
        values = (values * config.output_scale as f64)?;
    }
    if let Some(expert_scale) = expert_scale {
        let scales = expert_scale
            .to_dtype(DType::F32)?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape(indices.shape())?;
        values = (values * scales)?;
    }

    Ok(TopKOutput { values, indices })
}

fn f32_values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn softmax_in_place(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    xs.iter_mut().for_each(|x| *x = (*x - max).exp());
    let sum: f32 = xs.iter().sum();
    xs.iter_mut().for_each(|x| *x /= sum);
}

fn sigmoid(x: f32) -> f32 {
    (1.0 + (-x).exp()).recip()
}

// The tensor path's routes in one host pass: a decode step's dozen small tensor ops and the sort's dispatch cost more
// than the routing itself. The stable descending sort breaks ties toward the lower expert, as the tensor path's does.
fn cpu_moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    let experts = logits.dim(D::Minus1)?;
    let k = config.top_k;
    let mut xs = f32_values(logits)?;
    let bias = selection_bias.map(f32_values).transpose()?;
    let expert_scale = expert_scale.map(f32_values).transpose()?;
    let rows = xs.len() / experts;
    let mut indices = Vec::with_capacity(rows * k);
    let mut values = Vec::with_capacity(rows * k);
    let (mut scores, mut selection, mut order) = (
        vec![0f32; experts],
        vec![0f32; experts],
        Vec::with_capacity(experts),
    );
    for row in xs.chunks_exact_mut(experts) {
        if let Some((min, max)) = config.logit_clip {
            row.iter_mut().for_each(|x| *x = x.clamp(min, max));
        }
        scores.copy_from_slice(row);
        match config.score_function {
            MoeRouterScoreFunction::Raw => {}
            MoeRouterScoreFunction::Softmax => softmax_in_place(&mut scores),
            MoeRouterScoreFunction::Sigmoid => scores.iter_mut().for_each(|x| *x = sigmoid(*x)),
        }
        selection.copy_from_slice(&scores);
        if let Some(bias) = &bias {
            selection.iter_mut().zip(bias).for_each(|(s, b)| *s += b);
        }
        order.clear();
        order.extend(0..u32::try_from(experts)?);
        order.sort_by(|&j, &i| {
            selection[i as usize]
                .partial_cmp(&selection[j as usize])
                .unwrap_or(std::cmp::Ordering::Greater)
        });
        let top = &order[..k];
        let mut picked: Vec<f32> = match config.selected_weight {
            MoeRouterSelectedWeight::Score => top.iter().map(|&e| scores[e as usize]).collect(),
            MoeRouterSelectedWeight::Softmax => {
                let mut picked: Vec<f32> = top.iter().map(|&e| row[e as usize]).collect();
                softmax_in_place(&mut picked);
                picked
            }
            MoeRouterSelectedWeight::Sigmoid => {
                top.iter().map(|&e| sigmoid(row[e as usize])).collect()
            }
        };
        if config.renormalize {
            let sum: f32 = picked.iter().sum();
            let denominator = if config.norm_min > 0.0 {
                sum.max(config.norm_min)
            } else {
                sum
            };
            picked.iter_mut().for_each(|v| *v /= denominator);
        }
        if config.output_scale != 1.0 {
            picked.iter_mut().for_each(|v| *v *= config.output_scale);
        }
        if let Some(scale) = &expert_scale {
            picked
                .iter_mut()
                .zip(top)
                .for_each(|(v, &e)| *v *= scale[e as usize]);
        }
        indices.extend_from_slice(top);
        values.extend(picked);
    }
    let mut dims = logits.dims().to_vec();
    *dims.last_mut().expect("router logits have an expert dim") = k;
    Ok(TopKOutput {
        values: Tensor::from_vec(values, dims.as_slice(), logits.device())?,
        indices: Tensor::from_vec(indices, dims.as_slice(), logits.device())?,
    })
}

#[cfg(feature = "cuda")]
const MOE_ROUTER_MAX_POWER_OF_TWO_EXPERTS: usize = 512;

#[cfg(feature = "cuda")]
const MOE_ROUTER_EXTRA_EXPERT_COUNTS: &[usize] = &[576];

#[cfg(feature = "cuda")]
pub fn cuda_moe_router_topk_supports_experts(n_experts: usize) -> bool {
    (n_experts.is_power_of_two() && n_experts <= MOE_ROUTER_MAX_POWER_OF_TWO_EXPERTS)
        || MOE_ROUTER_EXTRA_EXPERT_COUNTS.contains(&n_experts)
}

#[cfg(feature = "cuda")]
pub fn cuda_moe_router_topk_if_supported(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<Option<TopKOutput>> {
    if !logits.device().is_cuda() {
        return Ok(None);
    }
    let n_experts = match logits.dims().last() {
        Some(n_experts) => *n_experts,
        None => return Ok(None),
    };
    if !cuda_moe_router_topk_supports_experts(n_experts) {
        return Ok(None);
    }
    cuda_moe_router_topk(logits, config, selection_bias, expert_scale).map(Some)
}

#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)]
pub fn cuda_moe_router_topk(
    logits: &Tensor,
    config: MoeRouterTopKConfig,
    selection_bias: Option<&Tensor>,
    expert_scale: Option<&Tensor>,
) -> Result<TopKOutput> {
    use inference_tensor::backend::BackendStorage;
    use inference_tensor::cuda_backend::CudaStorageSlice;
    use inference_tensor::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use std::ffi::c_void;

    let logits = logits.contiguous()?;
    let dims = logits.dims();
    let n_experts = *dims
        .last()
        .ok_or_else(|| inference_tensor::Error::Msg("empty dims".to_string()))?;
    if config.top_k == 0 || config.top_k > n_experts {
        inference_tensor::bail!(
            "cuda_moe_router_topk top_k={} must be in [1, {}]",
            config.top_k,
            n_experts
        );
    }
    if !cuda_moe_router_topk_supports_experts(n_experts) {
        inference_tensor::bail!("cuda_moe_router_topk unsupported expert count {n_experts}");
    }

    let selection_bias = selection_bias.map(Tensor::contiguous).transpose()?;
    if let Some(selection_bias) = &selection_bias
        && (selection_bias.dtype() != DType::F32 || selection_bias.elem_count() != n_experts)
    {
        inference_tensor::bail!("cuda_moe_router_topk selection_bias must be F32 [n_experts]");
    }

    let expert_scale = expert_scale.map(Tensor::contiguous).transpose()?;
    if let Some(expert_scale) = &expert_scale
        && (expert_scale.dtype() != DType::F32 || expert_scale.elem_count() != n_experts)
    {
        inference_tensor::bail!("cuda_moe_router_topk expert_scale must be F32 [n_experts]");
    }
    let selection_bias_storage_and_layout = selection_bias.as_ref().map(|t| t.storage_and_layout());
    let expert_scale_storage_and_layout = expert_scale.as_ref().map(|t| t.storage_and_layout());

    let nrows = logits.elem_count() / n_experts;
    let mut out_dims = dims.to_vec();
    *out_dims.last_mut().unwrap() = config.top_k;
    let out_elem_count = nrows * config.top_k;

    let (logits_storage, _logits_layout) = logits.storage_and_layout();
    let logits_storage = match &*logits_storage {
        inference_tensor::Storage::Cuda(s) => s,
        _ => inference_tensor::bail!("cuda_moe_router_topk requires CUDA logits"),
    };

    let dev = logits_storage.device();
    let stream = dev.cuda_stream();
    let stream_raw = stream.cu_stream() as i64;

    let mut weights_dst = unsafe { dev.alloc::<f32>(out_elem_count) }?;
    let mut ids_dst = unsafe { dev.alloc::<u32>(out_elem_count) }?;
    let (weights_ptr, weights_guard) = weights_dst.device_ptr_mut(&stream);
    let (ids_ptr, ids_guard) = ids_dst.device_ptr_mut(&stream);

    let (clip_min, clip_max, clamp_logits) = match config.logit_clip {
        Some((min, max)) => (min, max, true),
        None => (0.0, 0.0, false),
    };

    macro_rules! launch {
        ($variant:ident, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(logits_src) = &logits_storage.slice else {
                inference_tensor::bail!("cuda_moe_router_topk logits dtype mismatch");
            };
            let (logits_ptr, _logits_guard) = logits_src.device_ptr(&stream);

            let (selection_bias_ptr, _selection_bias_guard) = if let Some((storage, _layout)) =
                &selection_bias_storage_and_layout
            {
                let storage = match &**storage {
                    inference_tensor::Storage::Cuda(s) => s,
                    _ => {
                        inference_tensor::bail!("cuda_moe_router_topk requires CUDA selection_bias")
                    }
                };
                let CudaStorageSlice::F32(src) = &storage.slice else {
                    inference_tensor::bail!("cuda_moe_router_topk selection_bias dtype mismatch");
                };
                let (ptr, guard) = src.device_ptr(&stream);
                (ptr as *const c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };

            let (expert_scale_ptr, _expert_scale_guard) = if let Some((storage, _layout)) =
                &expert_scale_storage_and_layout
            {
                let storage = match &**storage {
                    inference_tensor::Storage::Cuda(s) => s,
                    _ => inference_tensor::bail!("cuda_moe_router_topk requires CUDA expert_scale"),
                };
                let CudaStorageSlice::F32(src) = &storage.slice else {
                    inference_tensor::bail!("cuda_moe_router_topk expert_scale dtype mismatch");
                };
                let (ptr, guard) = src.device_ptr(&stream);
                (ptr as *const c_void, Some(guard))
            } else {
                (std::ptr::null(), None)
            };

            unsafe {
                ffi::$ffi_fn(
                    logits_ptr as *const c_void,
                    weights_ptr as *mut c_void,
                    ids_ptr as *mut c_void,
                    selection_bias_ptr,
                    expert_scale_ptr,
                    nrows as i32,
                    n_experts as i32,
                    config.top_k as i32,
                    config.score_function.as_i32(),
                    config.selected_weight.as_i32(),
                    config.renormalize,
                    clamp_logits,
                    clip_min,
                    clip_max,
                    config.norm_min,
                    config.output_scale,
                    stream_raw,
                );
            }
        }};
    }

    match logits.dtype() {
        DType::BF16 => launch!(BF16, moe_router_topk_bf16),
        DType::F16 => launch!(F16, moe_router_topk_f16),
        DType::F32 => launch!(F32, moe_router_topk_f32),
        dt => inference_tensor::bail!("cuda_moe_router_topk unsupported dtype: {:?}", dt),
    }

    drop(weights_guard);
    drop(ids_guard);

    let weights_storage = inference_tensor::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::F32(weights_dst),
        device: dev.clone(),
    };
    let ids_storage = inference_tensor::cuda_backend::CudaStorage {
        slice: CudaStorageSlice::U32(ids_dst),
        device: dev.clone(),
    };

    Ok(TopKOutput {
        values: Tensor::from((
            inference_tensor::Storage::Cuda(weights_storage),
            Shape::from_dims(&out_dims),
        )),
        indices: Tensor::from((
            inference_tensor::Storage::Cuda(ids_storage),
            Shape::from_dims(&out_dims),
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use inference_tensor::Device;

    const EXPERTS: usize = 16;
    const TOKENS: usize = 5;
    const TOLERANCE: f32 = 1e-6;

    // every score and weight function, with and without the bias, clip, renormalization and scales
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn the_cpu_router_routes_as_the_tensor_path_does() -> Result<()> {
        let mut logits: Vec<f32> = (0..TOKENS * EXPERTS)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        // a tie, which both break toward the lower expert
        logits[3] = logits[7];
        let logits = Tensor::from_vec(logits, (1, TOKENS, EXPERTS), &Device::Cpu)?;
        let bias = Tensor::from_vec(
            (0..EXPERTS)
                .map(|e| e as f32 * 0.01 - 0.05)
                .collect::<Vec<_>>(),
            EXPERTS,
            &Device::Cpu,
        )?;
        let scale = (bias.clone() + 1.0)?;
        let scores = [
            MoeRouterScoreFunction::Raw,
            MoeRouterScoreFunction::Softmax,
            MoeRouterScoreFunction::Sigmoid,
        ];
        let weights = [
            MoeRouterSelectedWeight::Score,
            MoeRouterSelectedWeight::Softmax,
            MoeRouterSelectedWeight::Sigmoid,
        ];
        for score_function in scores {
            for selected_weight in weights {
                for (renormalize, clip, extra) in
                    [(true, None, true), (false, Some((-1.5, 1.5)), false)]
                {
                    let config = MoeRouterTopKConfig {
                        top_k: 4,
                        score_function,
                        selected_weight,
                        renormalize,
                        norm_min: 1e-20,
                        output_scale: 2.5,
                        logit_clip: clip,
                    };
                    let (bias, scale) = if extra {
                        (Some(&bias), Some(&scale))
                    } else {
                        (None, None)
                    };
                    let want = tensor_moe_router_topk(&logits, config, bias, scale)?;
                    let got = cpu_moe_router_topk(&logits, config, bias, scale)?;
                    assert_eq!(got.indices.dims(), want.indices.dims());
                    assert_eq!(
                        got.indices.flatten_all()?.to_vec1::<u32>()?,
                        want.indices.flatten_all()?.to_vec1::<u32>()?,
                        "{config:?}"
                    );
                    let got = got.values.flatten_all()?.to_vec1::<f32>()?;
                    let want = want.values.flatten_all()?.to_vec1::<f32>()?;
                    for (g, w) in got.iter().zip(&want) {
                        assert!(
                            (g - w).abs() <= TOLERANCE * w.abs().max(1.0),
                            "{config:?}: {g} vs {w}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
