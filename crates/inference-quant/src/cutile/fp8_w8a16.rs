//! Retained-weight E4M3 W8A16 GEMM for tensor, channel, and 128x128 block scales.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cutile::core::f8e4m3fn;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::{bf16, f16};
use inference_tensor::{
    CudaDevice, CudaStorage, DType, Device, DeviceLocation, Result, Shape, Storage, Tensor,
};

use super::gemm_tile::{GemmTileConfig, tile_space};
use super::tune::{
    Bucket, Prepared, Space, TUNE_WEIGHT_SETS, TuneMode, TuneRequest, TunedTable,
    buckets_from_breakpoints, cutile_error, tune,
};
use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, jit_available};
use crate::Fp8WeightScaleLayout;
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const BLOCK_SIZE: usize = 128;
const TUNE_KERNEL: &str = "fp8_w8a16";
const ROW_BREAKPOINTS: [usize; 3] = [16, 64, 256];
const PREFILL_PROBE_ROWS: usize = 1024;

#[cutile::module]
mod kernels {
    #![allow(deprecated)]
    use cutile::core::*;

    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_w8a16_post<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<E, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<E, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let k = num_tiles(&px, 1);
        let pws = ws.partition(const_shape![BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<E, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<E, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<E, { [BK, BN] }> = permute(wt, transpose);
                acc = mmaf(xt, wt, acc);
            }
            let scale: Tile<f32, { [BN] }> = pws.load([bid_n]);
            let scale: Tile<f32, { [BM, BN] }> = scale
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let out: Tile<E, { [BM, BN] }> = convert_tile(acc * scale);
            y.store(out, out_idx);
        }
    }

    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_w8a16_block<
        E: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<E, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<E, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let pws = ws.partition(const_shape![1, 1]);
        let k = num_tiles(&px, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<E, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<E, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<E, { [BK, BN] }> = permute(wt, transpose);
                let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let part: Tile<f32, { [BM, BN] }> = mmaf(xt, wt, zero);
                let scale: Tile<f32, { [1, 1] }> = pws.load([bid_n, kg]);
                acc = acc + part * scale.broadcast(const_shape![BM, BN]);
            }
            let out: Tile<E, { [BM, BN] }> = convert_tile(acc);
            y.store(out, out_idx);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GemmShape {
    device: DeviceLocation,
    n: usize,
    k: usize,
    dtype: ActivationDType,
    scales: Fp8WeightScaleLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ActivationDType {
    Bf16,
    F16,
}

impl TryFrom<DType> for ActivationDType {
    type Error = inference_tensor::Error;

    fn try_from(dtype: DType) -> Result<Self> {
        match dtype {
            DType::BF16 => Ok(Self::Bf16),
            DType::F16 => Ok(Self::F16),
            dtype => inference_tensor::bail!("cuTile W8A16 does not support {dtype:?} activations"),
        }
    }
}

const POLICY_SMALL: GemmTileConfig = GemmTileConfig::policy(16, 1, 8);

const POLICY_LARGE: GemmTileConfig = GemmTileConfig::policy(64, 4, 1);

static TUNED: TunedTable<GemmShape, GemmTileConfig> = TunedTable::new();

fn policy(rows: usize) -> GemmTileConfig {
    if rows <= ROW_BREAKPOINTS[0] {
        POLICY_SMALL
    } else {
        POLICY_LARGE
    }
}

fn gemm_config(shape: GemmShape, rows: usize) -> GemmTileConfig {
    TUNED.get(shape, rows).unwrap_or_else(|| policy(rows))
}

fn gemm_space(bucket: Bucket) -> Space {
    tile_space(
        [[16], [32], [64], [128]],
        [[1, 8], [2, 4], [4, 1], [8, 1], [1, 1]],
        policy(bucket.probe),
    )
}

fn gemm_buckets() -> Vec<Bucket> {
    buckets_from_breakpoints(&ROW_BREAKPOINTS, PREFILL_PROBE_ROWS, PREFILL_PROBE_ROWS)
}

pub struct Fp8W8A16Kernel;

pub(super) static FP8_W8A16: Fp8W8A16Kernel = Fp8W8A16Kernel;

#[derive(Clone)]
struct GemmWeights {
    weight: Tensor,
    scales: Tensor,
    key: GemmShape,
}

static SHAPES: OnceLock<Mutex<Vec<Vec<GemmWeights>>>> = OnceLock::new();

pub fn register_fp8_w8a16_shape(
    weight: &Tensor,
    weight_scales: &Tensor,
    scale_layout: Fp8WeightScaleLayout,
    activation_dtype: DType,
) {
    let Ok((n, k)) = weight.dims2() else {
        return;
    };
    let Ok(dtype) = ActivationDType::try_from(activation_dtype) else {
        return;
    };
    let entry = GemmWeights {
        weight: weight.clone(),
        scales: weight_scales.clone(),
        key: GemmShape {
            device: weight.device().location(),
            n,
            k,
            dtype,
            scales: scale_layout,
        },
    };
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if let Some(sets) = shapes.iter_mut().find(|sets| {
        sets[0].key == entry.key && sets[0].weight.device().same_device(entry.weight.device())
    }) {
        if sets.len() < TUNE_WEIGHT_SETS {
            sets.push(entry);
            super::warmup::mark_dirty();
        }
        return;
    }
    shapes.push(vec![entry]);
    super::warmup::mark_dirty();
}

pub fn fp8_w8a16_supported(
    dev: &CudaDevice,
    output_features: usize,
    input_features: usize,
    activation_dtype: DType,
) -> bool {
    jit_available(dev)
        && matches!(activation_dtype, DType::BF16 | DType::F16)
        && output_features > 0
        && input_features > 0
        && output_features.is_multiple_of(BLOCK_SIZE)
        && input_features.is_multiple_of(BLOCK_SIZE)
}

struct GemmOperands<'a> {
    activation: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
    scale_layout: Fp8WeightScaleLayout,
}

pub fn cutile_fp8_w8a16(
    activation: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    scale_layout: Fp8WeightScaleLayout,
) -> Result<Tensor> {
    let operands = GemmOperands {
        activation,
        weight,
        weight_scales,
        scale_layout,
    };
    let (n, k) = weight.dims2()?;
    let key = GemmShape {
        device: activation.device().location(),
        n,
        k,
        dtype: activation.dtype().try_into()?,
        scales: scale_layout,
    };
    let cfg = gemm_config(key, activation.dim(0)?);
    launch(&operands, cfg, false)
}

fn validate_scale_shape(
    scales: &Tensor,
    layout: Fp8WeightScaleLayout,
    n: usize,
    k: usize,
) -> Result<()> {
    let valid = match layout {
        Fp8WeightScaleLayout::Tensor => scales.elem_count() == 1,
        Fp8WeightScaleLayout::Channel => scales.elem_count() == n,
        Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => {
            scales.dims() == [n / BLOCK_SIZE, k / BLOCK_SIZE]
        }
        Fp8WeightScaleLayout::Block(_) => false,
    };
    if !valid {
        inference_tensor::bail!(
            "cuTile W8A16 scale shape {:?} does not match {layout:?} for weight [{n}, {k}]",
            scales.dims()
        )
    }
    Ok(())
}

fn launch(operands: &GemmOperands<'_>, cfg: GemmTileConfig, compile_only: bool) -> Result<Tensor> {
    let activation = operands.activation.contiguous()?;
    let weight = operands.weight.contiguous()?;
    let scales = operands.weight_scales.contiguous()?;
    let (rows, k) = activation.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if rows == 0
        || n == 0
        || k == 0
        || weight_k != k
        || !n.is_multiple_of(BLOCK_SIZE)
        || !k.is_multiple_of(BLOCK_SIZE)
    {
        inference_tensor::bail!("cuTile W8A16 got unsupported shape rows={rows} n={n} k={k}")
    }
    if !matches!(activation.dtype(), DType::BF16 | DType::F16)
        || weight.dtype() != DType::F8E4M3
        || scales.dtype() != DType::F32
    {
        inference_tensor::bail!("cuTile W8A16 needs A16 activations, E4M3 weights, and F32 scales")
    }
    if !activation.device().same_device(weight.device())
        || !activation.device().same_device(scales.device())
    {
        inference_tensor::bail!("cuTile W8A16 operands must be on the same device")
    }
    let Device::Cuda(dev) = activation.device() else {
        inference_tensor::bail!("cuTile W8A16 operands must be CUDA tensors")
    };
    validate_scale_shape(&scales, operands.scale_layout, n, k)?;
    let bm = usize::try_from(cfg.bm).unwrap_or(0);
    if bm == 0 || !BLOCK_SIZE.is_multiple_of(bm) {
        inference_tensor::bail!("cuTile W8A16 row tile {} must divide {BLOCK_SIZE}", cfg.bm)
    }
    if cfg.map_m <= 0 || cfg.map_n <= 0 {
        inference_tensor::bail!("cuTile W8A16 map dimensions must be positive")
    }
    let padded_rows = rows.div_ceil(bm) * bm;
    let padded_activation;
    let activation = if padded_rows == rows {
        &activation
    } else {
        padded_activation =
            Tensor::zeros((padded_rows, k), activation.dtype(), activation.device())?;
        padded_activation.slice_set(&activation, 0, 0)?;
        &padded_activation
    };

    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let (a_storage, a_layout) = activation.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (s_storage, s_layout) = scales.storage_and_layout();
    let (Storage::Cuda(a_cuda), Storage::Cuda(w_cuda), Storage::Cuda(s_cuda)) =
        (&*a_storage, &*w_storage, &*s_storage)
    else {
        inference_tensor::bail!("cuTile W8A16 operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<F8E4M3>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<f32>()?,
        s_layout.start_offset(),
        &stream,
    );
    let dims = |r: usize, c: usize| (vec![r as i32, c as i32], vec![c as i32, 1]);
    let (shape, strides) = dims(n, k);
    let w = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let post_scales = || unsafe {
        let stride = usize::from(operands.scale_layout == Fp8WeightScaleLayout::Channel);
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![n as i32],
            vec![stride as i32],
        )
    };
    let block_scales = || unsafe {
        let (shape, strides) = dims(n / BLOCK_SIZE, k / BLOCK_SIZE);
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let tiles = (padded_rows / bm) * (n / BLOCK_SIZE);
    let blocks_per_sm = usize::try_from(cfg.blocks_per_sm).unwrap_or(1).max(1);
    let tile_blocks = (blocks_per_sm * dev.sm_count()).clamp(1, tiles) as u32;
    let generics = vec![
        super::element_type(activation.dtype()).to_string(),
        cfg.bm.to_string(),
        BLOCK_SIZE.to_string(),
        BLOCK_SIZE.to_string(),
        cfg.map_m.to_string(),
        cfg.map_n.to_string(),
        cfg.latency.to_string(),
    ];
    let cutile_stream = context::stream(dev);

    macro_rules! run {
        ($launcher:expr, $label:literal) => {{
            let launcher = $launcher
                .generics(generics.clone())
                .compile_options(cfg.compile_options());
            if compile_only {
                catch_cutile_panic("W8A16 compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        inference_tensor::Error::Msg(format!(
                            "cuTile {} compile failed: {error:?}",
                            $label
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("W8A16 launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        inference_tensor::Error::Msg(format!(
                            "cuTile {} launch failed: {error:?}",
                            $label
                        ))
                    })
                })?;
            }
        }};
    }

    let output = match activation.dtype() {
        DType::BF16 => {
            let (a_addr, _a_guard) = slice_ptr_on_stream(
                a_cuda.as_cuda_slice::<bf16>()?,
                a_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<bf16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, k);
            let x = unsafe {
                cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
                    a_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let (shape, strides) = dims(padded_rows, n);
            let y = unsafe {
                cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let mapped = y
                .partition([bm, BLOCK_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            match operands.scale_layout {
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => run!(
                    kernels::fp8_w8a16_post(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(post_scales())
                    ),
                    "W8A16 BF16 post-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => run!(
                    kernels::fp8_w8a16_block(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(block_scales())
                    ),
                    "W8A16 BF16 block-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            }
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[padded_rows, n]),
            ))
        }
        DType::F16 => {
            let (a_addr, _a_guard) = slice_ptr_on_stream(
                a_cuda.as_cuda_slice::<f16>()?,
                a_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<f16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, k);
            let x = unsafe {
                cutile::tensor::Tensor::<f16>::borrow_raw_parts(
                    a_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let (shape, strides) = dims(padded_rows, n);
            let y = unsafe {
                cutile::tensor::Tensor::<f16>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let mapped = y
                .partition([bm, BLOCK_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            match operands.scale_layout {
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => run!(
                    kernels::fp8_w8a16_post(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(post_scales())
                    ),
                    "W8A16 F16 post-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => run!(
                    kernels::fp8_w8a16_block(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(block_scales())
                    ),
                    "W8A16 F16 block-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            }
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[padded_rows, n]),
            ))
        }
        _ => unreachable!(),
    };
    output.narrow(0, 0, rows)
}

struct GemmTuner {
    dev: CudaDevice,
    sets: Vec<GemmWeights>,
    operands: HashMap<usize, Tensor>,
}

impl GemmTuner {
    fn new(dev: &CudaDevice, sets: &[GemmWeights]) -> Self {
        Self {
            dev: dev.clone(),
            sets: sets.to_vec(),
            operands: HashMap::new(),
        }
    }

    fn prepare(&mut self, rows: usize, cfg: GemmTileConfig) -> Result<Prepared> {
        let dev = self.dev.clone();
        let sets = self.sets.clone();
        let key = sets[0].key;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.operands.entry(rows) {
            let device = inference_tensor::Device::Cuda(dev.clone());
            let activation =
                Tensor::rand(-1f32, 1f32, (rows, key.k), &device)?.to_dtype(match key.dtype {
                    ActivationDType::Bf16 => DType::BF16,
                    ActivationDType::F16 => DType::F16,
                })?;
            slot.insert(activation);
        }
        let activation = self.operands[&rows].clone();
        let launch = move |weights: &GemmWeights, compile_only: bool| -> Result<Tensor> {
            let operands = GemmOperands {
                activation: &activation,
                weight: &weights.weight,
                weight_scales: &weights.scales,
                scale_layout: key.scales,
            };
            launch(&operands, cfg, compile_only)
        };
        launch(&sets[0], true)?;
        let sample = launch(&sets[0], false)?;
        let mut next = 0usize;
        let run = Box::new(move |_: &Arc<cutile::cuda_core::Stream>| {
            let weights = &sets[next % sets.len()];
            next += 1;
            launch(weights, false).map(|_| ()).map_err(cutile_error)
        });
        Ok(Prepared { run, sample })
    }
}

impl CutileKernel for Fp8W8A16Kernel {
    fn warm(&self, _dev: &CudaDevice) -> Result<()> {
        let shapes = std::mem::take(
            &mut *SHAPES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap(),
        );
        if shapes.is_empty() {
            return Ok(());
        }
        let mode = TuneMode::from_env();
        let buckets = gemm_buckets();
        for sets in &shapes {
            let key = sets[0].key;
            let Device::Cuda(dev) = sets[0].weight.device() else {
                continue;
            };
            let request = TuneRequest {
                kernel: TUNE_KERNEL,
                source_hash: kernels::_SOURCE_HASH,
                shape: format!("n{}_k{}_a{:?}_s{:?}", key.n, key.k, key.dtype, key.scales),
                buckets: &buckets,
                space: &gemm_space,
            };
            let mut tuner = GemmTuner::new(dev, sets);
            let tuned = tune(dev, mode, &request, |rows, candidate| {
                let cfg = GemmTileConfig::from_config(candidate)
                    .ok_or_else(|| inference_tensor::Error::msg("config outside the space"))?;
                tuner.prepare(rows, cfg)
            });
            TUNED.set(key, &tuned, GemmTileConfig::from_config);
        }
        tracing::info!("Warming {} cuTile W8A16 GEMM kernels.", shapes.len());
        for sets in &shapes {
            let key = sets[0].key;
            let Device::Cuda(dev) = sets[0].weight.device() else {
                continue;
            };
            let device = Device::Cuda(dev.clone());
            for bucket in &buckets {
                let result = (|| -> Result<()> {
                    let dtype = match key.dtype {
                        ActivationDType::Bf16 => DType::BF16,
                        ActivationDType::F16 => DType::F16,
                    };
                    let activation = Tensor::zeros((bucket.probe, key.k), dtype, &device)?;
                    let operands = GemmOperands {
                        activation: &activation,
                        weight: &sets[0].weight,
                        weight_scales: &sets[0].scales,
                        scale_layout: key.scales,
                    };
                    launch(&operands, gemm_config(key, bucket.probe), true)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    tracing::warn!(
                        "cuTile W8A16 warmup failed (n={} k={} rows={}): {error}",
                        key.n,
                        key.k,
                        bucket.probe
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{cutile_fp8_w8a16, validate_scale_shape};
    use crate::Fp8WeightScaleLayout;
    use float8::F8E4M3;
    use inference_tensor::{DType, Device, Result, Tensor};

    #[test]
    fn kernels_compile_to_tile_ir() {
        use super::{BLOCK_SIZE, POLICY_LARGE, POLICY_SMALL, kernels};
        for (scales, dtype) in [
            ("post", "bf16"),
            ("block", "bf16"),
            ("post", "f16"),
            ("block", "f16"),
        ] {
            for cfg in [POLICY_SMALL, POLICY_LARGE] {
                let values = super::super::generics(&[
                    &dtype,
                    &cfg.bm,
                    &BLOCK_SIZE,
                    &BLOCK_SIZE,
                    &cfg.map_m,
                    &cfg.map_n,
                    &cfg.latency,
                ]);
                let entry = format!("fp8_w8a16_{scales}");
                let ws_rank = if scales == "post" { 1 } else { 2 };
                let tensors = [("y", 2), ("x", 2), ("w", 2), ("ws", ws_rank)];
                super::super::compile_tile_ir(
                    kernels::__module_ast_self,
                    "kernels",
                    &entry,
                    values,
                    &tensors,
                );
            }
        }
    }

    type ScaleAt = Box<dyn Fn(usize, usize) -> f32>;

    #[test]
    fn scale_shapes_are_validated_by_layout() -> Result<()> {
        let device = Device::Cpu;
        let scalar = Tensor::zeros((), DType::F32, &device)?;
        let channel = Tensor::zeros(256, DType::F32, &device)?;
        let block = Tensor::zeros((2, 4), DType::F32, &device)?;
        validate_scale_shape(&scalar, Fp8WeightScaleLayout::Tensor, 256, 512)?;
        validate_scale_shape(&channel, Fp8WeightScaleLayout::Channel, 256, 512)?;
        validate_scale_shape(&block, Fp8WeightScaleLayout::Block([128, 128]), 256, 512)?;
        assert!(
            validate_scale_shape(&channel, Fp8WeightScaleLayout::Block([128, 128]), 256, 512)
                .is_err()
        );
        Ok(())
    }

    fn value(index: usize, modulus: usize, offset: i32) -> f32 {
        ((index * 17 + 11) % modulus) as f32 + offset as f32
    }

    fn run_correctness(dtype: DType, scale_layout: Fp8WeightScaleLayout) -> Result<()> {
        const ROWS: usize = 7;
        const N: usize = 256;
        const K: usize = 256;

        let device = Device::new_cuda(0)?;
        let Device::Cuda(_cuda) = &device else {
            unreachable!()
        };
        let x_values = (0..ROWS * K)
            .map(|index| value(index, 9, -4) * 0.25)
            .collect::<Vec<_>>();
        let q_values = (0..N * K)
            .map(|index| F8E4M3::from_f32(value(index, 7, -3)))
            .collect::<Vec<_>>();
        let (scales, scale_at): (Vec<f32>, ScaleAt) = match scale_layout {
            Fp8WeightScaleLayout::Tensor => (vec![0.125], Box::new(|_, _| 0.125)),
            Fp8WeightScaleLayout::Channel => {
                let scales = (0..N)
                    .map(|row| 0.0625 * (row % 4 + 1) as f32)
                    .collect::<Vec<_>>();
                let copy = scales.clone();
                (scales, Box::new(move |row, _| copy[row]))
            }
            Fp8WeightScaleLayout::Block([128, 128]) => (
                vec![0.125, 0.25, 0.375, 0.5],
                Box::new(|row, col| [0.125, 0.25, 0.375, 0.5][(row / 128) * 2 + col / 128]),
            ),
            Fp8WeightScaleLayout::Block(_) => unreachable!(),
        };
        let scale_shape = match scale_layout {
            Fp8WeightScaleLayout::Tensor => inference_tensor::Shape::from_dims(&[]),
            Fp8WeightScaleLayout::Channel => inference_tensor::Shape::from_dims(&[N]),
            Fp8WeightScaleLayout::Block(_) => inference_tensor::Shape::from_dims(&[2, 2]),
        };
        let x = Tensor::from_vec(x_values.clone(), (ROWS, K), &device)?.to_dtype(dtype)?;
        let weight = Tensor::from_vec(q_values.clone(), (N, K), &device)?;
        let scales = Tensor::from_vec(scales, scale_shape, &device)?;
        let output = cutile_fp8_w8a16(&x, &weight, &scales, scale_layout)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let mut max_error = 0f32;
        let mut max_reference = 0f32;
        for row in 0..ROWS {
            for out in 0..N {
                let reference = (0..K)
                    .map(|col| {
                        x_values[row * K + col]
                            * q_values[out * K + col].to_f32()
                            * scale_at(out, col)
                    })
                    .sum::<f32>();
                max_error = max_error.max((output[row][out] - reference).abs());
                max_reference = max_reference.max(reference.abs());
            }
        }
        assert!(
            max_error <= max_reference * 0.02 + 0.25,
            "dtype={dtype:?} scales={scale_layout:?}: error {max_error}, reference {max_reference}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn retained_fp8_w8a16_matches_reference() -> Result<()> {
        for dtype in [DType::BF16, DType::F16] {
            for layout in [
                Fp8WeightScaleLayout::Tensor,
                Fp8WeightScaleLayout::Channel,
                Fp8WeightScaleLayout::Block([128, 128]),
            ] {
                run_correctness(dtype, layout)?;
            }
        }
        Ok(())
    }
}
