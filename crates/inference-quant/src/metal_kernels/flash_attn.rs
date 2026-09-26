// Portions of this file are adapted from Apple's MLX framework
// (https://github.com/ml-explore/mlx)
// Licensed under the Apache License 2.0
// Copyright © 2023 Apple Inc.

use super::*;

// Must stay in sync with the tile constants in flash_attn.metal.
pub const FA_NQPSG: usize = 8;
pub const FA_NCPSG: usize = 64;
const FA_NSG: usize = 8;
const FA_HEAD_DIM: usize = 512;

const FC_FLASH_ATTN_EXT_PAD: usize = 100;
const FC_FLASH_ATTN_EXT: usize = 300;

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnPadKargs {
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    _pad0: u32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    _pad1: u32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnBlkKargs {
    ne01: i32,
    ne30: i32,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    _pad0: u32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FlashAttnKargs {
    ne01: i32,
    ne02: i32,
    ne03: i32,
    _pad0: u32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    ns10: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ns20: i32,
    _pad1: u32,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    _pad2: u32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    logit_softcap: f32,
}

pub fn flash_attn_ext_blk_scratch_size(
    q_seq: usize,
    k_seq: usize,
    mask_heads: usize,
    mask_batches: usize,
) -> usize {
    let nblk0 = k_seq.div_ceil(FA_NCPSG);
    let nblk1 = q_seq.div_ceil(FA_NQPSG);
    nblk0 * nblk1 * mask_heads.max(1) * mask_batches.max(1)
}

fn flash_attn_ext_smem_bytes() -> usize {
    let nqptg = FA_NQPSG;
    let dk = FA_HEAD_DIM;
    let dv = FA_HEAD_DIM;
    let dv_pad = (dv + 63) & !63;
    let ncpsg = FA_NCPSG;
    let halves = nqptg * (dk + 2 * dv_pad + 4 * ncpsg);
    let bytes = halves * 2;
    (bytes + 15) & !15
}

#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_ext_bf16_dk512(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q: (&Buffer, usize),
    k: (&Buffer, usize),
    v: (&Buffer, usize),
    mask: (&Buffer, usize),
    out: &Buffer,
    blk_scratch: &Buffer,
    pad_scratch: Option<&Buffer>,
    q_shape: &[usize],
    q_stride_elems: &[usize],
    k_shape: &[usize],
    k_stride_elems: &[usize],
    v_stride_elems: &[usize],
    mask_shape: &[usize],
    mask_stride_elems: &[usize],
    scale: f32,
) -> Result<(), MetalKernelError> {
    assert_eq!(q_shape.len(), 4, "expected q rank 4");
    assert_eq!(k_shape.len(), 4, "expected k rank 4");
    assert_eq!(mask_shape.len(), 4, "expected mask rank 4");
    let (b, n_heads_q, q_seq, dk) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (b_kv, n_heads_kv, k_seq, dk_k) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    assert_eq!(dk, FA_HEAD_DIM, "this dispatcher specializes DK=DV=512");
    assert_eq!(dk_k, FA_HEAD_DIM);
    assert_eq!(b, b_kv);

    let bf16 = 2u64;
    let halfsz = 2u64;
    let q_nb = [
        q_stride_elems[3] as u64 * bf16,
        q_stride_elems[2] as u64 * bf16,
        q_stride_elems[1] as u64 * bf16,
        q_stride_elems[0] as u64 * bf16,
    ];
    let k_nb = [
        k_stride_elems[3] as u64 * bf16,
        k_stride_elems[2] as u64 * bf16,
        k_stride_elems[1] as u64 * bf16,
        k_stride_elems[0] as u64 * bf16,
    ];
    let v_nb = [
        v_stride_elems[3] as u64 * bf16,
        v_stride_elems[2] as u64 * bf16,
        v_stride_elems[1] as u64 * bf16,
        v_stride_elems[0] as u64 * bf16,
    ];
    let m_nb = [
        mask_stride_elems[3] as u64 * halfsz,
        mask_stride_elems[2] as u64 * halfsz,
        mask_stride_elems[1] as u64 * halfsz,
        mask_stride_elems[0] as u64 * halfsz,
    ];

    let has_kvpad = k_seq % FA_NCPSG != 0;
    if has_kvpad {
        assert!(
            pad_scratch.is_some(),
            "K seq {k_seq} requires pad scratch (not multiple of {FA_NCPSG})"
        );
    }

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();

    if has_kvpad {
        let constants =
            ConstantValues::new(vec![(FC_FLASH_ATTN_EXT_PAD, ConstantValue::Bool(true))]);
        let pipeline = kernels.load_pipeline_with_constants(
            device,
            "kernel_flash_attn_ext_pad",
            Some(constants),
        )?;
        let args = FlashAttnPadKargs {
            ne11: k_seq as i32,
            ne_12_2: n_heads_kv as i32,
            ne_12_3: b as i32,
            _pad0: 0,
            nb11: k_nb[1],
            nb12: k_nb[2],
            nb13: k_nb[3],
            nb21: v_nb[1],
            nb22: v_nb[2],
            nb23: v_nb[3],
            ne31: mask_shape[2] as i32,
            ne32: mask_shape[1] as i32,
            ne33: mask_shape[0] as i32,
            _pad1: 0,
            nb31: m_nb[1],
            nb32: m_nb[2],
            nb33: m_nb[3],
        };
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(k.0), k.1);
        encoder.set_input_buffer(2, Some(v.0), v.1);
        encoder.set_input_buffer(3, Some(mask.0), mask.1);
        encoder.set_output_buffer(4, Some(pad_scratch.unwrap()), 0);
        let grid = MTLSize {
            width: FA_NCPSG,
            height: n_heads_kv.max(mask_shape[1]),
            depth: b.max(mask_shape[0]),
        };
        let group = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, group);
    }

    {
        let pipeline = kernels.load_pipeline(device, "kernel_flash_attn_ext_blk")?;
        let args = FlashAttnBlkKargs {
            ne01: q_seq as i32,
            ne30: k_seq as i32,
            ne31: mask_shape[2] as i32,
            ne32: mask_shape[1] as i32,
            ne33: mask_shape[0] as i32,
            _pad0: 0,
            nb31: m_nb[1],
            nb32: m_nb[2],
            nb33: m_nb[3],
        };
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(mask.0), mask.1);
        encoder.set_output_buffer(2, Some(blk_scratch), 0);
        let nblk0 = k_seq.div_ceil(FA_NCPSG);
        let nblk1 = q_seq.div_ceil(FA_NQPSG);
        let grid = MTLSize {
            width: nblk0,
            height: nblk1,
            depth: (mask_shape[1] * mask_shape[0]).max(1),
        };
        let group = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, group);
    }

    {
        let constants = ConstantValues::new(vec![
            (FC_FLASH_ATTN_EXT, ConstantValue::Bool(true)),
            (FC_FLASH_ATTN_EXT + 4, ConstantValue::Bool(has_kvpad)),
        ]);
        let pipeline = kernels.load_pipeline_with_constants(
            device,
            "kernel_flash_attn_ext_bf16_dk512_dv512",
            Some(constants),
        )?;

        let args = FlashAttnKargs {
            ne01: q_seq as i32,
            ne02: n_heads_q as i32,
            ne03: b as i32,
            _pad0: 0,
            nb01: q_nb[1],
            nb02: q_nb[2],
            nb03: q_nb[3],
            ne11: k_seq as i32,
            ne_12_2: n_heads_kv as i32,
            ne_12_3: b as i32,
            ns10: (k_nb[1] / bf16) as i32,
            nb11: k_nb[1],
            nb12: k_nb[2],
            nb13: k_nb[3],
            ns20: (v_nb[1] / bf16) as i32,
            _pad1: 0,
            nb21: v_nb[1],
            nb22: v_nb[2],
            nb23: v_nb[3],
            ne31: mask_shape[2] as i32,
            ne32: mask_shape[1] as i32,
            ne33: mask_shape[0] as i32,
            _pad2: 0,
            nb31: m_nb[1],
            nb32: m_nb[2],
            nb33: m_nb[3],
            ne1: q_seq as i32,
            ne2: n_heads_q as i32,
            ne3: b as i32,
            scale,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
            logit_softcap: 0.0,
        };
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_bytes(0, &args);
        encoder.set_input_buffer(1, Some(q.0), q.1);
        encoder.set_input_buffer(2, Some(k.0), k.1);
        encoder.set_input_buffer(3, Some(v.0), v.1);
        encoder.set_input_buffer(4, Some(mask.0), mask.1);
        // sinks unused; bind mask as a dummy non-null buffer
        encoder.set_input_buffer(5, Some(mask.0), mask.1);
        encoder.set_input_buffer(6, pad_scratch.or(Some(blk_scratch)), 0);
        encoder.set_input_buffer(7, Some(blk_scratch), 0);
        encoder.set_output_buffer(8, Some(out), 0);
        encoder.set_threadgroup_memory_length(0, flash_attn_ext_smem_bytes());

        let grid = MTLSize {
            width: q_seq.div_ceil(FA_NQPSG),
            height: n_heads_q,
            depth: b,
        };
        let group = MTLSize {
            width: 32,
            height: FA_NSG,
            depth: 1,
        };
        encoder.dispatch_thread_groups(grid, group);
    }

    Ok(())
}

/// Decode-specialized flash attention for DK=DV=512 BF16, q_seq=1.
/// Threadgroup memory: DK*2 + NSG*SH*2 + NSG*DV*2 = 1024 + NSG*256 + NSG*1024
/// = 1024 + NSG*1280 bytes. For NSG=4: 6144 bytes (well under 32KB).
#[allow(clippy::too_many_arguments)]
pub fn call_flash_attn_ext_vec_bf16_dk512(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    q: (&Buffer, usize),
    k: (&Buffer, usize),
    v: (&Buffer, usize),
    mask: (&Buffer, usize),
    out: &Buffer,
    q_shape: &[usize],
    q_stride_elems: &[usize],
    k_shape: &[usize],
    k_stride_elems: &[usize],
    v_stride_elems: &[usize],
    mask_shape: &[usize],
    mask_stride_elems: &[usize],
    scale: f32,
) -> Result<(), MetalKernelError> {
    assert_eq!(q_shape.len(), 4, "expected q rank 4");
    assert_eq!(k_shape.len(), 4, "expected k rank 4");
    assert_eq!(mask_shape.len(), 4, "expected mask rank 4");
    let (b, n_heads_q, q_seq, dk) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (b_kv, n_heads_kv, k_seq, dk_k) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
    assert_eq!(dk, FA_HEAD_DIM);
    assert_eq!(dk_k, FA_HEAD_DIM);
    assert_eq!(b, b_kv);

    let bf16 = 2u64;
    let halfsz = 2u64;
    let q_nb = [
        q_stride_elems[3] as u64 * bf16,
        q_stride_elems[2] as u64 * bf16,
        q_stride_elems[1] as u64 * bf16,
        q_stride_elems[0] as u64 * bf16,
    ];
    let k_nb = [
        k_stride_elems[3] as u64 * bf16,
        k_stride_elems[2] as u64 * bf16,
        k_stride_elems[1] as u64 * bf16,
        k_stride_elems[0] as u64 * bf16,
    ];
    let v_nb = [
        v_stride_elems[3] as u64 * bf16,
        v_stride_elems[2] as u64 * bf16,
        v_stride_elems[1] as u64 * bf16,
        v_stride_elems[0] as u64 * bf16,
    ];
    let m_nb = [
        mask_stride_elems[3] as u64 * halfsz,
        mask_stride_elems[2] as u64 * halfsz,
        mask_stride_elems[1] as u64 * halfsz,
        mask_stride_elems[0] as u64 * halfsz,
    ];

    let pipeline = kernels.load_pipeline(device, "kernel_flash_attn_ext_vec_bf16_dk512_dv512")?;

    let args = FlashAttnKargs {
        ne01: q_seq as i32,
        ne02: n_heads_q as i32,
        ne03: b as i32,
        _pad0: 0,
        nb01: q_nb[1],
        nb02: q_nb[2],
        nb03: q_nb[3],
        ne11: k_seq as i32,
        ne_12_2: n_heads_kv as i32,
        ne_12_3: b as i32,
        ns10: (k_nb[1] / bf16) as i32,
        nb11: k_nb[1],
        nb12: k_nb[2],
        nb13: k_nb[3],
        ns20: (v_nb[1] / bf16) as i32,
        _pad1: 0,
        nb21: v_nb[1],
        nb22: v_nb[2],
        nb23: v_nb[3],
        ne31: mask_shape[2] as i32,
        ne32: mask_shape[1] as i32,
        ne33: mask_shape[0] as i32,
        _pad2: 0,
        nb31: m_nb[1],
        nb32: m_nb[2],
        nb33: m_nb[3],
        ne1: q_seq as i32,
        ne2: n_heads_q as i32,
        ne3: b as i32,
        scale,
        max_bias: 0.0,
        m0: 0.0,
        m1: 0.0,
        n_head_log2: 0,
        logit_softcap: 0.0,
    };

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(q.0), q.1);
    encoder.set_input_buffer(2, Some(k.0), k.1);
    encoder.set_input_buffer(3, Some(v.0), v.1);
    encoder.set_input_buffer(4, Some(mask.0), mask.1);
    encoder.set_output_buffer(8, Some(out), 0);

    // Shared mem: DK*2 + NSG*SH*2 + NSG*DV*2 bytes. For NSG=4: 1024 + 1024 + 4096 = 6144.
    const FA_VEC_NSG: usize = 4;
    const FA_VEC_C: usize = 32;
    let smem_bytes =
        FA_HEAD_DIM * 2 + FA_VEC_NSG * (4 * FA_VEC_C) * 2 + FA_VEC_NSG * FA_HEAD_DIM * 2;
    encoder.set_threadgroup_memory_length(0, smem_bytes);

    let grid = MTLSize {
        width: q_seq,
        height: n_heads_q,
        depth: b,
    };
    let group = MTLSize {
        width: 32,
        height: FA_VEC_NSG,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// Fused KV cache append. Writes K and V src tensors into their respective
/// cache slots at `dst_offset` in one dispatch. Both tensors must be
/// contiguous and share shape [b=1, n_kv, src_seq, head_dim]; cache layout
/// is [b=1, n_kv, max_seq, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn call_kv_append_dual(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: DType,
    k_src: &Buffer,
    k_src_offset: usize,
    v_src: &Buffer,
    v_src_offset: usize,
    k_dst: &Buffer,
    v_dst: &Buffer,
    head_dim: usize,
    n_kv: usize,
    src_seq: usize,
    max_seq: usize,
    dst_offset: usize,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        DType::BF16 => "kv_append_dual_bf16",
        DType::F16 => "kv_append_dual_f16",
        DType::F32 => "kv_append_dual_f32",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::BF16, DType::F16, DType::F32],
                got: other,
            })
        }
    };
    let pipeline = kernels.load_pipeline(device, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(k_src), k_src_offset);
    encoder.set_input_buffer(1, Some(v_src), v_src_offset);
    encoder.set_output_buffer(2, Some(k_dst), 0);
    encoder.set_output_buffer(3, Some(v_dst), 0);
    <i32 as EncoderParam>::set_param(encoder, 4, head_dim as i32);
    <i32 as EncoderParam>::set_param(encoder, 5, n_kv as i32);
    <i32 as EncoderParam>::set_param(encoder, 6, src_seq as i32);
    <i32 as EncoderParam>::set_param(encoder, 7, max_seq as i32);
    <i32 as EncoderParam>::set_param(encoder, 8, dst_offset as i32);

    let flat = n_kv * head_dim;
    let group_w = 32usize.min(flat.max(1));
    let group = MTLSize {
        width: group_w,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: flat.div_ceil(group_w),
        height: src_seq,
        depth: 2,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}
