//! Writing EXL3 tensors: exllamav3's trellis quantizer on the GPU, and the packing of its states into tile words.

use inference_tensor::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};

use super::{Exl3Bits, Exl3Codebook, TILE_VALUES, qffi};
use crate::utils::slice_ptr;

// the quantizer's cost and history scratch, at most; a wave that would take more runs in smaller batches
const SCRATCH_BYTES: i64 = 1 << 30;

impl Exl3Bits {
    // the state bits position `p` of a tile adds to the stream
    fn width(self, p: usize) -> usize {
        match self {
            Self::Whole(k) => k,
            // odd positions take the extra bit (exllamav3's mask 0xaaaa)
            Self::Half(ka) => ka + p % 2,
        }
    }

    /// The uint16 words a tile packs into.
    pub fn words(self) -> usize {
        match self {
            Self::Whole(k) => 16 * k,
            Self::Half(ka) => 16 * ka + 8,
        }
    }
}

/// A tile's 256 states (position order, as the quantizer returns them) as its trellis words: each state's new
/// bits, most significant first, into one stream of little-endian u32 words.
pub fn pack_tile(states: &[u16], bits: Exl3Bits) -> Vec<u16> {
    let mut stream = vec![0u32; bits.words() / 2];
    let mut pos = 0;
    for (p, &state) in states.iter().enumerate().take(TILE_VALUES) {
        let width = bits.width(p);
        let value = u32::from(state) & ((1 << width) - 1);
        for b in (0..width).rev() {
            if (value >> b) & 1 == 1 {
                stream[pos / 32] |= 1 << (31 - pos % 32);
            }
            pos += 1;
        }
    }
    stream
        .iter()
        .flat_map(|w| [(*w & 0xffff) as u16, (*w >> 16) as u16])
        .collect()
}

/// `tiles` `(n, 256)` f32 on a CUDA device, each encoded by a Viterbi search over the trellis: the values the
/// encodings decode to `(n, 256)` f32, and each position's 16-bit state `(n, 256)` (i16 holding the u16 bits, for
/// `pack_tile`).
pub fn quantize_tiles(
    tiles: &Tensor,
    bits: Exl3Bits,
    codebook: Exl3Codebook,
) -> Result<(Tensor, Tensor)> {
    let Device::Cuda(dev) = tiles.device() else {
        inference_tensor::bail!("the EXL3 quantizer runs on CUDA");
    };
    qffi::ensure_loaded().map_err(inference_tensor::Error::msg)?;
    let (n, len) = tiles.dims2()?;
    if len != TILE_VALUES {
        inference_tensor::bail!("EXL3 tiles hold {TILE_VALUES} values, not {len}");
    }
    let (k, half_k) = match bits {
        Exl3Bits::Whole(k) => (k, false),
        Exl3Bits::Half(ka) => (ka, true),
    };
    let (mut cost_bytes, mut edge_bytes) = (0i64, 0i64);
    // SAFETY: plain sizes in, two counts out
    let wave = unsafe {
        qffi::exl3_quantize_scratch(
            k as i32,
            codebook as i32,
            half_k,
            &mut cost_bytes,
            &mut edge_bytes,
        )
    };
    if wave < 1 {
        inference_tensor::bail!("no EXL3 quantizer for {bits:?} on {codebook:?}");
    }
    let scratch_tiles = wave
        .min(SCRATCH_BYTES / (cost_bytes + edge_bytes))
        .min(n as i64)
        .max(1);
    let x = tiles.to_dtype(DType::F32)?.contiguous()?;
    // SAFETY: the kernels write every tile's values and states, and fill the scratch before reading it
    let out = unsafe { dev.alloc::<f32>(n * TILE_VALUES)? };
    // SAFETY: as above
    let idx = unsafe { dev.alloc::<i16>(n * TILE_VALUES)? };
    // SAFETY: as above
    let costs = unsafe { dev.alloc::<u8>((scratch_tiles * cost_bytes) as usize)? };
    // SAFETY: as above
    let edges = unsafe { dev.alloc::<u8>((scratch_tiles * edge_bytes) as usize)? };
    let (x_s, x_l) = x.storage_and_layout();
    let Storage::Cuda(x_c) = &*x_s else {
        inference_tensor::bail!("EXL3 tiles live on CUDA");
    };
    let (input, _x) = slice_ptr(x_c.as_cuda_slice::<f32>()?, x_l.start_offset());
    let (o, _o) = slice_ptr(&out, 0);
    let (i, _i) = slice_ptr(&idx, 0);
    let (c, _c) = slice_ptr(&costs, 0);
    let (e, _e) = slice_ptr(&edges, 0);
    // SAFETY: the pointers are the live device buffers above, sized as the kernels read them (n tiles of input,
    // values and states; scratch_tiles tiles of each scratch), on this device's stream
    let ran = unsafe {
        qffi::launch_exl3_quantize_tiles(
            input as *const f32,
            o as *mut f32,
            i as *mut u16,
            c as *mut half::f16,
            e as *mut u16,
            n as i64,
            scratch_tiles,
            k as i32,
            codebook as i32,
            half_k,
            dev.cuda_stream().cu_stream(),
        )
    };
    if ran < 0 {
        inference_tensor::bail!("no EXL3 quantizer fits {bits:?} on {codebook:?}");
    }
    drop((_x, _o, _i, _c, _e));
    let shape = Shape::from((n, TILE_VALUES));
    Ok((
        Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
            shape.clone(),
        )),
        Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(idx, dev.clone())),
            shape,
        )),
    ))
}

#[cfg(test)]
mod tests {
    use inference_tensor::{Device, Result, Tensor};

    use super::{pack_tile, quantize_tiles};
    use crate::exl3::{Exl3Bits, Exl3Codebook, decode_tile, tile_order};

    // the quantizer's states packed into tile words decode, through our own decoder, to the values the quantizer
    // reported; and they are closer to the inputs than a coarser rate's (every codebook, a half-integer rate)
    #[test]
    fn quantized_tiles_pack_and_decode_to_their_values() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("SKIP: no CUDA device");
            return Ok(());
        };
        let n = 8;
        let x = Tensor::randn(0f32, 1f32, (n, 256), &device)?;
        let perm = tile_order();
        let mut errors = Vec::new();
        for (bits, codebook) in [
            (Exl3Bits::Whole(2), Exl3Codebook::ThreeInst),
            (Exl3Bits::Whole(4), Exl3Codebook::ThreeInst),
            (Exl3Bits::Whole(3), Exl3Codebook::Mcg),
            (Exl3Bits::Whole(4), Exl3Codebook::Mul1),
            (Exl3Bits::Half(2), Exl3Codebook::Mul1),
        ] {
            let (values, states) = quantize_tiles(&x, bits, codebook)?;
            let values = values.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
            let states = states.to_device(&Device::Cpu)?.to_vec2::<i16>()?;
            for (t, (v, s)) in values.iter().zip(&states).enumerate() {
                let s = s.iter().map(|&s| s as u16).collect::<Vec<_>>();
                let decoded = decode_tile(&pack_tile(&s, bits), bits, codebook);
                for p in 0..256 {
                    assert_eq!(
                        decoded[perm[p]], v[p],
                        "{bits:?} {codebook:?} tile {t} position {p}"
                    );
                }
            }
            let mse = (Tensor::new(values, &Device::Cpu)?.reshape((n, 256))?
                - x.to_device(&Device::Cpu)?)?
            .sqr()?
            .mean_all()?
            .to_scalar::<f32>()?;
            errors.push((bits, codebook, mse));
        }
        // more bits, smaller error: 2 bits against 4 on the same codebook
        assert!(errors[1].2 < errors[0].2, "{errors:?}");
        Ok(())
    }
}
