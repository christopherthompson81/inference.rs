//! Elementwise maps over strided operands: `map_f32` for f32 arithmetic, `copy_b32` for any 4-byte element bit for bit.

use std::sync::Arc;

use crate::memory::HostBuffer;
use crate::{Error, Recorder, Result};

/// The most dims a map's shape has (the shaders' `MAX_DIMS`).
pub const MAX_DIMS: usize = 6;
pub(crate) const MAP_BINDINGS: usize = 3;
// n, rank, op, p0, p1 and the three offsets, then each of dims and the three operands' strides
pub(crate) const MAP_PUSH_BYTES: u32 = ((8 + 4 * MAX_DIMS) * 4) as u32;
const THREADS: usize = 256;
// enough workgroups to fill the GPU; past them each thread loops
const MAX_GROUPS: usize = 4096;

/// An elementwise op, numbered as the shaders' constants.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u32)]
pub enum MapOp {
    Copy = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    Maximum = 5,
    Minimum = 6,
    Neg = 7,
    Exp = 8,
    Log = 9,
    Sqr = 10,
    Sqrt = 11,
    Recip = 12,
    Silu = 13,
    Relu = 14,
    Gelu = 15,
    Tanh = 16,
    Abs = 17,
    /// `x * p0 + p1`
    Affine = 18,
    /// `x ^ p0`
    Powf = 19,
    Sin = 20,
    Cos = 21,
    /// every element `p0`
    Fill = 22,
    Floor = 23,
    Ceil = 24,
    Round = 25,
    Sign = 26,
    /// `x` where positive, else `p0 * (e^x - 1)`
    Elu = 27,
}

impl MapOp {
    fn binary(self) -> bool {
        (MapOp::Add as u32..=MapOp::Minimum as u32).contains(&(self as u32))
    }
}

/// One operand: its buffer, the element its first index starts at, and its stride along each of the map's dims.
#[derive(Clone, Copy)]
pub struct Operand<'a> {
    pub buffer: &'a Arc<HostBuffer>,
    pub offset: usize,
    pub strides: &'a [usize],
}

/// `out = op(a[, b])` over `dims`; `b` is read only by binary ops, `a` by all but `Fill`.
pub struct Map<'a> {
    pub op: MapOp,
    pub dims: &'a [usize],
    pub a: Operand<'a>,
    pub b: Option<Operand<'a>>,
    pub out: Operand<'a>,
    pub params: [f32; 2],
}

impl Recorder<'_> {
    pub fn map_f32(&self, map: &Map<'_>) -> Result<()> {
        let params = map.params.map(f32::to_bits);
        self.map(map, params, false)
    }

    /// [`Self::map_f32`]'s `Copy` and `Fill` over 4-byte elements of any type, `fill` the word `Fill` writes.
    pub fn copy_b32(&self, map: &Map<'_>, fill: u32) -> Result<()> {
        if !matches!(map.op, MapOp::Copy | MapOp::Fill) {
            return Err(Error::Invalid(format!(
                "copy_b32 runs Copy and Fill, not {:?}",
                map.op
            )));
        }
        self.map(map, [fill, 0], true)
    }

    fn map(&self, map: &Map<'_>, params: [u32; 2], bits: bool) -> Result<()> {
        let rank = map.dims.len();
        let b = map.b.unwrap_or(map.a);
        let n: usize = map.dims.iter().product();
        if rank > MAX_DIMS
            || [map.a, b, map.out].iter().any(|o| o.strides.len() != rank)
            || (map.op.binary() && map.b.is_none())
        {
            return Err(Error::Invalid(format!(
                "a {:?} map over {:?}",
                map.op, map.dims
            )));
        }
        if n == 0 {
            return Ok(());
        }
        let small = |v: usize| {
            u32::try_from(v).map_err(|_| Error::Invalid(format!("map index {v} past u32")))
        };
        let mut words = vec![
            small(n)?,
            small(rank)?,
            map.op as u32,
            params[0],
            params[1],
            small(map.a.offset)?,
            small(b.offset)?,
            small(map.out.offset)?,
        ];
        for list in [map.dims, map.a.strides, b.strides, map.out.strides] {
            for i in 0..MAX_DIMS {
                words.push(small(list.get(i).copied().unwrap_or(0))?);
            }
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let pipeline = if bits {
            &self.context.kernels.copy_b32
        } else {
            &self.context.kernels.map_f32
        };
        let groups = n.div_ceil(THREADS).min(MAX_GROUPS) as u32;
        pipeline.dispatch(
            self,
            &[map.a.buffer.buffer, b.buffer.buffer, map.out.buffer.buffer],
            &bytes,
            [groups, 1, 1],
        );
        self.retain(map.a.buffer.clone());
        self.retain(b.buffer.clone());
        self.retain(map.out.buffer.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Context;

    const TOLERANCE: f32 = 1e-5;

    fn context() -> Option<Arc<Context>> {
        Context::new()
            .map_err(|err| eprintln!("skipping: {err}"))
            .ok()
    }

    fn buffer(context: &Arc<Context>, values: &[f32]) -> Result<Arc<HostBuffer>> {
        let mut buffer = HostBuffer::new(context, values.len() * 4)?;
        buffer.as_f32_mut()[..values.len()].copy_from_slice(values);
        Ok(Arc::new(buffer))
    }

    // a [2, 3] lhs times a broadcast row, a transposed read, and a unary op into a strided output
    #[test]
    fn maps_read_and_write_strided_operands() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        let lhs: Vec<f32> = (0..6).map(|i| i as f32 - 2.5).collect();
        let row = [2.0f32, -1.0, 0.5];
        let (a, b) = (buffer(&context, &lhs)?, buffer(&context, &row)?);
        let out = Arc::new(HostBuffer::new(&context, 64 * 4)?);
        let operand = |buffer, offset, strides| Operand {
            buffer,
            offset,
            strides,
        };
        context.submit(|r| {
            r.map_f32(&Map {
                op: MapOp::Mul,
                dims: &[2, 3],
                a: operand(&a, 0, &[3, 1]),
                b: Some(operand(&b, 0, &[0, 1])),
                out: operand(&out, 0, &[3, 1]),
                params: [0.0; 2],
            })?;
            // the transpose of lhs, [3, 2], silu'd into every other element from 10
            r.map_f32(&Map {
                op: MapOp::Silu,
                dims: &[3, 2],
                a: operand(&a, 0, &[1, 3]),
                b: None,
                out: operand(&out, 10, &[4, 2]),
                params: [0.0; 2],
            })
        })?;
        let got = out.as_f32();
        for r in 0..2 {
            for c in 0..3 {
                assert!((got[r * 3 + c] - lhs[r * 3 + c] * row[c]).abs() <= TOLERANCE);
            }
        }
        for r in 0..3 {
            for c in 0..2 {
                let x = lhs[c * 3 + r];
                let want = x / (1.0 + (-x).exp());
                assert!(
                    (got[10 + r * 4 + c * 2] - want).abs() <= TOLERANCE,
                    "{r},{c}"
                );
            }
        }
        Ok(())
    }

    // u32 words that read as denormal floats survive the copy and the fill unchanged
    #[test]
    fn copy_b32_keeps_bits() -> Result<()> {
        let Some(context) = context() else {
            return Ok(());
        };
        let words: Vec<u32> = (0..300).map(|i| i * 7 + 1).collect();
        let mut src = HostBuffer::new(&context, words.len() * 4)?;
        for (dst, w) in src
            .as_bytes_mut()
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&words)
        {
            dst.copy_from_slice(&w.to_le_bytes());
        }
        let src = Arc::new(src);
        let out = Arc::new(HostBuffer::new(&context, (words.len() + 4) * 4)?);
        let strides = [1usize];
        context.submit(|r| {
            let operand = |buffer, offset| Operand {
                buffer,
                offset,
                strides: &strides,
            };
            r.copy_b32(
                &Map {
                    op: MapOp::Copy,
                    dims: &[words.len()],
                    a: operand(&src, 0),
                    b: None,
                    out: operand(&out, 0),
                    params: [0.0; 2],
                },
                0,
            )?;
            r.copy_b32(
                &Map {
                    op: MapOp::Fill,
                    dims: &[4],
                    a: operand(&src, 0),
                    b: None,
                    out: operand(&out, words.len()),
                    params: [0.0; 2],
                },
                3,
            )
        })?;
        let got: Vec<u32> = out
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        assert_eq!(&got[..words.len()], &words[..]);
        assert_eq!(&got[words.len()..words.len() + 4], &[3; 4]);
        Ok(())
    }
}
