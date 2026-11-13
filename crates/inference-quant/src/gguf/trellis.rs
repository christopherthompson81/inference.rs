//! Procedural codebooks, as trellis quants use them: a code's 32-bit state steps by a multiply (and an add), and each
//! step's state decodes to one value. ik_llama's IQ*_KT types take [`IK`] from an index plus [`IK_INDEX_OFFSET`];
//! EXL3 decodes each 16-bit trellis window once through [`THREE_INST`] (QTIP's), [`MCG`] or [`MUL1`].

/// How a state becomes a value.
#[derive(Clone, Copy)]
pub(crate) enum Decode {
    /// The sum of the masked state's four bytes, less `bias`.
    ByteSum { mask: u32, bias: i32 },
    /// The masked and flipped state's two halves read as f16 and added in f16.
    Fp16Pair { mask: u32, xor: u32 },
    /// The state's byte sum over `base` read as f16, then `x * scale + bias` with one f16 rounding (`__hfma`).
    Fp16ByteSum { base: u16, scale: u16, bias: u16 },
}

#[derive(Clone, Copy)]
pub(crate) struct Codebook {
    pub mul: u32,
    pub add: u32,
    pub decode: Decode,
}

/// ik_llama's integer trellis: a multiply, then the sum of the state's four 6-bit bytes less 126 (-126..=126).
pub(crate) const IK: Codebook = Codebook {
    mul: 0xCBAC1FED,
    add: 0,
    decode: Decode::ByteSum {
        mask: 0x3f3f3f3f,
        bias: 126,
    },
};

/// QTIP's 3INST: a multiply and an add, then two f16 halves summed.
pub(crate) const THREE_INST: Codebook = Codebook {
    mul: 89226354,
    add: 64248484,
    decode: FP16_PAIR,
};

/// EXL3's "mcg": the multiply alone, then 3INST's decode.
pub(crate) const MCG: Codebook = Codebook {
    mul: 0xCBAC1FED,
    add: 0,
    decode: FP16_PAIR,
};

/// EXL3's "mul1": a multiply, then the byte sum (over 1024, an f16 with unit spacing) scaled to about -3.5..3.5.
pub(crate) const MUL1: Codebook = Codebook {
    mul: 0x83DCD12D,
    add: 0,
    decode: Decode::Fp16ByteSum {
        base: 0x6400,
        scale: 0x1eee,
        bias: 0xc931,
    },
};

const FP16_PAIR: Decode = Decode::Fp16Pair {
    mask: 0x8fff8fff,
    xor: 0x3b603b60,
};

/// ik_llama's codes start their states this far above the stored index.
pub(crate) const IK_INDEX_OFFSET: u32 = 4096;

impl Codebook {
    pub(crate) const fn step(&self, state: u32) -> u32 {
        state.wrapping_mul(self.mul).wrapping_add(self.add)
    }

    /// A byte-sum codebook's integer value of `state`.
    pub(crate) const fn decode(&self, state: u32) -> i32 {
        match self.decode {
            Decode::ByteSum { mask, bias } => {
                let b = (state & mask).to_le_bytes();
                b[0] as i32 + b[1] as i32 + b[2] as i32 + b[3] as i32 - bias
            }
            _ => panic!("an f16 codebook has no integer value"),
        }
    }

    /// The value of `state` (already stepped), for any decode; the f16 ones round once, as the CUDA ops do.
    pub(crate) fn value(&self, state: u32) -> f32 {
        let f16 = |bits: u32| f64::from(half::f16::from_bits(bits as u16).to_f32());
        match self.decode {
            Decode::ByteSum { .. } => self.decode(state) as f32,
            Decode::Fp16Pair { mask, xor } => {
                let x = (state & mask) ^ xor;
                half::f16::from_f64(f16(x & 0xffff) + f16(x >> 16)).to_f32()
            }
            Decode::Fp16ByteSum { base, scale, bias } => {
                let sum = state
                    .to_le_bytes()
                    .iter()
                    .map(|&b| u32::from(b))
                    .sum::<u32>();
                let x = f16(u32::from(base) + sum);
                half::f16::from_f64(x * f16(u32::from(scale)) + f16(u32::from(bias))).to_f32()
            }
        }
    }

    /// The `N` values a code starting at `state` emits, one step each.
    pub(crate) fn values<const N: usize>(&self, mut state: u32) -> [i32; N] {
        std::array::from_fn(|_| {
            state = self.step(state);
            self.decode(state)
        })
    }

    /// The state after each of `N` steps is the start times these, for a codebook with no add (which a SIMD decode
    /// uses to take a code's steps in one multiply).
    #[cfg(any(target_arch = "x86_64", test))]
    pub(crate) const fn powers<const N: usize>(&self) -> [u32; N] {
        assert!(
            self.add == 0,
            "steps with an add are not powers of the multiplier"
        );
        let mut out = [0u32; N];
        let mut power = 1u32;
        let mut k = 0;
        while k < N {
            power = power.wrapping_mul(self.mul);
            out[k] = power;
            k += 1;
        }
        out
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) const fn byte_sum(&self) -> (u32, i32) {
        match self.decode {
            Decode::ByteSum { mask, bias } => (mask, bias),
            _ => panic!("an f16 codebook has no byte-sum mask"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{IK, IK_INDEX_OFFSET};

    // a code's values by steps and by powers of the multiplier agree, and stay in -126..=126
    #[test]
    fn steps_and_powers_give_the_same_values() {
        let powers = IK.powers::<8>();
        for index in [0u32, 1, 4095, 65535] {
            let start = index + IK_INDEX_OFFSET;
            let by_steps = IK.values::<8>(start);
            let by_powers = powers.map(|p| IK.decode(start.wrapping_mul(p)));
            assert_eq!(by_steps, by_powers);
            assert!(by_steps.iter().all(|v| (-126..=126).contains(v)));
        }
    }
}
