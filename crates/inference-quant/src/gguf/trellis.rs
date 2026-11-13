//! Procedural codebooks, as trellis quants use them: a code's 32-bit state steps by a multiply (and an add), and each
//! step's state decodes to one value. ik_llama's IQ*_KT types take [`IK`] from an index plus [`IK_INDEX_OFFSET`];
//! QTIP's 3INST and EXL3's codebooks are other multipliers and decodes of the same shape.

/// How a state becomes a value.
#[derive(Clone, Copy)]
pub(crate) enum Decode {
    /// The sum of the masked state's four bytes, less `bias`.
    ByteSum { mask: u32, bias: i32 },
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

/// ik_llama's codes start their states this far above the stored index.
pub(crate) const IK_INDEX_OFFSET: u32 = 4096;

impl Codebook {
    pub(crate) const fn step(&self, state: u32) -> u32 {
        state.wrapping_mul(self.mul).wrapping_add(self.add)
    }

    pub(crate) const fn decode(&self, state: u32) -> i32 {
        match self.decode {
            Decode::ByteSum { mask, bias } => {
                let b = (state & mask).to_le_bytes();
                b[0] as i32 + b[1] as i32 + b[2] as i32 + b[3] as i32 - bias
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
