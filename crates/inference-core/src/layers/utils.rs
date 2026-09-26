// layers/mod.rs allows these for its own math; this file keeps the crate-wide deny
#![deny(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Result, Tensor};

pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        Tensor::cat(&vec![&x; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}
