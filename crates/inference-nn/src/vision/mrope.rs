use inference_tensor::{Result, Tensor};

use crate::gdn::RecurrentBatchKind;

fn mrope_position_deltas_for_broadcast(
    mrope_position_deltas: &Tensor,
    batch: usize,
) -> Result<Tensor> {
    match mrope_position_deltas.dims() {
        [b] if *b == batch => mrope_position_deltas.reshape((1, batch, 1)),
        [b, 1] if *b == batch => mrope_position_deltas.reshape((1, batch, 1)),
        [1, b, 1] if *b == batch => Ok(mrope_position_deltas.clone()),
        _ => inference_tensor::bail!(
            "MRoPE position deltas shape {:?} is incompatible with batch {batch}",
            mrope_position_deltas.shape()
        ),
    }
}

pub fn mrope_position_ids_for_input(
    position_ids: &Tensor,
    mrope_position_deltas: &Tensor,
    input_ids: &Tensor,
    seqlen_offsets: &[usize],
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let (planes, pos_batch, full_len) = position_ids.dims3()?;
    if pos_batch != batch || seqlen_offsets.len() != batch {
        inference_tensor::bail!(
            "MRoPE position ids shape {:?} is incompatible with input shape {:?}",
            position_ids.shape(),
            input_ids.shape()
        );
    }

    if seqlen_offsets.iter().all(|offset| {
        offset
            .checked_add(seq_len)
            .is_some_and(|end| end <= full_len)
    }) {
        let mut indices = Vec::with_capacity(planes * batch * seq_len);
        for _ in 0..planes {
            for offset in seqlen_offsets {
                for pos in *offset..*offset + seq_len {
                    indices.push(u32::try_from(pos).map_err(inference_tensor::Error::wrap)?);
                }
            }
        }
        let indices = Tensor::from_vec(indices, (planes, batch, seq_len), position_ids.device())?;
        return position_ids.gather(&indices, 2);
    }

    let offsets = seqlen_offsets
        .iter()
        .map(|offset| i64::try_from(*offset).map_err(inference_tensor::Error::wrap))
        .collect::<Result<Vec<_>>>()?;
    let offsets = Tensor::from_vec(offsets, (1, batch, 1), input_ids.device())?;
    let seq_len_i64 = i64::try_from(seq_len).map_err(inference_tensor::Error::wrap)?;
    let relative =
        Tensor::arange(0i64, seq_len_i64, input_ids.device())?.reshape((1, 1, seq_len))?;
    let position_ids = offsets.broadcast_add(&relative)?.repeat((planes, 1, 1))?;
    let mrope_position_deltas = mrope_position_deltas_for_broadcast(mrope_position_deltas, batch)?;
    position_ids.broadcast_add(&mrope_position_deltas)
}

pub fn text_position_ids(input_ids: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    if seqlen_offsets.len() != batch {
        inference_tensor::bail!(
            "RoPE offsets ({}) do not match batch size {batch}",
            seqlen_offsets.len()
        );
    }
    crate::model::text_positions_tensor(seqlen_offsets, seq_len, input_ids.device())?
        .reshape((batch, seq_len))
}

pub fn text_decode_mrope_position_ids_from_context(
    input_ids: &Tensor,
    ctx: &crate::model::ModelForwardContext<'_>,
) -> Result<Option<Tensor>> {
    text_decode_position_ids_from_context(input_ids, ctx).and_then(|positions| {
        positions
            .map(|positions| {
                let (batch, seq_len) = positions.dims2()?;
                positions
                    .to_dtype(inference_tensor::DType::I64)?
                    .reshape((1, batch, seq_len))?
                    .repeat((3, 1, 1))
            })
            .transpose()
    })
}

pub fn text_decode_position_ids_from_context(
    input_ids: &Tensor,
    ctx: &crate::model::ModelForwardContext<'_>,
) -> Result<Option<Tensor>> {
    let (batch, seq_len) = input_ids.dims2()?;
    let rope_positions = match ctx.cache().rope_positions(input_ids.device()) {
        Some(rope_positions) => rope_positions.clone(),
        None if matches!(
            ctx.recurrent_batch_kind(),
            Some(RecurrentBatchKind::Decode | RecurrentBatchKind::SpeculativeDecode)
        ) =>
        {
            crate::model::decode_positions_tensor(ctx.position_ids(), seq_len, input_ids.device())?
        }
        None => return Ok(None),
    };
    if rope_positions.dim(0)? != batch * seq_len {
        inference_tensor::bail!(
            "rope positions shape {:?} is incompatible with input shape {:?}",
            rope_positions.shape(),
            input_ids.shape()
        );
    }
    Ok(Some(
        rope_positions
            .to_dtype(inference_tensor::DType::U32)?
            .reshape((batch, seq_len))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::FlashParams;
    use crate::model::{ForwardCache, ModelForwardContext};
    use inference_tensor::IndexOp;

    #[test]
    fn mrope_position_ends_are_decode_only() -> Result<()> {
        let input_ids = Tensor::zeros(
            (2, 3),
            inference_tensor::DType::U32,
            &inference_tensor::Device::Cpu,
        )?;
        let offsets = [0, 0];
        let context_lens = [(0, 3), (0, 3)];
        let position_ids = [1, 10];
        let flash = FlashParams::empty(true);
        let prefill = ModelForwardContext::with_cache(
            ForwardCache::None,
            &offsets,
            &context_lens,
            &position_ids,
            &flash,
        )
        .with_recurrent_batch_kind(RecurrentBatchKind::Prefill);

        assert!(text_decode_mrope_position_ids_from_context(&input_ids, &prefill)?.is_none());

        let position_ids = [5, 10];
        let decode = ModelForwardContext::with_cache(
            ForwardCache::None,
            &offsets,
            &context_lens,
            &position_ids,
            &flash,
        )
        .with_recurrent_batch_kind(RecurrentBatchKind::SpeculativeDecode);
        let positions = text_decode_mrope_position_ids_from_context(&input_ids, &decode)?
            .expect("decode MRoPE positions missing");
        let text_positions = text_decode_position_ids_from_context(&input_ids, &decode)?
            .expect("decode text positions missing");

        assert_eq!(
            positions.i((0, .., ..))?.to_vec2::<i64>()?,
            vec![vec![2, 3, 4], vec![7, 8, 9]]
        );
        assert_eq!(text_positions.dims(), &[2, 3]);
        assert_eq!(
            text_positions.to_vec2::<u32>()?,
            vec![vec![2, 3, 4], vec![7, 8, 9]]
        );
        Ok(())
    }
}
