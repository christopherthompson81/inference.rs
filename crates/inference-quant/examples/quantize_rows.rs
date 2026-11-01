//! Rows of little-endian f32 values to raw GGUF blocks with our quantizers, for `scripts/iq_quantize_parity.py`.
//! Usage: quantize_rows <ggml type id> <cols> <values.f32> <imatrix.f32 | -> <out.bin>

use inference_quant::GgufType;

fn read_f32(path: &str) -> inference_tensor::Result<Vec<f32>> {
    Ok(std::fs::read(path)?
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn main() -> inference_tensor::Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let [_, id, cols, values, imatrix, out] = args.as_slice() else {
        inference_tensor::bail!(
            "usage: quantize_rows <ggml type id> <cols> <values.f32> <imatrix.f32 | -> <out.bin>"
        );
    };
    let id: u32 = id.parse().map_err(inference_tensor::Error::msg)?;
    let ty = GgufType::RAW_BLOCKS
        .into_iter()
        .find(|ty| ty.id() == id)
        .ok_or_else(|| inference_tensor::Error::msg(format!("no raw GGUF type with id {id}")))?;
    let imatrix = (imatrix != "-").then(|| read_f32(imatrix)).transpose()?;
    let bytes = inference_quant::iq_quantize::quantize_rows(
        ty,
        cols.parse().map_err(inference_tensor::Error::msg)?,
        &read_f32(values)?,
        imatrix.as_deref(),
    )?;
    std::fs::write(out, bytes)?;
    Ok(())
}
