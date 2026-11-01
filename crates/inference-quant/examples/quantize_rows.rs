//! Rows of little-endian f32 values to raw GGUF blocks with our quantizers, for `scripts/iq_quantize_parity.py`.
//! Usage: quantize_rows <ggml type id> <cols> <values.f32> <imatrix.f32 | -> <out.bin>

use inference_quant::GgufType;

fn read_f32(path: &str) -> inference_tensor::Result<Vec<f32>> {
    Ok(std::fs::read(path)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
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
        .chain([
            GgufType::Q8_0,
            GgufType::Q3K,
            GgufType::Q4K,
            GgufType::Q5K,
            GgufType::Q6K,
        ])
        .find(|ty| ty.id() == id)
        .ok_or_else(|| {
            inference_tensor::Error::msg(format!("no writable GGUF type with id {id}"))
        })?;
    let imatrix = (imatrix != "-").then(|| read_f32(imatrix)).transpose()?;
    let values = read_f32(values)?;
    let cols: usize = cols.parse().map_err(inference_tensor::Error::msg)?;
    let bytes = inference_quant::requantize::quantize_matrix(
        ty,
        cols,
        &values,
        imatrix.as_deref(),
        values.len() / cols,
    )?;
    std::fs::write(out, bytes)?;
    Ok(())
}
