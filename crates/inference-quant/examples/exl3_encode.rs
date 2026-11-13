//! One linear encoded as EXL3 against its input Hessian, for `scripts/exl3_encode_parity.py`: prints the output
//! error and writes `<key>.trellis`, `.suh` and `.svh` (and the codebook's marker) as exllamav3 stores them.
//! Usage: exl3_encode <model.safetensors> <key> <hessian.safetensors> <bits> <3inst|mcg|mul1> <seed> <out.safetensors>

#[cfg(feature = "cuda")]
fn main() -> inference_tensor::Result<()> {
    use inference_quant::exl3::{Exl3Bits, Exl3Codebook, encode::encode};
    use inference_tensor::{Device, Tensor};

    let args = std::env::args().collect::<Vec<_>>();
    let [_, model, key, hessian, bits, codebook, seed, out] = args.as_slice() else {
        inference_tensor::bail!(
            "usage: exl3_encode <model.safetensors> <key> <hessian.safetensors> <bits> <3inst|mcg|mul1> <seed> <out>"
        );
    };
    let device = Device::new_cuda(0)?;
    let weight = inference_tensor::safetensors::load(model, &device)?
        .remove(&format!("{key}.weight"))
        .ok_or_else(|| inference_tensor::Error::msg(format!("no {key}.weight")))?;
    let h = inference_tensor::safetensors::load(hessian, &device)?
        .remove("hin")
        .ok_or_else(|| inference_tensor::Error::msg("no hin"))?;
    let rate = bits.parse::<f64>().map_err(inference_tensor::Error::wrap)?;
    let bits = if rate.fract() == 0.0 {
        Exl3Bits::Whole(rate as usize)
    } else {
        Exl3Bits::Half(rate as usize)
    };
    let (codebook, marker) = match codebook.as_str() {
        "mcg" => (Exl3Codebook::Mcg, Some(("mcg", 0xCBAC1FEDu32))),
        "mul1" => (Exl3Codebook::Mul1, Some(("mul1", 0x83DCD12Du32))),
        _ => (Exl3Codebook::ThreeInst, None),
    };
    let seed = seed.parse::<u64>().map_err(inference_tensor::Error::wrap)?;
    let start = std::time::Instant::now();
    let e = encode(&weight, &h, bits, codebook, seed)?;
    println!(
        "proxy error {:.6} in {:.2}s",
        e.proxy_error,
        start.elapsed().as_secs_f32()
    );
    let mut tensors = std::collections::HashMap::from([
        (format!("{key}.trellis"), e.trellis),
        (format!("{key}.suh"), e.suh.to_device(&Device::Cpu)?),
        (format!("{key}.svh"), e.svh.to_device(&Device::Cpu)?),
    ]);
    if let Some((name, mult)) = marker {
        tensors.insert(
            format!("{key}.{name}"),
            Tensor::new(&[mult as i32], &Device::Cpu)?,
        );
    }
    inference_tensor::safetensors::save(&tensors, out)?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("exl3_encode runs on CUDA: build with --features cuda");
}
