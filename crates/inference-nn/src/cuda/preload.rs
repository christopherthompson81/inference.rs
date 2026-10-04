use candle_core::{Device, Result, cuda_backend::kernels};

pub fn preload_candle_kernels(device: &Device) -> Result<usize> {
    let Device::Cuda(cuda_device) = device else {
        return Ok(0);
    };

    let mut count = 0;
    for module in MODULES {
        for entry in module.entries() {
            let _func = cuda_device.get_or_load_func(entry, module)?;
            count += 1;
        }
    }

    Ok(count)
}

static MODULES: [&kernels::Module; 11] = [
    &kernels::AFFINE,
    &kernels::BINARY,
    &kernels::CAST,
    &kernels::CONV,
    &kernels::FILL,
    &kernels::INDEXING,
    &kernels::QUANTIZED,
    &kernels::REDUCE,
    &kernels::SORT,
    &kernels::TERNARY,
    &kernels::UNARY,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_module_loads_and_preloads_its_entries() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        assert!(MODULES.iter().all(|module| !module.entries().is_empty()));
        let total = MODULES
            .iter()
            .map(|module| module.entries().len())
            .sum::<usize>();
        assert_eq!(preload_candle_kernels(&device)?, total);
        Ok(())
    }
}
