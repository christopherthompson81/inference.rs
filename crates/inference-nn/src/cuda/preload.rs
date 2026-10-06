use candle_core::{Device, Result, cuda_backend::kernels};

pub fn preload_candle_kernels(device: &Device) -> Result<usize> {
    let Device::Cuda(cuda_device) = device else {
        return Ok(0);
    };

    let mut count = 0;
    for module in MODULES {
        // required entries first, so a module whose image fails to load errors on its first entry
        let (required, optional): (Vec<_>, Vec<_>) = module
            .entries()
            .iter()
            .partition(|entry| !module.is_optional(entry));
        for entry in required.into_iter().chain(optional) {
            match cuda_device.get_or_load_func(entry, module) {
                Ok(_) => count += 1,
                // a multi-arch build's lower archs lack the kernels a higher `__CUDA_ARCH__` guard keeps
                Err(_) if module.is_optional(entry) => {}
                Err(error) => return Err(error),
            }
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
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;

    #[test]
    fn every_module_loads_and_preloads_its_entries() -> Result<()> {
        skip_without_cuda!();
        let device = Device::new_cuda(0)?;
        assert!(MODULES.iter().all(|module| !module.entries().is_empty()));
        let total = MODULES
            .iter()
            .map(|module| module.entries().len())
            .sum::<usize>();
        let required = MODULES
            .iter()
            .flat_map(|module| {
                module
                    .entries()
                    .iter()
                    .filter(|entry| !module.is_optional(entry))
            })
            .count();
        let loaded = preload_candle_kernels(&device)?;
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let stream = cuda.cuda_stream();
        let context = stream.context();
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(candle_core::Error::wrap)?;
        let minor = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(candle_core::Error::wrap)?;
        let cc = (major * 10 + minor) as usize;
        let archs: Vec<usize> = kernels::ARCHS
            .iter()
            .filter_map(|arch| arch.trim_end_matches(['a', 'f']).parse().ok())
            .collect();
        let device_arch = archs
            .iter()
            .filter(|&&arch| arch / 10 == cc / 10 && arch <= cc)
            .max();
        // the build's highest arch keeps every entry; a lower one may lack the optional ones
        if device_arch == archs.iter().max() {
            assert_eq!(loaded, total);
        } else {
            assert!(
                (required..=total).contains(&loaded),
                "{loaded} of {required}..={total}"
            );
        }
        Ok(())
    }
}
