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
    use candle_core::cuda_backend;

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
        let device_arch = cuda_backend::kernel_arch(cuda.compute_cap());
        let highest = kernels::ARCHS
            .iter()
            .filter_map(|arch| arch.trim_end_matches(['a', 'f']).parse::<usize>().ok())
            .max();
        // the build's highest arch keeps every entry; a lower one may lack the optional ones
        if device_arch == highest {
            assert_eq!(loaded, total);
        } else {
            assert!(
                (required..=total).contains(&loaded),
                "{loaded} of {required}..={total}"
            );
        }
        Ok(())
    }

    #[test]
    fn a_device_opens_only_with_kernels_for_its_arch() {
        let Ok(cc) = cuda_backend::device_compute_cap(0) else {
            return;
        };
        match cuda_backend::kernel_arch(cc) {
            Some(_) => assert!(Device::new_cuda(0).is_ok()),
            None => {
                let error = Device::new_cuda(0).unwrap_err().to_string();
                assert!(error.contains("CUDA_COMPUTE_CAP"), "{error}");
            }
        }
    }

    #[test]
    fn a_device_runs_the_highest_built_arch_of_its_family() {
        let among = cuda_backend::kernel_arch_among;
        assert_eq!(among(&["86", "90a"], 89), Some(86));
        assert_eq!(among(&["80", "86", "89", "90a"], 89), Some(89));
        assert_eq!(among(&["86", "90a"], 90), Some(90));
        assert_eq!(among(&["86", "121f"], 121), Some(121));
        // arch-specific SASS runs only on its own capability: a 120a build has nothing for a 12.1 device
        assert_eq!(among(&["86", "120a"], 121), None);
        assert_eq!(among(&["100a"], 103), None);
        // no SASS in its family: the device runs nothing from this build
        assert_eq!(among(&["86", "90a"], 75), None);
        assert_eq!(among(&["86", "90a"], 120), None);
    }
}
