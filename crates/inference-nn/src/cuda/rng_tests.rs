use candle_core::{Device, Result, Tensor};

const SEED: u64 = 7;

#[test]
fn seeded_cuda_draws_repeat() -> Result<()> {
    skip_without_cuda!();
    let device = Device::new_cuda(0)?;
    let draw = || -> Result<(Vec<f32>, Vec<f32>)> {
        let normal = Tensor::randn(0f32, 1., 63, &device)?.to_vec1()?;
        let uniform = Tensor::rand(0f32, 1., 64, &device)?.to_vec1()?;
        Ok((normal, uniform))
    };
    device.set_seed(SEED)?;
    let first = draw()?;
    device.set_seed(SEED)?;
    assert_eq!(draw()?, first);
    assert!(first.1.iter().all(|u| (0.0..=1.0).contains(u)));
    Ok(())
}
