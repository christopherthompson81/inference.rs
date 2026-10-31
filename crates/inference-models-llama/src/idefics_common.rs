#![allow(clippy::cast_precision_loss)]

//! The image-side forward Idefics2 and Idefics3 share: padding sub-images, patch masks and per-image cached encoding.

use std::sync::Mutex;

use image::{DynamicImage, GenericImageView};
use inference_tensor::{D, DType, Device, Result, Tensor};
use inference_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};

use crate::media_inputs::preprocessor_config::PreProcessorConfig;
use crate::paged_attention::{
    block_hash::MultimodalKind,
    encoder_cache::{CacheModality, EncoderCacheManager},
};
use crate::vision::multimodal_layout::{MultimodalEncoderKey, MultimodalEncoderOutputs};

// Drops the all-zero padding sub-images and their masks; with no mask, every pixel is valid.
pub(crate) fn drop_padding_images(
    pixel_values: &Tensor,
    pixel_attention_mask: Option<Tensor>,
) -> Result<(Tensor, Tensor)> {
    let rank = pixel_values.rank();
    let nb_values_per_image = pixel_values.dims()[1..].iter().product::<usize>();
    let real_images = pixel_values
        .eq(0.0f64)?
        .sum(vec![rank - 1, rank - 2, rank - 3])?
        .ne(nb_values_per_image as f64)?
        .to_vec1::<u8>()?;
    let keep = |tensor: &Tensor| -> Result<Tensor> {
        let kept = real_images
            .iter()
            .enumerate()
            .filter(|(_, real)| **real != 0)
            .map(|(index, _)| tensor.narrow(0, index, 1))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&kept, 0)
    };
    let pixel_values = keep(pixel_values)?;
    let pixel_attention_mask = match pixel_attention_mask {
        Some(mask) => keep(&mask)?,
        None => Tensor::ones(
            (
                pixel_values.dims()[0],
                pixel_values.dims()[2],
                pixel_values.dims()[3],
            ),
            DType::U8,
            pixel_values.device(),
        )?,
    };
    Ok((pixel_values, pixel_attention_mask))
}

// The number of valid pixels in each patch, shaped (images, patches high, patches wide).
pub(crate) fn patch_valid_pixels(
    pixel_attention_mask: &Tensor,
    patch_size: usize,
) -> Result<Tensor> {
    pixel_attention_mask
        .unfold(1, patch_size, patch_size)?
        .unfold(2, patch_size, patch_size)?
        .sum((D::Minus1, D::Minus2))
}

pub(crate) struct SubimageBatch<'a> {
    pub label: &'static str,
    pub image_hashes: &'a [u64],
    pub subimage_counts: &'a [usize],
    pub pixel_values: &'a Tensor,
    pub patch_attention_mask: &'a Tensor,
    // rows each sub-image's output has, when the connector fixes it; cached outputs of another shape are misses
    pub rows: Option<usize>,
}

// Encodes the images the encoder cache misses, `encode` taking one image's sub-images and patch mask, and returns the
// stacked outputs of every sub-image with the per-image outputs a packed layout splices.
pub(crate) fn encode_subimages_cached(
    batch: &SubimageBatch<'_>,
    cache: &Mutex<EncoderCacheManager>,
    mut encode: impl FnMut(&Tensor, &Tensor) -> Result<Tensor>,
) -> Result<(Tensor, MultimodalEncoderOutputs)> {
    let SubimageBatch {
        label,
        image_hashes,
        subimage_counts,
        pixel_values,
        patch_attention_mask,
        rows,
    } = *batch;
    let mut offsets = Vec::with_capacity(subimage_counts.len() + 1);
    offsets.push(0usize);
    for &count in subimage_counts {
        if count == 0 {
            inference_tensor::bail!("{label} image has no encoder inputs");
        }
        let next = offsets[offsets.len() - 1]
            .checked_add(count)
            .ok_or_else(|| {
                inference_tensor::Error::msg(format!("{label} subimage count overflow"))
            })?;
        offsets.push(next);
    }
    let total = offsets[offsets.len() - 1];
    if total != pixel_values.dim(0)? {
        inference_tensor::bail!(
            "{label} has {} encoder inputs but subimage counts total {total}",
            pixel_values.dim(0)?
        );
    }
    let fits = |output: &Tensor| {
        rows.is_none_or(|rows| output.rank() == 2 && output.dim(0).ok() == Some(rows))
    };
    let mut per_image: Vec<Option<Vec<Tensor>>> = vec![None; image_hashes.len()];
    {
        let mut guard = cache.lock().expect("encoder cache lock poisoned");
        for (i, &hash) in image_hashes.iter().enumerate() {
            if let Some(cached) = guard.get(CacheModality::Image, hash)
                && cached.len() == subimage_counts[i]
                && cached.iter().all(fits)
            {
                per_image[i] = Some(cached);
            }
        }
    }
    for i in 0..image_hashes.len() {
        if per_image[i].is_some() {
            continue;
        }
        let count = subimage_counts[i];
        let hidden = encode(
            &pixel_values.narrow(0, offsets[i], count)?,
            &patch_attention_mask.narrow(0, offsets[i], count)?,
        )?;
        if hidden.dim(0)? != count || rows.is_some_and(|rows| hidden.dim(1).ok() != Some(rows)) {
            inference_tensor::bail!(
                "{label} encoder returned {:?} for {count} image inputs",
                hidden.dims()
            );
        }
        let outputs = (0..count)
            .map(|index| hidden.get(index))
            .collect::<Result<Vec<_>>>()?;
        cache.lock().expect("encoder cache lock poisoned").insert(
            CacheModality::Image,
            image_hashes[i],
            outputs.clone(),
        );
        per_image[i] = Some(outputs);
    }
    let per_image = per_image.into_iter().flatten().collect::<Vec<_>>();
    let flat = per_image.iter().flatten().cloned().collect::<Vec<_>>();
    let encoder_outputs = image_hashes
        .iter()
        .map(|&hash| MultimodalEncoderKey {
            kind: MultimodalKind::Image,
            hash,
        })
        .zip(per_image)
        .collect::<MultimodalEncoderOutputs>();
    Ok((Tensor::stack(&flat, 0)?, encoder_outputs))
}

// Rescales, normalizes and pads the images to the largest one, with the pixel mask of each image's unpadded area.
pub(crate) fn normalize_and_pad(
    images: &[DynamicImage],
    config: &PreProcessorConfig,
    (default_mean, default_std): ([f64; 3], [f64; 3]),
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let max_h = images
        .iter()
        .map(|image| image.dimensions().1)
        .max()
        .unwrap_or(0) as usize;
    let max_w = images
        .iter()
        .map(|image| image.dimensions().0)
        .max()
        .unwrap_or(0) as usize;
    let rescale = config.do_rescale.is_some_and(|x| x).then_some(Rescale {
        factor: config.rescale_factor,
    });
    let normalize = config.do_normalize.is_some_and(|x| x).then(|| Normalize {
        mean: config.image_mean.unwrap_or(default_mean).to_vec(),
        std: config.image_std.unwrap_or(default_std).to_vec(),
    });
    let mut pixel_values = Vec::with_capacity(images.len());
    let mut patch_masks = Vec::with_capacity(images.len());
    for image in images {
        let transforms = Transforms {
            input: &ToTensorNoNorm,
            inner_transforms: &[&rescale, &normalize],
        };
        let mut image = image.apply(transforms, device)?;
        if config.do_pad.is_some_and(|x| x) {
            let (_c, h, w) = image.dims3()?;
            let padded = inference_vision::pad(&image, max_h, max_w)?;
            patch_masks.push(inference_vision::make_pixel_mask(&padded, h, w)?.unsqueeze(0)?);
            image = padded;
        }
        pixel_values.push(image.unsqueeze(0)?);
    }
    Ok((
        Tensor::cat(&pixel_values, 0)?,
        Tensor::cat(&patch_masks, 0)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CACHE_ENTRIES: usize = 8;
    const SIDE: usize = 2;
    const ROWS: usize = 3;

    // one 1x2x2 sub-image per value, all-zero for a padding sub-image
    fn subimages(values: &[f32]) -> Result<Tensor> {
        let data = values
            .iter()
            .flat_map(|&value| [value; SIDE * SIDE])
            .collect::<Vec<_>>();
        Tensor::from_vec(data, (values.len(), 1, SIDE, SIDE), &Device::Cpu)
    }

    #[test]
    fn padding_subimages_are_dropped_with_their_masks() -> Result<()> {
        let pixels = subimages(&[1.0, 0.0, 2.0])?;
        let mask = Tensor::from_vec(
            vec![1u8, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            (3, SIDE, SIDE),
            &Device::Cpu,
        )?;
        let (kept, kept_mask) = drop_padding_images(&pixels, Some(mask))?;
        assert_eq!(
            kept.flatten_all()?.to_vec1::<f32>()?,
            [1.0; 4]
                .iter()
                .chain(&[2.0; 4])
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            kept_mask.flatten_all()?.to_vec1::<u8>()?,
            [1, 1, 1, 0, 1, 0, 1, 0]
        );
        let (_, all_valid) = drop_padding_images(&pixels, None)?;
        assert_eq!(all_valid.dims(), [2, SIDE, SIDE]);
        assert_eq!(
            all_valid
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()?,
            8.0
        );
        Ok(())
    }

    // Each image's sub-images encode to ROWS rows of their pixel value; a second batch hits the cache for the image
    // it shares and keeps the batch's image order.
    #[test]
    fn cached_images_skip_the_encoder_and_keep_their_order() -> Result<()> {
        let cache = Mutex::new(EncoderCacheManager::new(CACHE_ENTRIES));
        let mut encoded = Vec::new();
        let mut encode = |pv: &Tensor, _mask: &Tensor| -> Result<Tensor> {
            let values = pv.flatten_from(1)?.max(1)?;
            encoded.extend(values.to_vec1::<f32>()?);
            values.reshape(((), 1, 1))?.repeat((1, ROWS, 1))
        };
        let run = |hashes: &[u64],
                   counts: &[usize],
                   values: &[f32],
                   encode: &mut dyn FnMut(&Tensor, &Tensor) -> Result<Tensor>| {
            let pixel_values = subimages(values)?;
            let patch_attention_mask = Tensor::ones((values.len(), 1, 1), DType::U8, &Device::Cpu)?;
            let batch = SubimageBatch {
                label: "test",
                image_hashes: hashes,
                subimage_counts: counts,
                pixel_values: &pixel_values,
                patch_attention_mask: &patch_attention_mask,
                rows: Some(ROWS),
            };
            encode_subimages_cached(&batch, &cache, encode)
        };
        let (_, first) = run(&[10, 20], &[2, 1], &[1.0, 2.0, 3.0], &mut encode)?;
        assert_eq!(first.len(), 2);
        let (hidden, second) = run(&[30, 10], &[1, 2], &[4.0, 1.0, 2.0], &mut encode)?;
        let err = run(&[40], &[2], &[5.0], &mut encode).unwrap_err();
        assert!(err.to_string().contains("subimage counts total 2"), "{err}");
        assert_eq!(encoded, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(second.len(), 2);
        assert_eq!(hidden.dims(), [3, ROWS, 1]);
        assert_eq!(
            hidden.flatten_from(1)?.max(1)?.to_vec1::<f32>()?,
            [4.0, 1.0, 2.0]
        );
        Ok(())
    }
}
