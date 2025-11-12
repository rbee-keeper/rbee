// TEAM-488: Helper functions for SD generation
// Shared utilities used across txt2img, img2img, and inpaint

use crate::error::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use image::{DynamicImage, RgbImage};
use tokenizers::Tokenizer;

/// Generate text embeddings
///
/// TEAM-397: Candle idiom - direct from reference example
/// Based on reference/candle/.../stable-diffusion/main.rs lines 345-433
pub(super) fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: &Tokenizer,
    clip_config: &stable_diffusion::clip::Config,
    clip_weights: &std::path::Path,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer
            .get_vocab(true)
            .get(padding.as_str())
            .ok_or_else(|| Error::ModelLoading(format!("Pad token {} not found", padding)))?,
        None => *tokenizer
            .get_vocab(true)
            .get("</|endoftext|>")
            .ok_or_else(|| Error::ModelLoading("Default pad token not found".to_string()))?,
    };

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| Error::ModelLoading(format!("Tokenization failed: {}", e)))?
        .get_ids()
        .to_vec();

    if tokens.len() > clip_config.max_position_embeddings {
        return Err(Error::InvalidInput(format!(
            "Prompt too long: {} > {}",
            tokens.len(),
            clip_config.max_position_embeddings
        )));
    }

    while tokens.len() < clip_config.max_position_embeddings {
        tokens.push(pad_id);
    }

    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(|e| Error::ModelLoading(format!("Tokenization failed: {}", e)))?
            .get_ids()
            .to_vec();

        if uncond_tokens.len() > clip_config.max_position_embeddings {
            return Err(Error::InvalidInput(format!(
                "Negative prompt too long: {} > {}",
                uncond_tokens.len(),
                clip_config.max_position_embeddings
            )));
        }

        while uncond_tokens.len() < clip_config.max_position_embeddings {
            uncond_tokens.push(pad_id);
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };

    Ok(text_embeddings)
}

/// Convert tensor to image
pub(super) fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    let tensor = ((tensor / 2.)? + 0.5)?;
    let tensor = tensor.to_device(&Device::Cpu)?;
    let tensor = (tensor.clamp(0f32, 1.)? * 255.)?;
    let tensor = tensor.to_dtype(DType::U8)?;

    let (batch, channel, height, width) = tensor.dims4()?;

    if batch != 1 {
        return Err(Error::Generation(format!("Expected batch size 1, got {}", batch)));
    }

    if channel != 3 {
        return Err(Error::Generation(format!("Expected 3 channels, got {}", channel)));
    }

    let image_data = tensor.i(0)?.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;

    let img = RgbImage::from_raw(width as u32, height as u32, image_data)
        .ok_or_else(|| Error::Generation("Failed to create image from tensor".to_string()))?;

    Ok(DynamicImage::ImageRgb8(img))
}

/// Encode image to latent space
pub fn encode_image_to_latents(
    image: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    target_width: usize,
    target_height: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let resized = image.resize_exact(
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let tensor = image_to_tensor(&resized, device, dtype)?;
    let dist = vae.encode(&tensor)?;
    Ok(dist.sample()?)
}

fn image_to_tensor(image: &DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let data: Vec<f32> = rgb
        .pixels()
        .flat_map(|p| {
            let r = p[0] as f32 / 255.0;
            let g = p[1] as f32 / 255.0;
            let b = p[2] as f32 / 255.0;
            [r * 2.0 - 1.0, g * 2.0 - 1.0, b * 2.0 - 1.0]
        })
        .collect();

    let tensor = Tensor::from_vec(data, (height as usize, width as usize, 3), device)?;
    let tensor = tensor.permute((2, 0, 1))?.unsqueeze(0)?;

    Ok(tensor.to_dtype(dtype)?)
}

pub(super) fn add_noise_for_img2img(
    latents: &Tensor,
    strength: f64,
    num_steps: usize,
) -> Result<(Tensor, usize)> {
    let start_step = ((1.0 - strength) * num_steps as f64) as usize;

    if start_step >= num_steps {
        let noise = Tensor::randn(0f32, 1.0, latents.shape(), latents.device())?;
        return Ok((noise, 0));
    }

    let noise = Tensor::randn(0f32, 1.0, latents.shape(), latents.device())?;
    let noisy_latents = ((latents * (1.0 - strength))? + (noise * strength)?)?;

    Ok((noisy_latents, start_step))
}

/// Prepare inpainting latents
pub fn prepare_inpainting_latents(
    image: &DynamicImage,
    mask: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    target_width: usize,
    target_height: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    let image_latents =
        encode_image_to_latents(image, vae, target_width, target_height, device, dtype)?;

    let mask_latents = crate::backend::image_utils::mask_to_latent_tensor(mask, device, dtype)?;

    let masked_image_latents =
        (image_latents.clone() * (Tensor::ones_like(&mask_latents)? - &mask_latents)?)?;

    Ok((image_latents, mask_latents, masked_image_latents))
}
