// TEAM-488: Helper functions for SD generation
// Shared utilities used across txt2img, img2img, and inpaint
// TEAM-482: Uses shared helpers to avoid duplication with FLUX

use crate::backend::models::shared::{image_to_tensor, tensor_to_image_sd};
use crate::error::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use image::{DynamicImage, RgbImage};
use tokenizers::Tokenizer;

/// Parameters for text embedding generation
///
/// TEAM-482: Groups related parameters to avoid `too_many_arguments` warning
pub(super) struct TextEmbeddingParams<'a> {
    pub prompt: &'a str,
    pub uncond_prompt: &'a str,
    pub tokenizer: &'a Tokenizer,
    pub clip_config: &'a stable_diffusion::clip::Config,
    pub clip_weights: &'a std::path::Path,
    pub device: &'a Device,
    pub dtype: DType,
    pub use_guide_scale: bool,
}

/// Generate text embeddings
///
/// TEAM-397: Candle idiom - direct from reference example
/// Based on reference/candle/.../stable-diffusion/main.rs lines 345-433
/// TEAM-482: Uses parameter struct to avoid `too_many_arguments`
pub(super) fn text_embeddings(params: &TextEmbeddingParams<'_>) -> Result<Tensor> {
    let TextEmbeddingParams {
        prompt,
        uncond_prompt,
        tokenizer,
        clip_config,
        clip_weights,
        device,
        dtype,
        use_guide_scale,
    } = *params;
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer
            .get_vocab(true)
            .get(padding.as_str())
            .ok_or_else(|| Error::ModelLoading(format!("Pad token {padding} not found")))?,
        None => *tokenizer
            .get_vocab(true)
            .get("</|endoftext|>")
            .ok_or_else(|| Error::ModelLoading("Default pad token not found".to_string()))?,
    };

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(|e| Error::ModelLoading(format!("Tokenization failed: {e}")))?
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
            .map_err(|e| Error::ModelLoading(format!("Tokenization failed: {e}")))?
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
///
/// TEAM-482: Delegates to shared helper to avoid duplication
pub(super) fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    tensor_to_image_sd(tensor)
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

// TEAM-482: Removed - now using shared::image_to_tensor

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
