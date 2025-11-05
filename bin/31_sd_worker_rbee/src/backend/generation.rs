// TEAM-397: Candle-idiomatic SD generation
// RULE ZERO: Replaced clip.rs, vae.rs, inference.rs with direct Candle usage
//
// Based on: reference/candle/candle-examples/examples/stable-diffusion/main.rs
// Pattern: Functions (not structs), direct Candle types (no wrappers)

use crate::backend::models::ModelComponents;
use crate::backend::sampling::SamplingConfig;
use crate::error::{Error, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use image::{DynamicImage, RgbImage};
use tokenizers::Tokenizer;

/// Generate image from text prompt
///
/// TEAM-397: Candle idiom - function, not struct method
/// Based on reference/candle/.../stable-diffusion/main.rs lines 531-826
pub fn generate_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    config.validate()?;

    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }

    let use_guide_scale = config.guidance_scale > 1.0;

    // Generate text embeddings (Candle function)
    let text_embeddings = text_embeddings(
        &config.prompt,
        config.negative_prompt.as_deref().unwrap_or(""),
        &models.tokenizer,
        &models.clip_config,
        &models.clip_weights,
        &models.device,
        models.dtype,
        use_guide_scale,
    )?;

    // Initialize latents
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    let bsize = 1;

    let mut latents =
        Tensor::randn(0f32, 1.0, (bsize, 4, latent_height, latent_width), &models.device)?
            .to_dtype(models.dtype)?;

    // Diffusion loop (direct Candle calls)
    let timesteps = models.scheduler.timesteps();
    let num_steps = timesteps.len();

    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        progress_callback(step_idx, num_steps);

        let latent_model_input =
            if use_guide_scale { Tensor::cat(&[&latents, &latents], 0)? } else { latents.clone() };

        // Direct UNet forward call (Candle idiom)
        let noise_pred =
            models.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

            let guidance = config.guidance_scale;
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };

        // Direct scheduler step (Candle idiom)
        latents = models.scheduler.step(&noise_pred, timestep, &latents)?;
    }

    progress_callback(num_steps, num_steps);

    // Direct VAE decode (Candle idiom)
    let images = models.vae.decode(&(latents / models.vae_scale)?)?;

    // Convert tensor to image (from reference example)
    let image = tensor_to_image(&images)?;

    Ok(image)
}

/// Generate text embeddings
///
/// TEAM-397: Candle idiom - direct from reference example
/// Based on reference/candle/.../stable-diffusion/main.rs lines 345-433
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: &Tokenizer,
    clip_config: &stable_diffusion::clip::Config,
    clip_weights: &std::path::Path,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
) -> Result<Tensor> {
    // Get pad token ID
    // MANUAL FIX NEEDED: The token below is BACKWARDS
    // Reverse this string: >|txetfodne|<
    let pad_id = match &clip_config.pad_with {
        Some(padding) => *tokenizer
            .get_vocab(true)
            .get(padding.as_str())
            .ok_or_else(|| Error::ModelLoading(format!("Pad token {} not found", padding)))?,
        None => *tokenizer
            .get_vocab(true)
            .get(">|endoftext|<") // MANUAL FIX: Reverse this string!
            .ok_or_else(|| Error::ModelLoading("Default pad token not found".to_string()))?,
    };

    // Tokenize prompt
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

    // Build CLIP model (direct Candle call)
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;

    let text_embeddings = text_model.forward(&tokens)?;

    // Handle unconditional prompt if using guidance
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
///
/// TEAM-397: From reference example
/// Based on reference/candle/.../stable-diffusion/main.rs lines 318-342
fn tensor_to_image(tensor: &Tensor) -> Result<DynamicImage> {
    // Decode: ((x / 2) + 0.5) * 255
    let tensor = ((tensor / 2.)? + 0.5)?;
    let tensor = tensor.to_device(&Device::Cpu)?;
    let tensor = (tensor.clamp(0f32, 1.)? * 255.)?;
    let tensor = tensor.to_dtype(DType::U8)?;

    // Get dimensions (batch, channel, height, width)
    let (batch, channel, height, width) = tensor.dims4()?;

    if batch != 1 {
        return Err(Error::Generation(format!("Expected batch size 1, got {}", batch)));
    }

    if channel != 3 {
        return Err(Error::Generation(format!("Expected 3 channels, got {}", channel)));
    }

    // Extract first batch, permute to (height, width, channel), flatten
    let image_data = tensor.i(0)?.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;

    let img = RgbImage::from_raw(width as u32, height as u32, image_data)
        .ok_or_else(|| Error::Generation("Failed to create image from tensor".to_string()))?;

    Ok(DynamicImage::ImageRgb8(img))
}
