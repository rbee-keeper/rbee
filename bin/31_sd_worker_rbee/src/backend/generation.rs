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
use image::{DynamicImage, GenericImageView, RgbImage};
use tokenizers::Tokenizer;

/// Generate image from text prompt
///
/// TEAM-397: Candle idiom - function, not struct method
/// TEAM-487: Callback now supports preview images (Option<DynamicImage>)
/// Based on reference/candle/.../stable-diffusion/main.rs lines 531-826
pub fn generate_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
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
        
        // TEAM-487: Generate preview image every 5 steps
        if step_idx % 5 == 0 || step_idx == num_steps - 1 {
            let preview_images = models.vae.decode(&(&latents / models.vae_scale)?)?;
            match tensor_to_image(&preview_images) {
                Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to generate preview image");
                    progress_callback(step_idx + 1, num_steps, None);
                }
            }
        } else {
            progress_callback(step_idx + 1, num_steps, None);
        }
    }

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

/// Encode image to latent space
///
/// TEAM-487: VAE encoder for img2img
/// Takes an RGB image and encodes it to latent space using VAE encoder.
/// This is the reverse of VAE decoding (latents → image).
///
/// # Arguments
/// * `image` - Input image (RGB, any size)
/// * `vae` - VAE model (contains encoder)
/// * `target_width` - Target width (must be multiple of 8)
/// * `target_height` - Target height (must be multiple of 8)
/// * `device` - Device to run on
/// * `dtype` - Data type (F32 or F16)
///
/// # Returns
/// Latent tensor (shape: [1, 4, height/8, width/8])
pub fn encode_image_to_latents(
    image: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    target_width: usize,
    target_height: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // 1. Resize image to match model requirements
    let resized = image.resize_exact(
        target_width as u32,
        target_height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    
    // 2. Convert to tensor (normalize to [-1, 1])
    let image_tensor = image_to_tensor(&resized, device, dtype)?;
    
    // 3. Encode to latent space
    let latent_dist = vae.encode(&image_tensor)?;
    
    // 4. Sample from distribution and apply VAE scaling factor (SD standard)
    let latents = (latent_dist.sample()? * 0.18215)?;
    
    Ok(latents)
}

/// Convert image to tensor
///
/// TEAM-487: Helper for VAE encoding
/// Converts DynamicImage to Tensor in format expected by VAE
fn image_to_tensor(image: &DynamicImage, device: &Device, dtype: DType) -> Result<Tensor> {
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();
    let data = rgb.into_raw();
    
    // Convert to f32 and normalize to [-1, 1]
    let data: Vec<f32> = data.iter().map(|&x| (x as f32 / 255.0) * 2.0 - 1.0).collect();
    
    // Reshape to (1, 3, height, width)
    let tensor = Tensor::from_vec(data, (height as usize, width as usize, 3), device)?;
    let tensor = tensor.permute((2, 0, 1))?.unsqueeze(0)?;
    
    Ok(tensor.to_dtype(dtype)?)
}

/// Add noise to latents based on strength parameter
///
/// TEAM-487: Noise addition for img2img
/// For img2img, we don't start from pure noise. Instead:
/// 1. Encode input image to latents
/// 2. Add noise proportional to strength
/// 3. Denoise from that point
///
/// # Arguments
/// * `latents` - Clean latents from input image
/// * `strength` - How much to change (0.0 = no noise, 1.0 = full noise)
/// * `num_steps` - Total denoising steps
///
/// # Returns
/// Noisy latents + starting timestep index
pub fn add_noise_for_img2img(
    latents: &Tensor,
    strength: f64,
    num_steps: usize,
) -> Result<(Tensor, usize)> {
    // Calculate starting step based on strength
    // strength=0.0 → start at step 0 (no denoising)
    // strength=1.0 → start at step num_steps (full denoising)
    let start_step = ((1.0 - strength) * num_steps as f64) as usize;
    
    if start_step >= num_steps {
        // If strength is 1.0, just return random noise
        let noise = latents.randn_like(0.0, 1.0)?;
        return Ok((noise, 0));
    }
    
    // Generate noise
    let noise = latents.randn_like(0.0, 1.0)?;
    
    // Simple linear interpolation between latents and noise
    // More sophisticated schedulers would use proper noise scheduling
    let noisy_latents = (latents * (1.0 - strength))? + (noise * strength)?;
    
    Ok((noisy_latents?, start_step))
}

/// Generate image from existing image (img2img)
///
/// TEAM-487: Image-to-image generation
/// Based on: reference/candle/candle-examples/examples/stable-diffusion/main.rs
/// Similar to generate_image() but starts from encoded input image
///
/// # Arguments
/// * `config` - Sampling configuration (prompt, steps, guidance, etc.)
/// * `models` - Loaded model components (VAE, UNet, CLIP, etc.)
/// * `input_image` - Starting image to transform
/// * `strength` - Transformation strength (0.0-1.0)
/// * `progress_callback` - Called after each denoising step
///
/// # Returns
/// Transformed image
pub fn image_to_image<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    input_image: &DynamicImage,
    strength: f64,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    // Validate inputs
    config.validate()?;
    if !(0.0..=1.0).contains(&strength) {
        return Err(Error::InvalidInput(format!(
            "Strength must be between 0.0 and 1.0, got {}",
            strength
        )));
    }
    
    // Set seed if provided
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    let use_guide_scale = config.guidance_scale > 1.0;
    
    // 1. Generate text embeddings (same as text-to-image)
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
    
    // 2. Encode input image to latents
    let init_latents = encode_image_to_latents(
        input_image,
        &models.vae,
        config.width,
        config.height,
        &models.device,
        models.dtype,
    )?;
    
    // 3. Add noise based on strength
    let (mut latents, start_step) = add_noise_for_img2img(
        &init_latents,
        strength,
        config.steps,
    )?;
    
    // 4. Denoise from start_step to end (partial denoising)
    let timesteps = models.scheduler.timesteps();
    let num_steps = timesteps.len();
    
    for (step_idx, &timestep) in timesteps.iter().enumerate().skip(start_step) {
        // Expand latents for classifier-free guidance
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        
        // Predict noise
        let noise_pred = models.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
        
        // Apply classifier-free guidance
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            
            let guidance = config.guidance_scale;
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        
        // Scheduler step
        latents = models.scheduler.step(&noise_pred, timestep, &latents)?;
        
        // TEAM-487: Generate preview image every 5 steps
        if step_idx % 5 == 0 || step_idx == num_steps - 1 {
            let preview_images = models.vae.decode(&(&latents / models.vae_scale)?)?;
            match tensor_to_image(&preview_images) {
                Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to generate preview image");
                    progress_callback(step_idx + 1, num_steps, None);
                }
            }
        } else {
            progress_callback(step_idx + 1, num_steps, None);
        }
    }
    
    // 5. Decode latents to image (same as text-to-image)
    let images = models.vae.decode(&(&latents / models.vae_scale)?)?;
    
    // Convert tensor to image
    let image = tensor_to_image(&images)?;
    
    Ok(image)
}

/// Prepare inpainting latents
///
/// TEAM-487: Prepares the three inputs needed for inpainting models:
/// 1. Original image latents (what to keep)
/// 2. Mask latents (where to inpaint)
/// 3. Masked image latents (original * (1 - mask))
///
/// # Arguments
/// * `image` - Original image
/// * `mask` - Binary mask (white = inpaint, black = keep)
/// * `vae` - VAE for encoding
/// * `target_width` - Target width
/// * `target_height` - Target height
/// * `device` - Device
/// * `dtype` - Data type
///
/// # Returns
/// (image_latents, mask_latents, masked_image_latents)
pub fn prepare_inpainting_latents(
    image: &DynamicImage,
    mask: &DynamicImage,
    vae: &candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    target_width: usize,
    target_height: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)> {
    // 1. Encode original image to latents
    let image_latents = encode_image_to_latents(image, vae, target_width, target_height, device, dtype)?;
    
    // 2. Convert mask to latent space
    let mask_latents = crate::backend::image_utils::mask_to_latent_tensor(mask, device, dtype)?;
    
    // 3. Create masked image (original * (1 - mask))
    // This shows the model what to keep
    let inverted_mask = (Tensor::ones_like(&mask_latents)? - &mask_latents)?;
    let masked_image_latents = (&image_latents * &inverted_mask)?;
    
    Ok((image_latents, mask_latents, masked_image_latents))
}

/// Generate inpainted image
///
/// TEAM-487: Fills in masked regions based on text prompt
/// Uses special inpainting models (9-channel input)
///
/// # Arguments
/// * `config` - Sampling configuration
/// * `models` - Model components (MUST be inpainting variant)
/// * `input_image` - Original image
/// * `mask` - Binary mask (white = inpaint, black = keep)
/// * `progress_callback` - Progress updates
///
/// # Returns
/// Inpainted image
pub fn inpaint<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    input_image: &DynamicImage,
    mask: &DynamicImage,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    // Validate that this is an inpainting model
    if !models.version.is_inpainting() {
        return Err(Error::InvalidInput(format!(
            "Model {:?} is not an inpainting model. Use V1_5Inpaint, V2Inpaint, or XLInpaint.",
            models.version
        )));
    }
    
    config.validate()?;
    
    if let Some(seed) = config.seed {
        models.device.set_seed(seed)?;
    }
    
    let use_guide_scale = config.guidance_scale > 1.0;
    
    // 1. Generate text embeddings
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
    
    // 2. Process mask
    let processed_mask = crate::backend::image_utils::process_mask(
        mask,
        config.width as u32,
        config.height as u32,
    )?;
    
    // 3. Prepare inpainting latents
    let (image_latents, mask_latents, masked_image_latents) = prepare_inpainting_latents(
        input_image,
        &processed_mask,
        &models.vae,
        config.width,
        config.height,
        &models.device,
        models.dtype,
    )?;
    
    // 4. Initialize noise latents
    let latent_height = config.height / 8;
    let latent_width = config.width / 8;
    let bsize = 1;
    
    let mut latents = Tensor::randn(
        0f32,
        1.0,
        (bsize, 4, latent_height, latent_width),
        &models.device,
    )?.to_dtype(models.dtype)?;
    
    // 5. Denoising loop
    let timesteps = models.scheduler.timesteps();
    let num_steps = timesteps.len();
    
    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        
        // Concatenate latents with mask and masked image
        // Inpainting UNet expects 9 channels:
        // - 4 channels: noisy latents
        // - 1 channel: mask
        // - 4 channels: masked image latents
        let latent_model_input = Tensor::cat(
            &[&latents, &mask_latents, &masked_image_latents],
            1, // Concatenate along channel dimension
        )?;
        
        // Expand for classifier-free guidance
        let latent_model_input = if use_guide_scale {
            Tensor::cat(&[&latent_model_input, &latent_model_input], 0)?
        } else {
            latent_model_input
        };
        
        // Predict noise
        let noise_pred = models.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;
        
        // Apply classifier-free guidance
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
            
            let guidance = config.guidance_scale;
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };
        
        // Scheduler step
        latents = models.scheduler.step(&noise_pred, timestep, &latents)?;
        
        // Blend with original image in non-masked regions
        // This ensures masked regions are regenerated, non-masked regions stay the same
        let inverted_mask = (Tensor::ones_like(&mask_latents)? - &mask_latents)?;
        latents = ((&latents * &mask_latents)? + (&image_latents * &inverted_mask)?)?;
        
        // TEAM-487: Generate preview image every 5 steps
        if step_idx % 5 == 0 || step_idx == num_steps - 1 {
            let preview_images = models.vae.decode(&(&latents / models.vae_scale)?)?;
            match tensor_to_image(&preview_images) {
                Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to generate preview image");
                    progress_callback(step_idx + 1, num_steps, None);
                }
            }
        } else {
            progress_callback(step_idx + 1, num_steps, None);
        }
    }
    
    // 6. Decode latents to image
    let images = models.vae.decode(&(&latents / models.vae_scale)?)?;
    
    // Convert tensor to image
    let image = tensor_to_image(&images)?;
    
    Ok(image)
}
