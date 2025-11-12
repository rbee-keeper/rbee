// TEAM-483: FLUX image generation
//
// Implements FLUX.1-dev and FLUX.1-schnell generation pipeline.
// Based on: reference/candle/candle-examples/examples/flux/main.rs

use crate::error::{Error, Result};
use candle_core::{DType, IndexOp, Module, Tensor};
use candle_transformers::models::flux;
use image::DynamicImage;

use super::models::flux_loader::FluxComponents;

/// FLUX generation configuration
pub struct FluxConfig {
    pub prompt: String,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub guidance_scale: f64,
    pub seed: Option<u64>,
}

impl FluxConfig {
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.width % 8 != 0 {
            return Err(Error::InvalidInput(format!(
                "Width must be divisible by 8, got {}",
                self.width
            )));
        }
        if self.height % 8 != 0 {
            return Err(Error::InvalidInput(format!(
                "Height must be divisible by 8, got {}",
                self.height
            )));
        }
        if self.prompt.is_empty() {
            return Err(Error::InvalidInput("Prompt cannot be empty".to_string()));
        }
        Ok(())
    }
}

/// Generate image with FLUX
///
/// # Arguments
/// * `config` - Generation configuration (prompt, size, steps, etc.)
/// * `models` - Loaded FLUX model components
/// * `progress_callback` - Optional callback for progress updates (step, total_steps)
///
/// # Returns
/// Generated image as `DynamicImage`
///
/// # Errors
/// Returns error if generation fails or config is invalid
pub fn generate_flux<F>(
    config: &FluxConfig,
    models: &FluxComponents,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize),
{
    config.validate()?;
    
    // Set seed if provided
    if let Some(seed) = config.seed {
        models
            .device
            .set_seed(seed)
            .map_err(|e| Error::Generation(format!("Failed to set seed: {}", e)))?;
    }
    
    tracing::info!(
        "Starting FLUX generation: {}x{}, {} steps",
        config.width,
        config.height,
        config.steps
    );
    
    // 1. Encode text with T5-XXL
    tracing::debug!("Encoding text with T5-XXL...");
    let t5_tokens = models
        .t5_tokenizer
        .encode(&config.prompt, true)
        .map_err(|e| Error::Generation(format!("T5 tokenization failed: {}", e)))?;
    
    let mut t5_token_ids = t5_tokens.get_ids().to_vec();
    // Pad to 256 tokens (FLUX requirement)
    t5_token_ids.resize(256, 0);
    
    let t5_token_ids = Tensor::new(&t5_token_ids[..], &models.device)
        .map_err(|e| Error::Generation(format!("Failed to create T5 token tensor: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| Error::Generation(format!("Failed to unsqueeze T5 tokens: {}", e)))?;
    
    let t5_emb = models
        .t5_model
        .forward(&t5_token_ids)
        .map_err(|e| Error::Generation(format!("T5 forward pass failed: {}", e)))?;
    
    // 2. Encode text with CLIP
    tracing::debug!("Encoding text with CLIP...");
    let clip_tokens = models
        .clip_tokenizer
        .encode(&config.prompt, true)
        .map_err(|e| Error::Generation(format!("CLIP tokenization failed: {}", e)))?;
    
    let clip_token_ids = Tensor::new(clip_tokens.get_ids(), &models.device)
        .map_err(|e| Error::Generation(format!("Failed to create CLIP token tensor: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| Error::Generation(format!("Failed to unsqueeze CLIP tokens: {}", e)))?;
    
    let clip_emb = models
        .clip_model
        .forward(&clip_token_ids)
        .map_err(|e| Error::Generation(format!("CLIP forward pass failed: {}", e)))?;
    
    // 3. Initialize noise
    tracing::debug!("Initializing noise...");
    let img = flux::sampling::get_noise(1, config.height, config.width, &models.device)
        .map_err(|e| Error::Generation(format!("Failed to generate noise: {}", e)))?
        .to_dtype(models.dtype)
        .map_err(|e| Error::Generation(format!("Failed to convert noise dtype: {}", e)))?;
    
    // 4. Create sampling state
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)
        .map_err(|e| Error::Generation(format!("Failed to create sampling state: {}", e)))?;
    
    // 5. Get timestep schedule
    let timesteps = match models.version {
        super::models::SDVersion::FluxDev => {
            // Dev: uses shift for better quality
            let img_dim = state
                .img
                .dim(1)
                .map_err(|e| Error::Generation(format!("Failed to get img dim: {}", e)))?;
            flux::sampling::get_schedule(config.steps, Some((img_dim, 0.5, 1.15)))
        }
        super::models::SDVersion::FluxSchnell => {
            // Schnell: no shift, just linear schedule
            flux::sampling::get_schedule(config.steps, None)
        }
        _ => {
            return Err(Error::InvalidInput(format!(
                "Expected FLUX model, got {:?}",
                models.version
            )))
        }
    };
    
    tracing::info!("Running denoising loop with {} timesteps", timesteps.len());
    
    // 6. Denoising loop
    let img = flux::sampling::denoise(
        models.flux_model.as_ref(),
        &state.img,
        &state.img_ids,
        &state.txt,
        &state.txt_ids,
        &state.vec,
        &timesteps,
        config.guidance_scale as f32,
    )
    .map_err(|e| Error::Generation(format!("Denoising failed: {}", e)))?;
    
    // Report progress (denoising complete)
    progress_callback(config.steps, config.steps);
    
    // 7. Unpack latents
    tracing::debug!("Unpacking latents...");
    let img = flux::sampling::unpack(&img, config.height, config.width)
        .map_err(|e| Error::Generation(format!("Failed to unpack latents: {}", e)))?;
    
    // 8. Decode with VAE
    tracing::debug!("Decoding with VAE...");
    let img = models
        .vae
        .decode(&img)
        .map_err(|e| Error::Generation(format!("VAE decode failed: {}", e)))?;
    
    // 9. Convert to image
    tracing::debug!("Converting to image...");
    
    // Clamp and scale to [0, 255]
    let img = img
        .clamp(-1f32, 1f32)
        .map_err(|e| Error::Generation(format!("Failed to clamp: {}", e)))?;
    let img = ((img + 1.0)
        .map_err(|e| Error::Generation(format!("Failed to add: {}", e)))?
        * 127.5)
        .map_err(|e| Error::Generation(format!("Failed to multiply: {}", e)))?;
    let img = img
        .to_dtype(DType::U8)
        .map_err(|e| Error::Generation(format!("Failed to convert to U8: {}", e)))?;
    
    // Move to CPU
    let img = img
        .to_device(&candle_core::Device::Cpu)
        .map_err(|e| Error::Generation(format!("Failed to move to CPU: {}", e)))?;
    
    // Extract first batch item
    let img = img
        .i(0)
        .map_err(|e| Error::Generation(format!("Failed to index batch: {}", e)))?;
    
    // Get dimensions (C, H, W)
    let (channels, height, width) = img
        .dims3()
        .map_err(|e| Error::Generation(format!("Failed to get dims: {}", e)))?;
    
    if channels != 3 {
        return Err(Error::Generation(format!(
            "Expected 3 channels, got {}",
            channels
        )));
    }
    
    // Convert tensor to image
    tensor_to_image(&img, width, height)
}

/// Convert Candle tensor to image::DynamicImage
///
/// Tensor shape: (C, H, W) where C=3 (RGB)
fn tensor_to_image(tensor: &Tensor, width: usize, height: usize) -> Result<DynamicImage> {
    // Get raw data
    let data = tensor
        .to_vec3::<u8>()
        .map_err(|e| Error::Generation(format!("Failed to convert tensor to vec: {}", e)))?;
    
    // Flatten from (C, H, W) to (H, W, C)
    let mut img_data = vec![0u8; width * height * 3];
    for h in 0..height {
        for w in 0..width {
            for c in 0..3 {
                img_data[(h * width + w) * 3 + c] = data[c][h][w];
            }
        }
    }
    
    // Create image
    let img = image::RgbImage::from_raw(width as u32, height as u32, img_data)
        .ok_or_else(|| Error::Generation("Failed to create image from raw data".to_string()))?;
    
    Ok(DynamicImage::ImageRgb8(img))
}
