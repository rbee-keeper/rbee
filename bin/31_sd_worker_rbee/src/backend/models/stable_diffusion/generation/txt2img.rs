// TEAM-488: Text-to-image generation
// 
// Generates images from text prompts using Stable Diffusion

use super::super::ModelComponents;
use super::helpers::{text_embeddings, tensor_to_image};
use crate::backend::traits::GenerationRequest;
use crate::error::Result;
use candle_core::{Module, Tensor};
use image::DynamicImage;

pub fn txt2img<F>(
    components: &ModelComponents,
    request: &GenerationRequest,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    if let Some(seed) = request.seed {
        components.device.set_seed(seed)?;
    }

    let use_guide_scale = request.guidance_scale > 1.0;

    // Generate text embeddings (Candle function)
    let text_embeddings = text_embeddings(
        &request.prompt,
        request.negative_prompt.as_deref().unwrap_or(""),
        &components.tokenizer,
        &components.clip_config,
        &components.clip_weights,
        &components.device,
        components.dtype,
        use_guide_scale,
    )?;

    // Initialize latents
    let latent_height = request.height / 8;
    let latent_width = request.width / 8;
    let bsize = 1;

    let mut latents =
        Tensor::randn(0f32, 1.0, (bsize, 4, latent_height, latent_width), &components.device)?
            .to_dtype(components.dtype)?;

    // Diffusion loop (direct Candle calls)
    let timesteps = components.scheduler.timesteps();
    let num_steps = timesteps.len();

    for (step_idx, &timestep) in timesteps.iter().enumerate() {
        let latent_model_input =
            if use_guide_scale { Tensor::cat(&[&latents, &latents], 0)? } else { latents.clone() };

        // Direct UNet forward call (Candle idiom)
        let noise_pred =
            components.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

            let guidance = request.guidance_scale;
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };

        // Direct scheduler step (Candle idiom)
        latents = components.scheduler.step(&noise_pred, timestep, &latents)?;

        // TEAM-487: Generate preview image every 5 steps
        if step_idx % 5 == 0 || step_idx == num_steps - 1 {
            let preview_images = components.vae.decode(&(&latents / components.vae_scale)?)?;
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
    let images = components.vae.decode(&(latents / components.vae_scale)?)?;

    // Convert tensor to image (from reference example)
    let image = tensor_to_image(&images)?;

    Ok(image)
}
