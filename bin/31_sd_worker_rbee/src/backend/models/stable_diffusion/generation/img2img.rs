// TEAM-488: Image-to-image generation
//
// Transforms existing images using text prompts

use super::super::ModelComponents;
use super::helpers::{
    add_noise_for_img2img, encode_image_to_latents, tensor_to_image, text_embeddings,
};
use crate::backend::traits::GenerationRequest;
use crate::error::{Error, Result};
use candle_core::Tensor;
use image::DynamicImage;

pub fn img2img<F>(
    components: &ModelComponents,
    request: &GenerationRequest,
    input_image: &DynamicImage,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    // Validate inputs
    if !(0.0..=1.0).contains(&request.strength) {
        return Err(Error::InvalidInput(format!(
            "Strength must be between 0.0 and 1.0, got {}",
            request.strength
        )));
    }

    // Set seed if provided
    if let Some(seed) = request.seed {
        components.device.set_seed(seed)?;
    }

    let use_guide_scale = request.guidance_scale > 1.0;

    // 1. Generate text embeddings (same as text-to-image)
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

    // 2. Encode input image to latents
    let init_latents = encode_image_to_latents(
        input_image,
        &components.vae,
        request.width,
        request.height,
        &components.device,
        components.dtype,
    )?;

    // 3. Add noise based on request.strength
    let (mut latents, start_step) =
        add_noise_for_img2img(&init_latents, request.strength, request.steps)?;

    // 4. Denoise from start_step to end (partial denoising)
    let timesteps = components.scheduler.timesteps();
    let num_steps = timesteps.len();

    for (step_idx, &timestep) in timesteps.iter().enumerate().skip(start_step) {
        // Expand latents for classifier-free guidance
        let latent_model_input =
            if use_guide_scale { Tensor::cat(&[&latents, &latents], 0)? } else { latents.clone() };

        // Predict noise
        let noise_pred =
            components.unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

        // Apply classifier-free guidance
        let noise_pred = if use_guide_scale {
            let noise_pred = noise_pred.chunk(2, 0)?;
            let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

            let guidance = request.guidance_scale;
            (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance)?)?
        } else {
            noise_pred
        };

        // Scheduler step
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

    // 5. Decode latents to image (same as text-to-image)
    let images = components.vae.decode(&(&latents / components.vae_scale)?)?;

    // Convert tensor to image
    let image = tensor_to_image(&images)?;

    Ok(image)
}
