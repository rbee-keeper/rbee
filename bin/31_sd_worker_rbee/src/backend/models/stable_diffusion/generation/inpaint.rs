// TEAM-488: Inpainting generation
//
// Fills masked regions of images using text prompts

use super::super::ModelComponents;
use super::helpers::{text_embeddings, tensor_to_image, prepare_inpainting_latents};
use crate::backend::traits::GenerationRequest;
use crate::error::{Error, Result};
use candle_core::{Module, Tensor};
use image::DynamicImage;

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
pub fn inpaint<F>(
    components: &ModelComponents,
    request: &GenerationRequest,
    input_image: &DynamicImage,
    mask: &DynamicImage,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    // Validate that this is an inpainting model
    if !components.version.is_inpainting() {
        return Err(Error::InvalidInput(format!(
            "Model {:?} is not an inpainting model. Use V1_5Inpaint, V2Inpaint, or XLInpaint.",
            components.version
        )));
    }

    if let Some(seed) = request.seed {
        components.device.set_seed(seed)?;
    }

    let use_guide_scale = request.guidance_scale > 1.0;

    // 1. Generate text embeddings
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

    // 2. Process mask
    let processed_mask = crate::backend::image_utils::process_mask(
        mask,
        request.width as u32,
        request.height as u32,
    )?;

    // 3. Prepare inpainting latents
    let (image_latents, mask_latents, masked_image_latents) = prepare_inpainting_latents(input_image, &processed_mask, &components.vae, request.width, request.height, &components.device, components.dtype)?;

    // 4. Initialize noise latents
    let latent_height = request.height / 8;
    let latent_width = request.width / 8;
    let bsize = 1;

    let mut latents =
        Tensor::randn(0f32, 1.0, (bsize, 4, latent_height, latent_width), &components.device)?
            .to_dtype(components.dtype)?;

    // 5. Denoising loop
    let timesteps = components.scheduler.timesteps();
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

        // Blend with original image in non-masked regions
        // This ensures masked regions are regenerated, non-masked regions stay the same
        let inverted_mask = (Tensor::ones_like(&mask_latents)? - &mask_latents)?;
        latents = ((&latents * &mask_latents)? + (&image_latents * &inverted_mask)?)?;

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

    // 6. Decode latents to image
    let images = components.vae.decode(&(&latents / components.vae_scale)?)?;

    // Convert tensor to image
    let image = tensor_to_image(&images)?;

    Ok(image)
}
