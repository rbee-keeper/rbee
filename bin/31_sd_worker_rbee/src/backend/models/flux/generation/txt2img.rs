// TEAM-488: FLUX text-to-image generation
//
// Generates images from text prompts using FLUX.1-dev or FLUX.1-schnell
// Based on: reference/candle/candle-examples/examples/flux/main.rs

use super::super::ModelComponents;
use super::helpers::{clip_embeddings, t5_embeddings, tensor_to_image};
use crate::backend::traits::GenerationRequest;
use crate::error::Result;
use candle_transformers::models::flux;
use image::DynamicImage;

/// Generate image from text prompt using FLUX
///
/// TEAM-488: FLUX generation pipeline
/// Based on: reference/candle/candle-examples/examples/flux/main.rs lines 104-232
pub fn txt2img<F>(
    components: &mut ModelComponents,
    request: &GenerationRequest,
    mut progress_callback: F,
) -> Result<DynamicImage>
where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    if let Some(seed) = request.seed {
        components.device.set_seed(seed)?;
    }

    let width = request.width;
    let height = request.height;
    let steps = request.steps;
    let guidance = request.guidance_scale;

    // 1. Generate T5-XXL text embeddings
    tracing::info!("Generating T5 embeddings...");
    let t5_emb = t5_embeddings(
        &request.prompt,
        &components.t5_tokenizer,
        &mut components.t5_model,
        &components.device,
    )?;

    // 2. Generate CLIP text embeddings
    tracing::info!("Generating CLIP embeddings...");
    let clip_emb = clip_embeddings(
        &request.prompt,
        &components.clip_tokenizer,
        &components.clip_model,
        &components.device,
    )?;

    // 3. Initialize noise
    tracing::info!("Initializing noise...");
    let img = flux::sampling::get_noise(1, height, width, &components.device)?
        .to_dtype(components.dtype)?;

    // 4. Create sampling state
    let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;

    // 5. Get timestep schedule and start denoising
    tracing::info!("Starting denoising loop ({} steps)...", steps);
    progress_callback(0, steps, None);

    let mut img = state.img.clone();
    let timesteps = match components.version {
        crate::backend::models::SDVersion::FluxDev => {
            // Dev uses time shift for better quality
            flux::sampling::get_schedule(steps, Some((state.img.dim(1)?, 0.5, 1.15)))
        }
        crate::backend::models::SDVersion::FluxSchnell => {
            // Schnell uses simple schedule
            flux::sampling::get_schedule(steps, None)
        }
        _ => unreachable!("Non-FLUX model passed to FLUX generation"),
    };

    // Denoise with intermediate previews
    let b_sz = img.dim(0)?;
    let guidance_tensor = candle_core::Tensor::full(guidance as f32, b_sz, &components.device)?;

    for (step_idx, window) in timesteps.windows(2).enumerate() {
        let (t_curr, t_prev) = match window {
            [a, b] => (a, b),
            _ => continue,
        };

        let t_vec = candle_core::Tensor::full(*t_curr as f32, b_sz, &components.device)?;
        let pred = components.flux_model_mut().forward(
            &img,
            &state.img_ids,
            &state.txt,
            &state.txt_ids,
            &t_vec,
            &state.vec,
            Some(&guidance_tensor),
        )?;
        img = (img + pred * (t_prev - t_curr))?;

        // Send progress with preview every 5 steps
        if step_idx % 5 == 0 || step_idx == steps - 1 {
            // Unpack and decode for preview
            let preview_img = flux::sampling::unpack(&img, height, width)?;
            let preview_decoded = components.vae.decode(&preview_img)?;

            match tensor_to_image(&preview_decoded) {
                Ok(preview) => progress_callback(step_idx + 1, steps, Some(preview)),
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to generate preview image");
                    progress_callback(step_idx + 1, steps, None);
                }
            }
        } else {
            progress_callback(step_idx + 1, steps, None);
        }
    }

    // 7. Final unpack and decode
    tracing::info!("Unpacking final latents...");
    let img = flux::sampling::unpack(&img, height, width)?;

    tracing::info!("Decoding final image with VAE...");
    let img = components.vae.decode(&img)?;

    // 8. Convert to image
    let image = tensor_to_image(&img)?;

    tracing::info!("FLUX generation complete");
    Ok(image)
}
