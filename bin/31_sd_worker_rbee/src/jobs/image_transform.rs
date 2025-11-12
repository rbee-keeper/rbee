// TEAM-487: Image transform job handler (img2img)

use anyhow::{anyhow, Result};
use image::GenericImageView;
use operations_contract::ImageTransformRequest;

use crate::backend::request_queue::GenerationRequest;
use crate::backend::sampling::SamplingConfig;
use crate::job_router::JobState;
use crate::jobs::JobResponse;

/// Execute image transform operation (img2img)
///
/// TEAM-487: Full implementation with VAE encoding and noise addition
pub fn execute(state: JobState, req: ImageTransformRequest) -> Result<JobResponse> {
    // 1. Decode base64 input image
    let input_image = crate::backend::image_utils::base64_to_image(&req.init_image)
        .map_err(|e| anyhow!("Failed to decode input image: {}", e))?;

    // 2. Get image dimensions for config
    let (img_width, img_height) = input_image.dimensions();

    // 3. Create sampling config
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        seed: req.seed,
        width: img_width as usize,
        height: img_height as usize,
    };

    // 4. Create job and SSE sink
    let job_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    state.registry.set_token_receiver(&job_id, rx);

    // 5. Submit to generation queue
    let request = GenerationRequest {
        request_id: job_id.clone(),
        config,
        input_image: Some(input_image),  // TEAM-487: Enable img2img
        mask: None,                      // TEAM-487: No mask for img2img
        strength: req.strength,          // TEAM-487: From request (has default)
        response_tx: tx,
    };

    state.queue.add_request(request).map_err(|e| anyhow!("Failed to queue request: {}", e))?;

    Ok(JobResponse { job_id: job_id.clone(), sse_url: format!("/v1/jobs/{}/stream", job_id) })
}
