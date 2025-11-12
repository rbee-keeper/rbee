// TEAM-487: Image inpaint job handler (inpainting)

use anyhow::{anyhow, Result};
use image::GenericImageView;
use operations_contract::ImageInpaintRequest;

use crate::backend::request_queue::GenerationRequest;
use crate::backend::sampling::SamplingConfig;
use crate::job_router::JobState;
use crate::jobs::JobResponse;

/// Execute inpaint operation
///
/// TEAM-487: Full implementation with mask processing and 9-channel UNet input
pub fn execute(state: JobState, req: ImageInpaintRequest) -> Result<JobResponse> {
    // 1. Decode base64 input image
    let input_image = crate::backend::image_utils::base64_to_image(&req.init_image)
        .map_err(|e| anyhow!("Failed to decode input image: {}", e))?;
    
    // 2. Decode base64 mask image
    let mask_image = crate::backend::image_utils::base64_to_image(&req.mask_image)
        .map_err(|e| anyhow!("Failed to decode mask image: {}", e))?;
    
    // 3. Get image dimensions for config
    let (img_width, img_height) = input_image.dimensions();
    
    // 4. Create sampling config
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        seed: req.seed,
        width: img_width as usize,
        height: img_height as usize,
        loras: vec![],  // TEAM-487: No LoRAs for now
    };
    
    // 5. Create job and SSE sink
    let job_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    
    state.registry.set_token_receiver(&job_id, rx);
    
    // 6. Submit to generation queue
    let request = GenerationRequest {
        request_id: job_id.clone(),
        config,
        input_image: Some(input_image),  // TEAM-487: Enable inpainting
        mask: Some(mask_image),          // TEAM-487: Mask for inpainting
        strength: 0.0,                   // TEAM-487: Not used for inpainting
        response_tx: tx,
    };
    
    state.queue.add_request(request)
        .map_err(|e| anyhow!("Failed to queue request: {}", e))?;
    
    Ok(JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    })
}
