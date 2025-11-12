// TEAM-487: Image generation job handler (text-to-image)

use anyhow::{anyhow, Result};
use operations_contract::ImageGenerationRequest;

use crate::backend::request_queue::GenerationRequest;
use crate::backend::sampling::SamplingConfig;
use crate::job_router::JobState;
use crate::jobs::JobResponse;

/// Execute image generation operation (text-to-image)
pub fn execute(state: JobState, req: ImageGenerationRequest) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
    state.registry.set_token_receiver(&job_id, response_rx);
    
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        width: req.width,
        height: req.height,
        seed: req.seed,
        ..Default::default()
    };
    
    let request = GenerationRequest {
        request_id: job_id.clone(),
        config,
        input_image: None,  // TEAM-487: Text-to-image (no input image)
        mask: None,         // TEAM-487: No mask for text-to-image
        strength: 0.8,      // TEAM-487: Not used for text-to-image
        response_tx,
    };
    
    state.queue.add_request(request)
        .map_err(|e| anyhow!("Failed to add request to queue: {}", e))?;
    
    Ok(JobResponse {
        job_id: job_id.clone(),
        sse_url: format!("/v1/jobs/{}/stream", job_id),
    })
}
