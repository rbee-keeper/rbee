// TEAM-487: Image generation job handler (text-to-image)
// TEAM-481: Added tracing instrumentation for better observability

use anyhow::{anyhow, Result};
use operations_contract::ImageGenerationRequest;

use crate::backend::request_queue::GenerationRequest;
use crate::backend::sampling::SamplingConfig;
use crate::job_router::JobState;
use crate::jobs::JobResponse;

/// Execute image generation operation (text-to-image)
/// 
/// TEAM-481: Instrumented for tracing - automatically logs function entry/exit
#[tracing::instrument(skip(state), fields(job_id))]
pub fn execute(state: JobState, req: ImageGenerationRequest) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    tracing::Span::current().record("job_id", &job_id);
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);
    
    let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
    state.registry.set_token_receiver(&job_id, response_rx);
    
    let config = SamplingConfig {
        prompt: req.prompt,
        negative_prompt: req.negative_prompt,
        steps: req.steps,
        guidance_scale: req.guidance_scale,
        seed: req.seed,
        width: req.width,
        height: req.height,
        sampler: crate::backend::schedulers::SamplerType::default(),     // TEAM-482: Use default sampler (Euler)
        schedule: crate::backend::schedulers::NoiseSchedule::default(),  // TEAM-482: Use default schedule (Simple)
        loras: req.loras.iter().map(|l| crate::backend::lora::LoRAConfig {
            path: l.path.clone(),
            strength: l.strength as f64,
        }).collect(),  // TEAM-488: LoRA support wired up!
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
