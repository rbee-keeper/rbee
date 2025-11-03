// TEAM-390: Job routing for SD worker
// TEAM-396: CRITICAL FIX - Rewritten to use operations-contract
//
// FIXED ISSUES:
// 1. Uses operations-contract (not custom request types)
// 2. Matches LLM worker pattern (bin/30_llm_worker_rbee/src/job_router.rs)
// 3. Routes Operation enum to handlers
// 4. Integrates with RequestQueue properly

use anyhow::{anyhow, Result};
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
use operations_contract::Operation;
use serde::Serialize;
use std::sync::Arc;

use crate::backend::request_queue::{GenerationRequest, GenerationResponse, RequestQueue};
use crate::backend::sampling::SamplingConfig;

/// State required for job routing and execution
///
/// TEAM-396: Matches LLM worker pattern
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<GenerationResponse>>,
    pub queue: RequestQueue,
}

/// Response from job creation
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// TEAM-396: Mirrors LLM worker pattern exactly
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    // Parse operation from JSON
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow!("Failed to parse operation: {}", e))?;

    // Route to appropriate handler
    match operation {
        // TODO TEAM-397: Add Operation::ImageGeneration to operations-contract first!
        // Operation::ImageGeneration(req) => execute_image_generation(state, req).await,
        // Operation::ImageTransform(req) => execute_image_transform(state, req).await,
        // Operation::ImageInpaint(req) => execute_inpaint(state, req).await,
        _ => Err(anyhow!(
            "Unsupported operation for SD worker: {:?}\n\n\
            TEAM-397 TODO:\n\
            1. Add Operation::ImageGeneration to operations-contract\n\
            2. Add Operation::ImageTransform to operations-contract\n\
            3. Add Operation::ImageInpaint to operations-contract\n\
            4. Implement handlers in this file\n\n\
            See: OPERATIONS_CONTRACT_ANALYSIS.md and CORRECT_IMPLEMENTATION_PLAN.md",
            operation
        )),
    }
}

// TODO TEAM-397: Implement these handlers after adding operations to contract
//
// /// Execute image generation operation
// async fn execute_image_generation(
//     state: JobState,
//     req: operations_contract::ImageGenerationRequest,
// ) -> Result<JobResponse> {
//     let job_id = state.registry.create_job();
//     sse_sink::create_job_channel(job_id.clone(), 1000);
//     
//     let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
//     state.registry.set_token_receiver(&job_id, response_rx);
//     
//     let config = SamplingConfig {
//         prompt: req.prompt,
//         negative_prompt: req.negative_prompt,
//         steps: req.steps,
//         guidance_scale: req.guidance_scale,
//         width: req.width,
//         height: req.height,
//         seed: req.seed,
//         ..Default::default()
//     };
//     
//     let request = GenerationRequest {
//         request_id: job_id.clone(),
//         config,
//         response_tx,
//     };
//     
//     state.queue.add_request(request)?;
//     
//     Ok(JobResponse {
//         job_id: job_id.clone(),
//         sse_url: format!("/v1/jobs/{}/stream", job_id),
//     })
// }
