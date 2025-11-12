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
use operations_contract::Operation;
use std::sync::Arc;

use crate::backend::request_queue::{GenerationResponse, RequestQueue};

/// State required for job routing and execution
///
/// TEAM-396: Matches LLM worker pattern
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<GenerationResponse>>,
    pub queue: RequestQueue,
}

// TEAM-487: JobResponse is shared across all handlers
pub use crate::jobs::JobResponse;

/// Create a new job and store its payload
///
/// TEAM-396: Mirrors LLM worker pattern exactly
/// TEAM-487: Simplified to just route to handlers in jobs/ folder
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    // Parse operation from JSON
    let operation: Operation =
        serde_json::from_value(payload).map_err(|e| anyhow!("Failed to parse operation: {e}"))?;

    // Route to appropriate handler (TEAM-487: handlers in jobs/ folder)
    match operation {
        Operation::ImageGeneration(req) => crate::jobs::image_generation::execute(state, req),
        Operation::ImageTransform(req) => crate::jobs::image_transform::execute(state, req),
        Operation::ImageInpaint(req) => crate::jobs::image_inpaint::execute(state, req),
        _ => Err(anyhow!("Unsupported operation for SD worker: {operation:?}")),
    }
}

// TEAM-487: All job handlers moved to jobs/ folder
// - jobs/image_generation.rs
// - jobs/image_transform.rs
// - jobs/image_inpaint.rs
