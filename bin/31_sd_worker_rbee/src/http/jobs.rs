// TEAM-396: Job submission endpoint (operations-contract integration)
//
// Matches LLM worker pattern: bin/30_llm_worker_rbee/src/http/jobs.rs

use crate::job_router::{create_job, JobResponse, JobState};
use axum::{extract::State, http::StatusCode, Json};

/// Handle job creation via POST /v1/jobs
///
/// TEAM-396: Uses operations-contract (Operation enum)
/// Accepts any Operation variant, routes to appropriate handler
pub async fn handle_create_job(
    State(state): State<JobState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    create_job(state, payload).await.map(Json).map_err(|e| {
        tracing::error!(error = %e, "Job creation failed");
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_endpoint_compiles() {
        // Verify the endpoint signature is correct
        // Actual testing requires operations-contract integration
    }
}
