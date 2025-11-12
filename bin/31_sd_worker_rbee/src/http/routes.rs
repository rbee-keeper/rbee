// Created by: TEAM-394
// TEAM-394: HTTP route configuration with middleware stack

use crate::http::backend::AppState;
use crate::http::{health, jobs, ready, stream}; // TEAM-396: Added jobs and stream
use crate::job_router::JobState;
use axum::{
    routing::{get, post},
    Router,
}; // TEAM-396: Added post
use job_server::JobRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower_http::{
    cors::CorsLayer,
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

/// Create HTTP router with all endpoints and middleware
///
/// # Arguments
/// * `state` - Application state (contains generation engine)
///
/// # Returns
/// Router with all endpoints and middleware configured
///
/// # Middleware Stack
/// - **CORS** - Permissive CORS for development (TEAM-396 will tighten)
/// - **Logging** - Request/response tracing
/// - **Timeout** - 300 second timeout for long-running generation
///
/// # Endpoints
/// - `GET /health` - Health check (liveness probe)
/// - `GET /ready` - Readiness check (model loading status)
/// - Future: `POST /v1/jobs` - Job submission (TEAM-395)
/// - Future: `GET /v1/jobs/{job_id}/stream` - SSE streaming (TEAM-395)
///
/// # Example
/// ```no_run
/// # use sd_worker_rbee::http::{backend::AppState, routes::create_router};
/// # use std::sync::Arc;
/// # use sd_worker_rbee::backend::inference::InferencePipeline;
/// # fn example(pipeline: Arc<InferencePipeline>) {
/// let state = AppState::new(pipeline, 10);
/// let router = create_router(state);
/// # }
/// ```
pub fn create_router(state: AppState) -> Router {
    // TEAM-394: Simplified middleware stack (no ServiceBuilder needed)
    // TEAM-396: Added job routes with operations-contract integration

    // Create JobState for job routing
    // TEAM-396: JobRegistry uses GenerationResponse (not TokenResponse like LLM)
    let registry =
        Arc::new(JobRegistry::<crate::backend::request_queue::GenerationResponse>::new());
    let job_state = JobState { registry, queue: state.request_queue().clone() };

    Router::new()
        // Health and readiness endpoints
        .route("/health", get(health::health_check))
        .route("/ready", get(ready::readiness_check))
        .with_state(state)
        // Job endpoints (operations-contract integration)
        // TEAM-396: Same endpoints as LLM worker!
        .route("/v1/jobs", post(jobs::handle_create_job))
        .route("/v1/jobs/:job_id/stream", get(stream::handle_stream_job))
        .with_state(job_state)
        // Middleware layers (applied in reverse order)
        .layer(TimeoutLayer::new(Duration::from_secs(300)))
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
                .on_response(DefaultOnResponse::new().level(Level::INFO)),
        )
        .layer(CorsLayer::permissive())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        // We can't create a real AppState without a pipeline,
        // but we can verify the router creation pattern compiles

        // This would be tested in integration tests with a real pipeline
        // For now, just verify the function signature is correct
    }
}
