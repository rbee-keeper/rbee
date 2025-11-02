//! Job creation and streaming HTTP endpoints
//!
//! TEAM-261: Mirrors queen-rbee pattern for consistency
//!
//! This module is a thin HTTP wrapper that delegates to job_router for business logic.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
use rbee_hive_model_catalog::ModelCatalog; // TEAM-268: Model catalog
use rbee_hive_model_provisioner::ModelProvisioner; // Model provisioner for downloads
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use std::{convert::Infallible, sync::Arc};

/// State for HTTP job endpoints
#[derive(Clone)]
pub struct HiveState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    /// Model catalog for model management
    pub model_catalog: Arc<ModelCatalog>, // TEAM-268: Added
    /// Model provisioner for downloading models
    pub model_provisioner: Arc<ModelProvisioner>,
    /// Worker catalog for worker binary management
    pub worker_catalog: Arc<WorkerCatalog>, // TEAM-274: Added
}

/// Convert HTTP state to router state
impl From<HiveState> for crate::job_router::JobState {
    fn from(state: HiveState) -> Self {
        Self {
            registry: state.registry,
            model_catalog: state.model_catalog, // TEAM-268: Added
            model_provisioner: state.model_provisioner,
            worker_catalog: state.worker_catalog, // TEAM-274: Added
        }
    }
}

/// POST /v1/jobs - Create a new job (ALL operations)
///
/// TEAM-261: Thin HTTP wrapper that delegates to job_router::create_job()
///
/// Example payloads (from queen-rbee):
/// - {"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu", "device": 0}
/// - {"operation": "model_download", "hive_id": "localhost", "model": "..."}
/// - {"operation": "infer", "hive_id": "localhost", "model": "...", "prompt": "...", ...}
pub async fn handle_create_job(
    State(state): State<HiveState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<crate::job_router::JobResponse>, (StatusCode, String)> {
    // Delegate to router
    crate::job_router::create_job(state.into(), payload)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// DELETE /v1/jobs/{job_id} - Cancel a job
///
/// TEAM-305-FIX: Allow users to cancel running or queued jobs
///
/// Returns:
/// - 200 OK with job_id if cancelled successfully
/// - 404 NOT FOUND if job doesn't exist or cannot be cancelled
pub async fn handle_cancel_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // TEAM-388: Add narration to debug cancellation
    observability_narration_core::n!("cancel_request", "🛑 Received cancel request for job {}", job_id);
    
    let cancelled = state.registry.cancel_job(&job_id);

    if cancelled {
        observability_narration_core::n!("cancel_success", "✅ Job {} cancellation token triggered", job_id);
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        observability_narration_core::n!("cancel_failed", "❌ Job {} not found or cannot be cancelled", job_id);
        Err((
            StatusCode::NOT_FOUND,
            format!("Job {} not found or cannot be cancelled", job_id),
        ))
    }
}

/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
///
/// TEAM-305-FIX: Mirrors queen-rbee pattern
///
/// This handler:
/// 1. Takes the job-specific SSE receiver (MPSC - can only be done once)
/// 2. Triggers job execution (which emits narrations)
/// 3. Streams narration events back to queen-rbee
/// 4. Sends [DONE] marker when complete
/// 5. When receiver drops, sender fails gracefully (natural cleanup)
///
/// Client (queen-rbee) connects here, which triggers job execution and streams results.
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Take the receiver FIRST (can only be done once per job)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);

    // TEAM-384: Get registry before moving state
    let registry = state.registry.clone();
    let job_id_for_state = job_id.clone();

    // Trigger job execution (spawns in background)
    // The return value is a token stream for job-server internal use - we don't need it
    // The SSE channel closing IS our completion signal
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;

    // Single stream that handles both error and success cases
    let job_id_for_stream = job_id.clone();
    let combined_stream = async_stream::stream! {
        // Check if channel exists
        let Some(mut sse_rx) = sse_rx_opt else {
            // TEAM-384: Send error AND [DONE] marker so frontend doesn't hang
            yield Ok(Event::default().data("ERROR: Job channel not found. This may indicate a race condition or job creation failure."));
            yield Ok(Event::default().data("[DONE]"));
            return;
        };

        // TEAM-384: Stream ALL SSE events (narration + data) from channel
        // When channel closes, check job state and send appropriate completion signal
        
        use observability_narration_core::SseEvent;
        while let Some(event) = sse_rx.recv().await {
            match event {
                SseEvent::Narration(n) => {
                    // Send narration as plain text (backward compatible)
                    yield Ok(Event::default().data(&n.formatted));
                }
                SseEvent::Data(d) => {
                    // Send data as JSON with event type marker
                    let json = serde_json::to_string(&d).unwrap_or_else(|_| "{}".to_string());
                    yield Ok(Event::default().event("data").data(&json));
                }
                SseEvent::Done => {
                    // Job marked as done
                    break;
                }
            }
        }
        
        // TEAM-384: SSE channel closed - job completed!
        // Check final job state to determine success/failure/cancelled
        use job_server::JobState;
        let state = registry.get_job_state(&job_id_for_state);
        let signal = match state {
            Some(JobState::Failed(err)) => format!("[ERROR] {}", err),
            Some(JobState::Cancelled) => "[CANCELLED]".to_string(),
            _ => "[DONE]".to_string(),
        };
        yield Ok(Event::default().data(&signal));

        // Cleanup - remove sender from HashMap to prevent memory leak
        // Receiver is already dropped by moving out of this scope
        sse_sink::remove_job_channel(&job_id_for_stream);
    };

    Sse::new(combined_stream)
}
