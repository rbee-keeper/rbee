// TEAM-396: SSE streaming endpoint
//
// Matches LLM worker pattern: bin/30_llm_worker_rbee/src/http/stream.rs

use crate::backend::request_queue::GenerationResponse;
use crate::job_router::JobState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use std::convert::Infallible;

/// Handle SSE streaming via GET /`v1/jobs/{job_id}/stream`
///
/// TEAM-396: Streams generation progress and results
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<JobState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // Take receiver from registry
    let mut response_rx = state.registry.take_token_receiver(&job_id).ok_or_else(|| {
        (StatusCode::NOT_FOUND, format!("Job {job_id} not found or already streaming"))
    })?;

    // Stream events
    let stream = async_stream::stream! {
        while let Some(response) = response_rx.recv().await {
            match response {
                GenerationResponse::Progress { step, total } => {
                    let json = serde_json::json!({
                        "type": "progress",
                        "step": step,
                        "total": total,
                        "percent": (step as f32 / total as f32) * 100.0,
                    });
                    yield Ok(Event::default()
                        .event("progress")
                        .data(json.to_string()));
                }
                GenerationResponse::Preview { step, total, image } => {
                    // TEAM-487: Send preview image as base64 via SSE
                    let base64 = match crate::backend::image_utils::image_to_base64(&image) {
                        Ok(b64) => b64,
                        Err(e) => {
                            tracing::warn!(error = %e, "Failed to encode preview image");
                            continue; // Skip this preview, don't break the stream
                        }
                    };

                    let json = serde_json::json!({
                        "type": "preview",
                        "step": step,
                        "total": total,
                        "percent": (step as f32 / total as f32) * 100.0,
                        "image": base64,
                        "format": "png",
                    });
                    yield Ok(Event::default()
                        .event("preview")
                        .data(json.to_string()));
                }
                GenerationResponse::Complete { image } => {
                    // Convert image to base64
                    let base64 = match crate::backend::image_utils::image_to_base64(&image) {
                        Ok(b64) => b64,
                        Err(e) => {
                            let error_json = serde_json::json!({
                                "type": "error",
                                "message": format!("Failed to encode image: {}", e),
                            });
                            yield Ok(Event::default()
                                .event("error")
                                .data(error_json.to_string()));
                            break;
                        }
                    };

                    let json = serde_json::json!({
                        "type": "complete",
                        "image": base64,
                        "format": "png",
                    });
                    yield Ok(Event::default()
                        .event("complete")
                        .data(json.to_string()));
                }
                GenerationResponse::Error { message } => {
                    let json = serde_json::json!({
                        "type": "error",
                        "message": message,
                    });
                    yield Ok(Event::default()
                        .event("error")
                        .data(json.to_string()));
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_endpoint_compiles() {
        // Verify the endpoint signature is correct
        // Actual testing requires full integration
    }
}
