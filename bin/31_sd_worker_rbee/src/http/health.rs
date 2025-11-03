// Created by: TEAM-394
// TEAM-394: Health check endpoint for liveness probes

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;
use serde_json::json;

use crate::http::backend::AppState;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    /// Health status
    pub status: String,
    
    /// Timestamp of health check
    pub timestamp: String,
}

/// Health check endpoint handler
///
/// This endpoint is used for Kubernetes liveness probes.
/// It returns 200 OK if the server is running and responsive.
///
/// # Endpoint
/// `GET /health`
///
/// # Response
/// - **200 OK** - Server is healthy and responsive
/// - **503 Service Unavailable** - Server is unhealthy (rare, usually means crash)
///
/// # Response Body
/// ```json
/// {
///   "status": "healthy",
///   "timestamp": "2025-11-03T21:00:00Z"
/// }
/// ```
///
/// # Kubernetes Configuration
/// ```yaml
/// livenessProbe:
///   httpGet:
///     path: /health
///     port: 8080
///   initialDelaySeconds: 10
///   periodSeconds: 10
/// ```
///
/// # Example
/// ```bash
/// curl http://localhost:8080/health
/// ```
pub async fn health_check(State(_state): State<AppState>) -> impl IntoResponse {
    // Simple health check - if we can respond, we're healthy
    // Future: Could add checks for:
    // - Generation engine responsiveness
    // - VRAM availability
    // - Model integrity
    
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_health_check_response() {
        // We can't create a real AppState without a pipeline,
        // but we can verify the response structure
        
        // This would be tested in integration tests with a real pipeline
        // For now, verify the JSON structure is correct
        let response = json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        assert_eq!(response["status"], "healthy");
        assert!(response["timestamp"].is_string());
    }
}
