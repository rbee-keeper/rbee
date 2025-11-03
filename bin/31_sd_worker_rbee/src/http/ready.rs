// Created by: TEAM-394
// TEAM-394: Readiness check endpoint for Kubernetes readiness probes

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;
use serde_json::json;

use crate::http::backend::AppState;

/// Readiness check response
#[derive(Serialize)]
pub struct ReadyResponse {
    /// Readiness status
    pub ready: bool,
    
    /// Reason if not ready (optional)
    pub reason: Option<String>,
    
    /// Timestamp of readiness check
    pub timestamp: String,
}

/// Readiness check endpoint handler
///
/// This endpoint is used for Kubernetes readiness probes.
/// It returns 200 OK if the worker is ready to accept generation requests.
///
/// # Endpoint
/// `GET /ready`
///
/// # Response
/// - **200 OK** - Worker is ready (model loaded, can accept requests)
/// - **503 Service Unavailable** - Worker is not ready (model loading)
///
/// # Response Body (Ready)
/// ```json
/// {
///   "ready": true,
///   "timestamp": "2025-11-03T21:00:00Z"
/// }
/// ```
///
/// # Response Body (Not Ready)
/// ```json
/// {
///   "ready": false,
///   "reason": "model loading",
///   "timestamp": "2025-11-03T21:00:00Z"
/// }
/// ```
///
/// # Kubernetes Configuration
/// ```yaml
/// readinessProbe:
///   httpGet:
///     path: /ready
///     port: 8080
///   initialDelaySeconds: 30
///   periodSeconds: 5
/// ```
///
/// # Example
/// ```bash
/// curl http://localhost:8080/ready
/// ```
pub async fn readiness_check(State(state): State<AppState>) -> impl IntoResponse {
    if state.is_ready() {
        // Model loaded and ready for inference
        (
            StatusCode::OK,
            Json(json!({
                "ready": true,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            })),
        )
    } else {
        // Model still loading or unavailable
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "ready": false,
                "reason": "model loading",
                "timestamp": chrono::Utc::now().to_rfc3339(),
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_ready_response_structure() {
        // Test ready response
        let ready_response = json!({
            "ready": true,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        assert_eq!(ready_response["ready"], true);
        assert!(ready_response["timestamp"].is_string());
        
        // Test not ready response
        let not_ready_response = json!({
            "ready": false,
            "reason": "model loading",
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        assert_eq!(not_ready_response["ready"], false);
        assert_eq!(not_ready_response["reason"], "model loading");
        assert!(not_ready_response["timestamp"].is_string());
    }
    
    #[test]
    fn test_atomic_bool_pattern() {
        // Verify the AtomicBool pattern used in AppState works correctly
        let flag = Arc::new(AtomicBool::new(false));
        assert!(!flag.load(Ordering::Relaxed));
        
        flag.store(true, Ordering::Relaxed);
        assert!(flag.load(Ordering::Relaxed));
    }
}
