// TEAM-390: HTTP server for SD worker
//
// Provides REST API endpoints for image generation.
// Placeholder for future implementation.

use axum::{
    routing::{get, post},
    Router,
};

/// Health check handler
async fn health() -> &'static str {
    "OK"
}

/// Text-to-image endpoint (placeholder)
async fn text_to_image() -> &'static str {
    "Text-to-image endpoint (not yet implemented)"
}

/// Image-to-image endpoint (placeholder)
async fn image_to_image() -> &'static str {
    "Image-to-image endpoint (not yet implemented)"
}

/// Inpainting endpoint (placeholder)
async fn inpaint() -> &'static str {
    "Inpainting endpoint (not yet implemented)"
}

/// Create the HTTP router
pub fn create_router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/execute", post(text_to_image))
        .route("/img2img", post(image_to_image))
        .route("/inpaint", post(inpaint))
}
