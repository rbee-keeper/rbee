// TEAM-390: Stable Diffusion backend implementations
//
// Backend modules for SD inference using Candle.

pub mod model_loader;
pub mod models;

// TEAM-392: Inference pipeline modules
pub mod clip;
pub mod vae;
pub mod scheduler;
pub mod sampling;
pub mod inference;

// TEAM-393: Generation engine modules
pub mod generation_engine;  // TEAM-396: Added missing module declaration
pub mod request_queue;
pub mod image_utils;

// TEAM-396: Public exports
pub use generation_engine::GenerationEngine;
pub use inference::InferencePipeline;
pub use request_queue::{GenerationRequest, GenerationResponse, RequestQueue};

// TEAM-396: Removed old CandleSDBackend and SDBackend trait
// These used the old request types (TextToImageRequest, etc.)
// New pattern: InferencePipeline handles generation, called by GenerationEngine
// See: src/backend/inference.rs and src/backend/generation_engine.rs
