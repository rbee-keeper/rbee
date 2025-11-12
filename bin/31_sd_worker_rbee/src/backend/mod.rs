// TEAM-390: Stable Diffusion backend implementations
//
// Backend modules for SD inference using Candle.

// TEAM-397: Core generation modules (Candle-idiomatic, no wrappers)
pub mod generation;
pub mod image_utils;
pub mod lora; // TEAM-487: LoRA support
pub mod model_loader;
pub mod models;

// TEAM-392: Inference pipeline modules
pub mod sampling;
pub mod scheduler;

// TEAM-393: Generation engine modules
pub mod generation_engine;
pub mod request_queue;

// TEAM-396/397: Public exports
pub use generation_engine::GenerationEngine;
pub use request_queue::{GenerationRequest, GenerationResponse, RequestQueue};

// TEAM-397: RULE ZERO APPLIED
// Removed: clip.rs, vae.rs, inference.rs (custom wrappers - not Candle idiomatic)
// Added: generation.rs (direct Candle functions - idiomatic)
// Pattern: Functions (not structs), direct Candle types (no wrappers)
