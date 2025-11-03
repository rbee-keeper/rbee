// TEAM-390: Stable Diffusion backend implementations
//
// Backend modules for SD inference using Candle.

pub mod model_loader;
pub mod models;

// TEAM-392: Inference pipeline modules
pub mod scheduler;
pub mod sampling;

// TEAM-397: RULE ZERO - Deleted clip.rs, vae.rs, inference.rs (custom wrappers)
// TEAM-397: NEW - Direct Candle usage (no wrappers)
pub mod generation;  // Candle-idiomatic generation functions

// TEAM-393: Generation engine modules
pub mod generation_engine;  // TEAM-396: Added missing module declaration
pub mod request_queue;
pub mod image_utils;

// TEAM-396/397: Public exports
pub use generation_engine::GenerationEngine;
pub use request_queue::{GenerationRequest, GenerationResponse, RequestQueue};

// TEAM-397: RULE ZERO APPLIED
// Removed: clip.rs, vae.rs, inference.rs (custom wrappers - not Candle idiomatic)
// Added: generation.rs (direct Candle functions - idiomatic)
// Pattern: Functions (not structs), direct Candle types (no wrappers)
