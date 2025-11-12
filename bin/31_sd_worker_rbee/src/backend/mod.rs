// TEAM-390: Stable Diffusion backend implementations
//
// Backend modules for SD inference using Candle.

// TEAM-488: Trait-based architecture for clean model abstraction
pub mod traits;

// TEAM-488: Model implementations (self-contained)
pub mod models;

// TEAM-488: OLD FILES DELETED - generation.rs, flux_generation.rs, generation_engine.rs, model_loader.rs
// Generation logic will be in models/stable_diffusion/generator.rs and models/flux/generator.rs
// Loading logic will be in models/stable_diffusion/loader.rs and models/flux/loader.rs

pub mod lora; // TEAM-487: LoRA support

// TEAM-392: Inference pipeline modules
pub mod sampling;
pub mod scheduler;

// TEAM-488: Keep these for now (used by jobs and other modules)
pub mod request_queue;
pub mod image_utils;

// TEAM-488: Trait exports
pub use traits::{ImageModel, ModelCapabilities};

// TEAM-397: RULE ZERO APPLIED
// Removed: clip.rs, vae.rs, inference.rs (custom wrappers - not Candle idiomatic)
// Added: generation.rs (direct Candle functions - idiomatic)
// Pattern: Functions (not structs), direct Candle types (no wrappers)
