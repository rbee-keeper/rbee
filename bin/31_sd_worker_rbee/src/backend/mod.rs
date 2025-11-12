// TEAM-390: Stable Diffusion backend implementations
//
// Backend modules for SD inference using Candle.

// TEAM-488: Trait-based architecture for clean model abstraction
pub mod traits;

// TEAM-488: Model implementations (self-contained)
pub mod models;

// TEAM-488: Unified infrastructure
pub mod generation_engine;
pub mod model_loader;

pub mod lora; // TEAM-487: LoRA support

// TEAM-392: Inference pipeline modules
pub mod sampling;

// TEAM-481: Modular scheduler architecture
pub mod schedulers;

// TEAM-481: Type-safe IDs
pub mod ids;

// TEAM-488: Keep these for now (used by jobs and other modules)
pub mod image_utils;
pub mod request_queue;

// TEAM-488: Trait exports
pub use traits::{ImageModel, ModelCapabilities};

// TEAM-397: RULE ZERO APPLIED
// Removed: clip.rs, vae.rs, inference.rs (custom wrappers - not Candle idiomatic)
// Added: generation.rs (direct Candle functions - idiomatic)
// Pattern: Functions (not structs), direct Candle types (no wrappers)
