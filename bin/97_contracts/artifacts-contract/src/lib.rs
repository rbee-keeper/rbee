// TEAM-402: Artifact types contract
// TEAM-405: Enhanced with trait-based model system
//! Artifact types contract
//!
//! Pure data types for models and workers.
//! Shared across catalogs, provisioners, marketplace, and UI.
//!
//! This crate contains ONLY pure data types with no business logic.
//! It depends ONLY on serde, chrono, tsify, and wasm-bindgen.

#![warn(missing_docs)]

/// Model types with trait-based configuration
pub mod model;
/// Worker binary type
pub mod worker;
/// Artifact status type
pub mod status;

// Re-export main types
// TEAM-407: Added ModelMetadata exports
pub use model::{
    ModelEntry, ModelType, ModelSource,
    ModelConfig, InferenceParams,
    LlmConfig, TokenizerConfig,
    ImageConfig, CheckpointType, ImagePreview,
    ModelArchitecture, ModelFormat, Quantization, ModelMetadata,
};
pub use worker::{Platform, WorkerBinary, WorkerType};
pub use status::ArtifactStatus;
