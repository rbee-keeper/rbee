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
/// Worker catalog entry types
pub mod worker_catalog;
/// Artifact status type
pub mod status;
/// NSFW filtering levels (TEAM-464)
pub mod nsfw;
/// Marketplace filter types (TEAM-464)
pub mod filters;

// Re-export main types
// TEAM-407: Added ModelMetadata exports
// TEAM-463: Added ModelFile and CivitAI types export
pub use model::{
    ModelEntry, ModelType, ModelSource, ModelFile,
    ModelConfig, InferenceParams,
    LlmConfig, TokenizerConfig,
    ImageConfig, CheckpointType, ImagePreview,
    ModelArchitecture, ModelFormat, Quantization, ModelMetadata,
    CivitaiModel, CivitaiModelVersion, CivitaiStats, CivitaiCreator,
    CivitaiFile, CivitaiImage,
};
pub use worker::{Platform, WorkerBinary, WorkerType};
pub use worker_catalog::{
    Architecture, WorkerImplementation, BuildSystem,
    SourceInfo, BuildConfig, WorkerCatalogEntry,
};
pub use status::ArtifactStatus;
pub use nsfw::{NsfwLevel, NsfwFilter};
pub use filters::{
    TimePeriod, CivitaiModelType, BaseModel, CivitaiSort, HuggingFaceSort,
    CivitaiFilters, HuggingFaceFilters,
};
