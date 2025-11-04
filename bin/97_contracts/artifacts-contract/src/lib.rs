// TEAM-402: Artifact types contract
//! Artifact types contract
//!
//! Pure data types for models and workers.
//! Shared across catalogs, provisioners, marketplace, and UI.
//!
//! This crate contains ONLY pure data types with no business logic.
//! It depends ONLY on serde, chrono, tsify, and wasm-bindgen.

#![warn(missing_docs)]

/// Model entry type
pub mod model;
/// Worker binary type
pub mod worker;
/// Artifact status type
pub mod status;

// Re-export main types
pub use model::ModelEntry;
pub use worker::{Platform, WorkerBinary, WorkerType};
pub use status::ArtifactStatus;
