// TEAM-402: Model entry type
//! Model entry type
//!
//! Pure data type for model artifacts.
//! Migrated from model-catalog/src/types.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tsify::Tsify;

use crate::status::ArtifactStatus;

/// Model entry in the catalog
///
/// Represents a model artifact with metadata.
/// Used by model-catalog, model-provisioner, marketplace-sdk, and UI.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct ModelEntry {
    /// Unique model ID (e.g., "meta-llama/Llama-2-7b")
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Filesystem path to model files
    #[tsify(type = "string")]
    pub path: PathBuf,

    /// Size in bytes
    pub size: u64,

    /// Current status
    pub status: ArtifactStatus,

    /// When the model was added
    #[serde(default = "chrono::Utc::now")]
    pub added_at: chrono::DateTime<chrono::Utc>,
}

impl ModelEntry {
    /// Create a new model entry
    pub fn new(id: String, name: String, path: PathBuf, size: u64) -> Self {
        Self {
            id,
            name,
            path,
            size,
            status: ArtifactStatus::Available,
            added_at: chrono::Utc::now(),
        }
    }

    /// Set the status (for Artifact trait compatibility)
    pub fn set_status(&mut self, status: ArtifactStatus) {
        self.status = status;
    }
}
