// TEAM-402: Artifact status type
//! Artifact status type
//!
//! Pure data type for artifact status.
//! Migrated from artifact-catalog/src/types.rs

use serde::{Deserialize, Serialize};
use tsify::Tsify;

/// Artifact status
///
/// Represents the current state of an artifact (model or worker).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum ArtifactStatus {
    /// Artifact is available and ready to use
    Available,

    /// Artifact is currently being downloaded/provisioned
    Downloading,

    /// Artifact download/provisioning failed
    Failed {
        /// Error message
        error: String,
    },
}
