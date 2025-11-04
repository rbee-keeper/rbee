// TEAM-273: Shared artifact types
// TEAM-402: Re-export types from artifacts-contract and implement Artifact trait
use serde::{Deserialize, Serialize};
use std::path::Path;

// Re-export from artifacts-contract
pub use artifacts_contract::{ArtifactStatus, ModelEntry, WorkerBinary};

/// Core artifact trait
///
/// Implemented by ModelEntry, WorkerBinary, etc.
pub trait Artifact: Clone + Serialize + for<'de> Deserialize<'de> {
    /// Unique identifier for this artifact
    fn id(&self) -> &str;

    /// Filesystem path to the artifact
    fn path(&self) -> &Path;

    /// Size in bytes
    fn size(&self) -> u64;

    /// Current status
    fn status(&self) -> &ArtifactStatus;

    /// Set status (mutable)
    fn set_status(&mut self, status: ArtifactStatus);

    /// Human-readable name
    fn name(&self) -> &str {
        self.id()
    }
}

/// Metadata for filesystem-based catalogs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMetadata<T> {
    /// The artifact itself
    pub artifact: T,

    /// When it was added to catalog
    pub added_at: chrono::DateTime<chrono::Utc>,

    /// Last accessed time (optional)
    pub last_accessed: Option<chrono::DateTime<chrono::Utc>>,
}

impl<T> ArtifactMetadata<T> {
    /// Create new metadata
    pub fn new(artifact: T) -> Self {
        Self { artifact, added_at: chrono::Utc::now(), last_accessed: None }
    }

    /// Update last accessed time
    pub fn touch(&mut self) {
        self.last_accessed = Some(chrono::Utc::now());
    }
}

// TEAM-402: Implement Artifact trait for ModelEntry
impl Artifact for ModelEntry {
    fn id(&self) -> &str {
        &self.id
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn status(&self) -> &ArtifactStatus {
        &self.status
    }

    fn set_status(&mut self, status: ArtifactStatus) {
        self.set_status(status);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// TEAM-402: Implement Artifact trait for WorkerBinary
impl Artifact for WorkerBinary {
    fn id(&self) -> &str {
        &self.id
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn size(&self) -> u64 {
        self.size
    }

    fn status(&self) -> &ArtifactStatus {
        &self.status
    }

    fn set_status(&mut self, status: ArtifactStatus) {
        self.status = status;
    }

    fn name(&self) -> &str {
        &self.id
    }
}
