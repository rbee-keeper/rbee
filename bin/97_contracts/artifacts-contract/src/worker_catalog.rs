// TEAM-408: Worker catalog entry types
//! Worker catalog entry types
//!
//! Represents workers available in the catalog for download/installation.
//! This is DIFFERENT from WorkerBinary which represents installed workers.

use serde::{Deserialize, Serialize};
use tsify::Tsify;

use crate::{Platform, WorkerType};

/// CPU Architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    /// x86_64 (AMD64)
    X86_64,
    /// ARM64 (aarch64)
    Aarch64,
}

impl Architecture {
    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
        }
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Worker implementation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum WorkerImplementation {
    /// Rust implementation
    Rust,
    /// Python implementation
    Python,
    /// C++ implementation
    Cpp,
}

/// Build system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum BuildSystem {
    /// Cargo (Rust)
    Cargo,
    /// Make
    Make,
    /// CMake
    Cmake,
}

/// Source repository information
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct SourceInfo {
    /// Source type (git or tarball)
    #[serde(rename = "type")]
    pub source_type: String,
    /// Repository URL
    pub url: String,
    /// Git branch
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Git tag
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    /// Path within repository
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct BuildConfig {
    /// Cargo features (for Rust)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<Vec<String>>,
    /// Build profile (release, debug)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
    /// Additional build flags
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flags: Option<Vec<String>>,
}

/// Worker catalog entry
///
/// Represents a worker available in the catalog for download/installation.
/// Contains build instructions, dependencies, and capability information.
///
/// **This is DIFFERENT from WorkerBinary:**
/// - WorkerCatalogEntry = Available worker (catalog/provisioner)
/// - WorkerBinary = Installed worker (local filesystem)
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WorkerCatalogEntry {
    // ━━━ Identity ━━━
    /// Unique worker ID (e.g., "llm-worker-rbee-cpu")
    pub id: String,
    
    /// Worker implementation type
    pub implementation: WorkerImplementation,
    
    /// Worker type (backend)
    pub worker_type: WorkerType,
    
    /// Version (semver)
    pub version: String,
    
    // ━━━ Platform Support ━━━
    /// Supported platforms
    pub platforms: Vec<Platform>,
    
    /// Supported architectures
    pub architectures: Vec<Architecture>,
    
    // ━━━ Metadata ━━━
    /// Human-readable name
    pub name: String,
    
    /// Short description
    pub description: String,
    
    /// License (SPDX identifier)
    pub license: String,
    
    // ━━━ Build Instructions ━━━
    /// URL to PKGBUILD file
    pub pkgbuild_url: String,
    
    /// Build system
    pub build_system: BuildSystem,
    
    /// Source repository
    pub source: SourceInfo,
    
    /// Build configuration
    pub build: BuildConfig,
    
    // ━━━ Dependencies ━━━
    /// Runtime dependencies
    pub depends: Vec<String>,
    
    /// Build dependencies
    pub makedepends: Vec<String>,
    
    // ━━━ Binary Info ━━━
    /// Binary name (output)
    pub binary_name: String,
    
    /// Installation path
    pub install_path: String,
    
    // ━━━ Capabilities ━━━
    /// Supported model formats
    pub supported_formats: Vec<String>,
    
    /// Maximum context length
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_context_length: Option<u32>,
    
    /// Supports streaming
    pub supports_streaming: bool,
    
    /// Supports batching
    pub supports_batching: bool,
}

impl WorkerCatalogEntry {
    /// Check if this worker supports a specific platform
    pub fn supports_platform(&self, platform: Platform) -> bool {
        self.platforms.contains(&platform)
    }
    
    /// Check if this worker supports a specific architecture
    pub fn supports_architecture(&self, arch: Architecture) -> bool {
        self.architectures.contains(&arch)
    }
    
    /// Check if this worker supports a specific model format
    pub fn supports_format(&self, format: &str) -> bool {
        self.supported_formats
            .iter()
            .any(|f| f.eq_ignore_ascii_case(format))
    }
}
