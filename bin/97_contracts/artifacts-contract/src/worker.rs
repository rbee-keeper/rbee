// TEAM-402: Worker binary type
//! Worker binary type
//!
//! Pure data type for worker artifacts.
//! Migrated from worker-catalog/src/types.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tsify::Tsify;

use crate::status::ArtifactStatus;

/// Worker type (canonical source of truth)
/// TEAM-404: Simplified to match Hono catalog types
/// TEAM-423: Added ROCm support for AMD GPUs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// CPU-based worker
    Cpu,
    /// CUDA-based worker (NVIDIA GPU)
    Cuda,
    /// Metal-based worker (Apple GPU)
    Metal,
    /// ROCm-based worker (AMD GPU)
    Rocm,
}

impl WorkerType {
    /// Get the binary name for this worker type
    /// TEAM-404: Updated to match simplified enum
    /// TEAM-423: Added ROCm support
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::Cpu => "llm-worker-rbee-cpu",
            WorkerType::Cuda => "llm-worker-rbee-cuda",
            WorkerType::Metal => "llm-worker-rbee-metal",
            WorkerType::Rocm => "llm-worker-rbee-rocm",
        }
    }

    /// Get the crate name (for building)
    pub fn crate_name(&self) -> &str {
        "llm-worker-rbee"
    }

    /// Get the features needed to build this worker type
    /// TEAM-423: Added ROCm support
    pub fn build_features(&self) -> Option<&str> {
        match self {
            WorkerType::Cpu => Some("cpu"),
            WorkerType::Cuda => Some("cuda"),
            WorkerType::Metal => Some("metal"),
            WorkerType::Rocm => Some("rocm"),
        }
    }
}

/// Platform (canonical source of truth)
/// TEAM-404: Matches Hono catalog types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[cfg_attr(feature = "specta", derive(specta::Type))]
#[serde(rename_all = "lowercase")]
pub enum Platform {
    /// Linux
    Linux,
    /// macOS
    #[serde(rename = "macos")]
    MacOS,
    /// Windows
    Windows,
}

impl Platform {
    /// Get the current platform
    pub fn current() -> Self {
        #[cfg(target_os = "linux")]
        {
            return Platform::Linux;
        }

        #[cfg(target_os = "macos")]
        {
            return Platform::MacOS;
        }

        #[cfg(target_os = "windows")]
        {
            return Platform::Windows;
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            panic!("Unsupported platform");
        }
    }

    /// Get the file extension for this platform
    pub fn extension(&self) -> &str {
        match self {
            Platform::Linux | Platform::MacOS => "",
            Platform::Windows => ".exe",
        }
    }
}

/// Worker binary entry in the catalog
///
/// Represents a worker binary artifact with metadata.
/// Used by worker-catalog, worker-provisioner, marketplace-sdk, and UI.
///
/// TEAM-407: Added capability fields for marketplace compatibility filtering
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct WorkerBinary {
    /// Unique worker ID (e.g., "cpu-llm-worker-rbee-v0.1.0-linux")
    pub id: String,

    /// Worker type
    pub worker_type: WorkerType,

    /// Platform
    pub platform: Platform,

    /// Filesystem path to worker binary
    #[tsify(type = "string")]
    pub path: PathBuf,

    /// Size in bytes
    pub size: u64,

    /// Current status
    pub status: ArtifactStatus,

    /// Version
    pub version: String,

    /// When the worker was added
    #[serde(default = "chrono::Utc::now")]
    pub added_at: chrono::DateTime<chrono::Utc>,

    // ========== TEAM-407: Capability Fields ==========
    /// Supported model architectures (e.g., ["llama", "mistral", "phi", "qwen"])
    #[serde(default)]
    pub supported_architectures: Vec<String>,

    /// Supported model formats (e.g., ["safetensors", "gguf"])
    #[serde(default)]
    pub supported_formats: Vec<String>,

    /// Maximum context length supported
    #[serde(default = "default_max_context")]
    pub max_context_length: u32,

    /// Supports token streaming
    #[serde(default = "default_true")]
    pub supports_streaming: bool,

    /// Supports request batching
    #[serde(default)]
    pub supports_batching: bool,
}

// TEAM-407: Default values for capability fields
fn default_max_context()
-> u32 {
    8192
}

fn default_true() -> bool {
    true
}

impl WorkerBinary {
    /// Create a new worker binary entry
    /// TEAM-407: Updated to include capability fields
    pub fn new(
        id: String,
        worker_type: WorkerType,
        platform: Platform,
        path: PathBuf,
        size: u64,
        version: String,
    ) -> Self {
        Self {
            id,
            worker_type,
            platform,
            path,
            size,
            status: ArtifactStatus::Available,
            version,
            added_at: chrono::Utc::now(),
            // TEAM-407: Default capabilities (can be updated later)
            supported_architectures: vec![],
            supported_formats: vec![],
            max_context_length: 8192,
            supports_streaming: true,
            supports_batching: false,
        }
    }

    /// Create a new worker binary with full capabilities
    /// TEAM-407: Builder method for workers with known capabilities
    pub fn with_capabilities(
        id: String,
        worker_type: WorkerType,
        platform: Platform,
        path: PathBuf,
        size: u64,
        version: String,
        supported_architectures: Vec<String>,
        supported_formats: Vec<String>,
        max_context_length: u32,
        supports_streaming: bool,
        supports_batching: bool,
    ) -> Self {
        Self {
            id,
            worker_type,
            platform,
            path,
            size,
            status: ArtifactStatus::Available,
            version,
            added_at: chrono::Utc::now(),
            supported_architectures,
            supported_formats,
            max_context_length,
            supports_streaming,
            supports_batching,
        }
    }
}
