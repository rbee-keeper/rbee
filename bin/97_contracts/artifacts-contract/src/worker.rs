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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum WorkerType {
    /// CPU-based worker
    Cpu,
    /// CUDA-based worker (NVIDIA GPU)
    Cuda,
    /// Metal-based worker (Apple GPU)
    Metal,
}

impl WorkerType {
    /// Get the binary name for this worker type
    /// TEAM-404: Updated to match simplified enum
    pub fn binary_name(&self) -> &str {
        match self {
            WorkerType::Cpu => "llm-worker-rbee-cpu",
            WorkerType::Cuda => "llm-worker-rbee-cuda",
            WorkerType::Metal => "llm-worker-rbee-metal",
        }
    }

    /// Get the crate name (for building)
    pub fn crate_name(&self) -> &str {
        "llm-worker-rbee"
    }

    /// Get the features needed to build this worker type
    pub fn build_features(&self) -> Option<&str> {
        match self {
            WorkerType::Cpu => Some("cpu"),
            WorkerType::Cuda => Some("cuda"),
            WorkerType::Metal => Some("metal"),
        }
    }
}

/// Platform (canonical source of truth)
/// TEAM-404: Matches Hono catalog types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
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
}

impl WorkerBinary {
    /// Create a new worker binary entry
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
        }
    }
}
