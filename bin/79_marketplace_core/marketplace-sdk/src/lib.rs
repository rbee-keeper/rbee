// TEAM-402: Marketplace SDK main entry point
// TEAM-405: Added HuggingFace client for native Rust (non-WASM) usage

#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

use wasm_bindgen::prelude::*;

// Modules
mod types;

// TEAM-405: Native Rust API clients (not WASM)
#[cfg(not(target_arch = "wasm32"))]
mod huggingface;

// TEAM-460: Native Civitai API client (not WASM)
#[cfg(not(target_arch = "wasm32"))]
mod civitai;

// TEAM-408: Worker catalog client (native only - uses reqwest)
// TEAM-429: Made conditional - not available on WASM
#[cfg(not(target_arch = "wasm32"))]
pub mod worker_catalog;

// TEAM-409: Compatibility checking for HuggingFace filtering
/// Compatibility checking module for filtering HuggingFace models
pub mod compatibility;

// TEAM-408: WASM bindings for worker catalog
// TEAM-429: Disabled - worker catalog uses reqwest which doesn't work on WASM
// #[cfg(target_arch = "wasm32")]
// mod wasm_worker;

// TEAM-460: WASM bindings for HuggingFace
#[cfg(target_arch = "wasm32")]
mod wasm_huggingface;

// TEAM-429: WASM bindings for Civitai
#[cfg(target_arch = "wasm32")]
mod wasm_civitai;

// TEAM-408: Re-export WASM worker functions
// TEAM-429: Disabled - worker catalog uses reqwest which doesn't work on WASM
// #[cfg(target_arch = "wasm32")]
// pub use wasm_worker::*;

// TEAM-460: Re-export WASM HuggingFace functions
#[cfg(target_arch = "wasm32")]
pub use wasm_huggingface::*;

// TEAM-429: Re-export WASM Civitai functions
#[cfg(target_arch = "wasm32")]
pub use wasm_civitai::*;

// Re-export types
pub use types::*;

// TEAM-405: Re-export native clients
#[cfg(not(target_arch = "wasm32"))]
pub use huggingface::HuggingFaceClient;

// TEAM-463: Re-export Civitai client
// TEAM-464: CivitaiModelType now comes from artifacts-contract
#[cfg(not(target_arch = "wasm32"))]
pub use civitai::{
    CivitaiClient,
    CivitaiListResponse, CivitaiMetadata,
};

// TEAM-408: Re-export worker catalog (native only)
// TEAM-429: Made conditional - not available on WASM
#[cfg(not(target_arch = "wasm32"))]
pub use worker_catalog::{WorkerCatalogClient, WorkerFilter};

// TEAM-409: Re-export compatibility checking
pub use compatibility::{
    is_model_compatible, filter_compatible_models, check_model_worker_compatibility,
    CompatibilityResult, CompatibilityConfidence,
};

// TEAM-404: Explicitly re-export WorkerType and Platform for WASM/TypeScript generation
// TEAM-407: Added ModelMetadata types for marketplace compatibility filtering
// TEAM-463: Added CivitAI types from artifacts-contract
// TEAM-429: Added filter types for WASM/TypeScript generation
// These are canonical types from artifacts-contract
// Note: Platform is used in wasm_bindgen function below
#[allow(unused_imports)]
pub use artifacts_contract::{
    ModelEntry, ModelType, ModelConfig,
    WorkerBinary, WorkerType, Platform,
    ModelArchitecture, ModelFormat, Quantization, ModelMetadata,
    WorkerCatalogEntry, Architecture, WorkerImplementation, BuildSystem,
    ArtifactStatus,
    CivitaiModel, CivitaiModelVersion, CivitaiStats, CivitaiCreator,
    CivitaiFile, CivitaiImage,
    // TEAM-429: Filter types
    CivitaiFilters, CivitaiModelType, TimePeriod, BaseModel, CivitaiSort,
    HuggingFaceFilters, HuggingFaceSort,
    NsfwLevel, NsfwFilter,
};

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    web_sys::console::log_1(&"🛒 [Marketplace SDK] WASM initialized!".into());
}

// TEAM-404: Dummy functions to force TypeScript type generation for WorkerType and Platform
// These ensure tsify generates the types even though they're re-exported from artifacts-contract

/// Get WorkerType as string (forces TypeScript type generation)
/// TEAM-423: Added ROCm support
#[wasm_bindgen]
pub fn worker_type_to_string(worker_type: WorkerType) -> String {
    match worker_type {
        WorkerType::Cpu => "cpu".to_string(),
        WorkerType::Cuda => "cuda".to_string(),
        WorkerType::Metal => "metal".to_string(),
        WorkerType::Rocm => "rocm".to_string(),
    }
}

/// Get Platform as string (forces TypeScript type generation)
#[wasm_bindgen]
pub fn platform_to_string(platform: Platform) -> String {
    match platform {
        Platform::Linux => "linux".to_string(),
        Platform::MacOS => "macos".to_string(),
        Platform::Windows => "windows".to_string(),
    }
}

/// Check if a model is compatible with our workers (WASM binding)
/// TEAM-463: Exposed compatibility checking to TypeScript/Node.js
///
/// # Arguments
/// * `metadata` - Model metadata extracted from HuggingFace
///
/// # Returns
/// CompatibilityResult with detailed compatibility information
#[wasm_bindgen]
pub fn is_model_compatible_wasm(metadata: ModelMetadata) -> CompatibilityResult {
    is_model_compatible(&metadata)
}
