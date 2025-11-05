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

// TEAM-408: Worker catalog client
pub mod worker_catalog;

// TEAM-408: WASM bindings for worker catalog
#[cfg(target_arch = "wasm32")]
mod wasm_worker;

// TEAM-408: Re-export WASM worker functions
#[cfg(target_arch = "wasm32")]
pub use wasm_worker::*;

// Re-export types
pub use types::*;

// TEAM-405: Re-export native clients
#[cfg(not(target_arch = "wasm32"))]
pub use huggingface::HuggingFaceClient;

// TEAM-408: Re-export worker catalog
pub use worker_catalog::{WorkerCatalogClient, WorkerFilter};

// TEAM-404: Explicitly re-export WorkerType and Platform for WASM/TypeScript generation
// TEAM-407: Added ModelMetadata types for marketplace compatibility filtering
// These are canonical types from artifacts-contract
// Note: Platform is used in wasm_bindgen function below
#[allow(unused_imports)]
pub use artifacts_contract::{
    WorkerType, Platform, WorkerBinary,
    ModelArchitecture, ModelFormat, Quantization, ModelMetadata,
};

/// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    web_sys::console::log_1(&"🛒 [Marketplace SDK] WASM initialized!".into());
}

// TEAM-404: Dummy functions to force TypeScript type generation for WorkerType and Platform
// These ensure tsify generates the types even though they're re-exported from artifacts-contract

/// Get WorkerType as string (forces TypeScript type generation)
#[wasm_bindgen]
pub fn worker_type_to_string(worker_type: WorkerType) -> String {
    match worker_type {
        WorkerType::Cpu => "cpu".to_string(),
        WorkerType::Cuda => "cuda".to_string(),
        WorkerType::Metal => "metal".to_string(),
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
