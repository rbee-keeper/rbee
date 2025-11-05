// TEAM-408: WASM bindings for worker catalog
// TEAM-409: Added compatibility checking bindings
//! WASM bindings for worker catalog and compatibility checking
//!
//! Exposes worker catalog and compatibility functions to JavaScript/TypeScript via wasm-bindgen.

use wasm_bindgen::prelude::*;
use crate::worker_catalog::{WorkerCatalogClient, WorkerFilter};
use crate::compatibility::{is_model_compatible, filter_compatible_models, check_model_worker_compatibility};
use artifacts_contract::ModelMetadata;

/// List all available workers
///
/// Returns a JavaScript array of worker binaries.
///
/// # Returns
/// JavaScript array of `WorkerBinary` objects
///
/// # Errors
/// Returns JavaScript error if network request fails
#[wasm_bindgen]
pub async fn list_workers() -> Result<JsValue, JsValue> {
    let client = WorkerCatalogClient::default();
    
    let workers = client
        .list_workers()
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to list workers: {}", e)))?;
    
    serde_wasm_bindgen::to_value(&workers)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize workers: {}", e)))
}

/// Get a specific worker by ID
///
/// # Arguments
/// * `id` - Worker ID (e.g., "llm-worker-rbee-cuda")
///
/// # Returns
/// JavaScript object representing the worker, or `null` if not found
///
/// # Errors
/// Returns JavaScript error if network request fails
#[wasm_bindgen]
pub async fn get_worker(id: String) -> Result<JsValue, JsValue> {
    let client = WorkerCatalogClient::default();
    
    let worker = client
        .get_worker(&id)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to get worker: {}", e)))?;
    
    match worker {
        Some(w) => serde_wasm_bindgen::to_value(&w)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize worker: {}", e))),
        None => Ok(JsValue::NULL),
    }
}

/// Filter workers by criteria
///
/// # Arguments
/// * `filter` - JavaScript object with filter criteria
///
/// # Returns
/// JavaScript array of matching workers
///
/// # Errors
/// Returns JavaScript error if network request fails or filter is invalid
#[wasm_bindgen]
pub async fn filter_workers(filter: JsValue) -> Result<JsValue, JsValue> {
    let client = WorkerCatalogClient::default();
    
    let filter: WorkerFilter = serde_wasm_bindgen::from_value(filter)
        .map_err(|e| JsValue::from_str(&format!("Invalid filter: {}", e)))?;
    
    let workers = client
        .filter_workers(filter)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to filter workers: {}", e)))?;
    
    serde_wasm_bindgen::to_value(&workers)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize workers: {}", e)))
}

/// Find workers compatible with a model
///
/// # Arguments
/// * `architecture` - Model architecture (e.g., "llama", "mistral")
/// * `format` - Model format (e.g., "safetensors", "gguf")
///
/// # Returns
/// JavaScript array of compatible workers
///
/// # Errors
/// Returns JavaScript error if network request fails
#[wasm_bindgen]
pub async fn find_compatible_workers(
    architecture: String,
    format: String,
) -> Result<JsValue, JsValue> {
    let client = WorkerCatalogClient::default();
    
    let workers = client
        .find_compatible_workers(&architecture, &format)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to find compatible workers: {}", e)))?;
    
    serde_wasm_bindgen::to_value(&workers)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize workers: {}", e)))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-409: COMPATIBILITY CHECKING WASM BINDINGS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Check if a model is compatible with ANY worker
///
/// # Arguments
/// * `metadata` - JavaScript object with model metadata
///
/// # Returns
/// JavaScript object with compatibility result
///
/// # Errors
/// Returns JavaScript error if metadata is invalid
#[wasm_bindgen]
pub fn is_model_compatible_wasm(metadata: JsValue) -> Result<JsValue, JsValue> {
    let metadata: ModelMetadata = serde_wasm_bindgen::from_value(metadata)
        .map_err(|e| JsValue::from_str(&format!("Invalid model metadata: {}", e)))?;
    
    let result = is_model_compatible(&metadata);
    
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
}

/// Filter models to only include compatible ones
///
/// # Arguments
/// * `models` - JavaScript array of model metadata objects
///
/// # Returns
/// JavaScript array of compatible models
///
/// # Errors
/// Returns JavaScript error if models array is invalid
#[wasm_bindgen]
pub fn filter_compatible_models_wasm(models: JsValue) -> Result<JsValue, JsValue> {
    let models: Vec<ModelMetadata> = serde_wasm_bindgen::from_value(models)
        .map_err(|e| JsValue::from_str(&format!("Invalid models array: {}", e)))?;
    
    let compatible = filter_compatible_models(models);
    
    serde_wasm_bindgen::to_value(&compatible)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize models: {}", e)))
}

/// Check if a specific model is compatible with a specific worker
///
/// # Arguments
/// * `metadata` - JavaScript object with model metadata
/// * `worker_architectures` - JavaScript array of supported architectures
/// * `worker_formats` - JavaScript array of supported formats
/// * `worker_max_context` - Maximum context length supported by worker
///
/// # Returns
/// JavaScript object with compatibility result
///
/// # Errors
/// Returns JavaScript error if arguments are invalid
#[wasm_bindgen]
pub fn check_model_worker_compatibility_wasm(
    metadata: JsValue,
    worker_architectures: JsValue,
    worker_formats: JsValue,
    worker_max_context: u32,
) -> Result<JsValue, JsValue> {
    let metadata: ModelMetadata = serde_wasm_bindgen::from_value(metadata)
        .map_err(|e| JsValue::from_str(&format!("Invalid model metadata: {}", e)))?;
    
    let worker_architectures: Vec<String> = serde_wasm_bindgen::from_value(worker_architectures)
        .map_err(|e| JsValue::from_str(&format!("Invalid worker architectures: {}", e)))?;
    
    let worker_formats: Vec<String> = serde_wasm_bindgen::from_value(worker_formats)
        .map_err(|e| JsValue::from_str(&format!("Invalid worker formats: {}", e)))?;
    
    let result = check_model_worker_compatibility(
        &metadata,
        &worker_architectures,
        &worker_formats,
        worker_max_context,
    );
    
    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize result: {}", e)))
}
