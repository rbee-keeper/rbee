// TEAM-408: WASM bindings for worker catalog
//! WASM bindings for worker catalog
//!
//! Exposes worker catalog functions to JavaScript/TypeScript via wasm-bindgen.

use wasm_bindgen::prelude::*;
use crate::worker_catalog::{WorkerCatalogClient, WorkerFilter};

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
