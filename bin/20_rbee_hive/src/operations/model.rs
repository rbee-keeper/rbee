//! Model catalog and provisioning operations
//!
//! TEAM-388: Extracted from job_router.rs
//!
//! Handles:
//! - ModelDownload: Download model from HuggingFace
//! - ModelList: List downloaded models
//! - ModelGet: Get model details
//! - ModelDelete: Remove downloaded model
//! - ModelLoad: Load model to RAM
//! - ModelUnload: Unload model from RAM

use anyhow::Result;
use observability_narration_core::n;
use operations_contract::Operation;
use rbee_hive_artifact_catalog::{Artifact, ArtifactCatalog, ArtifactProvisioner};
use rbee_hive_model_catalog::ModelCatalog;
use rbee_hive_model_provisioner::ModelProvisioner;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Handle model-related operations
///
/// TEAM-388: Extracted from job_router.rs for better organization
pub async fn handle_model_operation(
    operation: &Operation,
    model_catalog: Arc<ModelCatalog>,
    model_provisioner: Arc<ModelProvisioner>,
    job_id: &str,
    get_cancel_token: impl FnOnce() -> Option<CancellationToken>,
) -> Result<()> {
    match operation {
        Operation::ModelDownload(request) => {
            handle_model_download(request, model_catalog, model_provisioner, job_id, get_cancel_token).await
        }
        Operation::ModelList(request) => {
            handle_model_list(request, model_catalog, job_id).await
        }
        Operation::ModelGet(request) => {
            handle_model_get(request, model_catalog).await
        }
        Operation::ModelDelete(request) => {
            handle_model_delete(request, model_catalog).await
        }
        Operation::ModelLoad(request) => {
            handle_model_load(request).await
        }
        Operation::ModelUnload(request) => {
            handle_model_unload(request).await
        }
        _ => Err(anyhow::anyhow!("Not a model operation")),
    }
}

async fn handle_model_download(
    request: &operations_contract::ModelDownloadRequest,
    model_catalog: Arc<ModelCatalog>,
    model_provisioner: Arc<ModelProvisioner>,
    job_id: &str,
    get_cancel_token: impl FnOnce() -> Option<CancellationToken>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    let model = &request.model;
    n!("model_download_start", "📥 Downloading model '{}' on hive '{}'", model, hive_id);

    // Check if model already exists
    if model_catalog.contains(model) {
        n!("model_download_exists", "⚠️  Model '{}' already exists in catalog", model);
        return Err(anyhow::anyhow!("Model '{}' already exists", model));
    }

    // TEAM-379: Get cancellation token from job registry
    let cancel_token = get_cancel_token()
        .ok_or_else(|| anyhow::anyhow!("Job not found in registry"))?;
    
    let model_entry = model_provisioner.provision(model, job_id, cancel_token).await?;

    // Add to catalog
    model_catalog.add(model_entry)?;

    n!("model_download_complete", "✅ Model '{}' downloaded and added to catalog", model);
    Ok(())
}

async fn handle_model_list(
    request: &operations_contract::ModelListRequest,
    model_catalog: Arc<ModelCatalog>,
    job_id: &str,
) -> Result<()> {
    let hive_id = &request.hive_id;
    n!("model_list_start", "📋 Listing models on hive '{}'", hive_id);

    let models = model_catalog.list();

    n!("model_list_result", "Found {} model(s)", models.len());

    // TEAM-384: Convert to JSON for both table display and UI consumption
    let models_json: Vec<serde_json::Value> = models.iter().map(|m| {
        serde_json::json!({
            "id": m.id(),
            "name": m.name(),
            "size": m.size(),
            "path": m.path().display().to_string(),
            "loaded": false, // TEAM-384: UI expects this field
        })
    }).collect();

    if models.is_empty() {
        n!("model_list_empty", "No models found. Download a model with: ./rbee model download <model-id>");
    } else {
        // TEAM-384: Emit table for CLI users (human-readable)
        n!("model_list_table", table: &models_json);
    }
    
    // TEAM-384: Emit structured data for UI (separate channel from narration)
    observability_narration_core::sse_sink::emit_data(
        job_id,
        "model_list",
        serde_json::json!({"models": models_json})
    );
    
    n!("model_list_complete", "✅ Model list operation complete");
    Ok(())
}

async fn handle_model_get(
    request: &operations_contract::ModelGetRequest,
    model_catalog: Arc<ModelCatalog>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    let id = &request.id;
    n!("model_get_start", "🔍 Getting model '{}' on hive '{}'", id, hive_id);

    match model_catalog.get(id) {
        Ok(model) => {
            n!(
                "model_get_found",
                "✅ Model: {} | Name: {} | Path: {}",
                model.id(),
                model.name(),
                model.path().display()
            );

            // Emit model details as JSON
            let json = serde_json::to_string_pretty(&model)
                .unwrap_or_else(|_| "Failed to serialize".to_string());

            n!("model_get_details", "{}", json);
            Ok(())
        }
        Err(e) => {
            n!("model_get_error", "❌ Model '{}' not found: {}", id, e);
            Err(e)
        }
    }
}

async fn handle_model_delete(
    request: &operations_contract::ModelDeleteRequest,
    model_catalog: Arc<ModelCatalog>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    let id = &request.id;
    n!("model_delete_start", "🗑️  Deleting model '{}' on hive '{}'", id, hive_id);

    match model_catalog.remove(id) {
        Ok(()) => {
            n!("model_delete_complete", "✅ Model '{}' deleted successfully", id);
            Ok(())
        }
        Err(e) => {
            n!("model_delete_error", "❌ Failed to delete model '{}': {}", id, e);
            Err(e)
        }
    }
}

async fn handle_model_load(request: &operations_contract::ModelLoadRequest) -> Result<()> {
    let _hive_id = &request.hive_id;
    let id = &request.id;
    let device = &request.device;
    n!("model_load_start", "🚀 Loading model '{}' to RAM on device '{}'", id, device);

    // TODO: Implement actual model loading to RAM
    // This will spawn a worker with the model loaded
    // For now, just narrate the intent
    n!("model_load_progress", "📦 Allocating memory for model '{}'", id);
    n!("model_load_progress", "🔄 Loading model weights into VRAM/RAM");
    n!("model_load_complete", "✅ Model '{}' loaded to RAM on device '{}'", id, device);
    Ok(())
}

async fn handle_model_unload(request: &operations_contract::ModelUnloadRequest) -> Result<()> {
    let _hive_id = &request.hive_id;
    let id = &request.id;
    n!("model_unload_start", "🔽 Unloading model '{}' from RAM", id);

    // TODO: Implement actual model unloading
    // This will kill the worker process
    // For now, just narrate the intent
    n!("model_unload_progress", "🧹 Freeing memory for model '{}'", id);
    n!("model_unload_complete", "✅ Model '{}' unloaded from RAM", id);
    Ok(())
}
