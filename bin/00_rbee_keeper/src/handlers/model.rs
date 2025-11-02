//! Model command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-186: Use typed Operation enum instead of JSON strings
//! TEAM-187: Match on &action to avoid cloning hive_id multiple times
//! TEAM-324: Moved ModelAction enum here to eliminate duplication
//! TEAM-384: Changed to connect directly to hive (no queen) - queen doesn't handle model ops anyway
//! TEAM-385: Added default test model (TinyLlama) for development

/// Default test model for development and testing
/// 
/// TinyLlama-1.1B-Chat-v1.0 (Q4_K_M quantization)
/// - Size: ~600 MB
/// - VRAM: ~1 GB (with 2K context)
/// - Perfect for testing model download/list/delete operations
const DEFAULT_TEST_MODEL: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

use anyhow::Result;
use clap::Subcommand;
use operations_contract::{
    ModelDeleteRequest, ModelDownloadRequest, ModelGetRequest, ModelListRequest,
    ModelLoadRequest, ModelUnloadRequest, Operation,
}; // TEAM-284: Renamed from rbee_operations

use crate::job_client::submit_and_stream_job_to_hive;
use crate::handlers::hive_jobs::get_hive_url;

#[derive(Subcommand)]
pub enum ModelAction {
    /// Download a model from HuggingFace
    /// 
    /// If no model is specified, downloads the default test model (TinyLlama-1.1B-Chat-v1.0).
    /// This is perfect for testing and development.
    #[command(visible_alias = "dl")]
    Download {
        /// Model identifier (e.g., "meta-llama/Llama-3.2-1B")
        /// 
        /// Defaults to TinyLlama-1.1B-Chat-v1.0 (Q4_K_M, ~600MB) if not specified.
        /// This allows quick testing with: ./rbee model download
        model: Option<String>,
    },
    
    /// List all downloaded models
    #[command(visible_alias = "ls")]
    List,
    
    /// Show details of a specific model
    #[command(visible_alias = "show")]
    Get {
        /// Model ID
        id: String,
    },
    
    /// Remove a downloaded model
    #[command(visible_alias = "rm")]
    Delete {
        /// Model ID to delete
        id: String,
    },
    
    /// Preload a model into RAM (for faster VRAM loading)
    /// 
    /// This caches the model in system RAM so that when a worker spawns,
    /// loading from RAM → VRAM is much faster than disk → VRAM.
    /// No worker is spawned by this command.
    Preload {
        /// Model ID to preload
        id: String,
    },
    
    /// Unload a model from RAM cache
    Unpreload {
        /// Model ID to unload from RAM
        id: String,
    },
}

pub async fn handle_model(hive_id: String, action: ModelAction) -> Result<()> {
    // TEAM-384: Build operation
    let operation = match &action {
        ModelAction::Download { model } => {
            // TEAM-385: Use default test model if none specified
            let model_id = model.as_deref().unwrap_or(DEFAULT_TEST_MODEL);
            Operation::ModelDownload(ModelDownloadRequest { 
                hive_id: hive_id.clone(), 
                model: model_id.to_string() 
            })
        }
        ModelAction::List => {
            Operation::ModelList(ModelListRequest { hive_id: hive_id.clone() })
        }
        ModelAction::Get { id } => {
            Operation::ModelGet(ModelGetRequest { 
                hive_id: hive_id.clone(), 
                id: id.clone() 
            })
        }
        ModelAction::Delete { id } => {
            Operation::ModelDelete(ModelDeleteRequest { 
                hive_id: hive_id.clone(), 
                id: id.clone() 
            })
        }
        ModelAction::Preload { id } => {
            Operation::ModelLoad(ModelLoadRequest {
                hive_id: hive_id.clone(),
                id: id.clone(),
                device: "ram".to_string(), // RAM preload, not VRAM
            })
        }
        ModelAction::Unpreload { id } => {
            Operation::ModelUnload(ModelUnloadRequest {
                hive_id: hive_id.clone(),
                id: id.clone(),
            })
        }
    };
    
    // TEAM-384: Connect directly to hive (no queen)
    // Queen doesn't handle model operations anyway, it just forwards to hive
    let hive_url = get_hive_url(&hive_id);
    submit_and_stream_job_to_hive(&hive_url, operation).await
}
