//! Model command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-186: Use typed Operation enum instead of JSON strings
//! TEAM-187: Match on &action to avoid cloning hive_id multiple times
//! TEAM-324: Moved ModelAction enum here to eliminate duplication
//! TEAM-384: Changed to connect directly to hive (no queen) - queen doesn't handle model ops anyway

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
    #[command(visible_alias = "dl")]
    Download {
        /// Model identifier (e.g., "meta-llama/Llama-3.2-1B")
        model: String,
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
            Operation::ModelDownload(ModelDownloadRequest { 
                hive_id: hive_id.clone(), 
                model: model.clone() 
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
