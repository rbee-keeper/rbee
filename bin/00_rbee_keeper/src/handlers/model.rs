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
    ModelDeleteRequest, ModelDownloadRequest, ModelGetRequest, ModelListRequest, Operation,
}; // TEAM-284: Renamed from rbee_operations

use crate::job_client::submit_and_stream_job_to_hive;
use crate::handlers::hive_jobs::get_hive_url;

#[derive(Subcommand)]
pub enum ModelAction {
    Download { model: String },
    List,
    Get { id: String },
    Delete { id: String },
}

pub async fn handle_model(hive_id: String, action: ModelAction) -> Result<()> {
    // TEAM-384: Build operation
    let operation = match &action {
        ModelAction::Download { model } => {
            Operation::ModelDownload(ModelDownloadRequest { hive_id: hive_id.clone(), model: model.clone() })
        }
        ModelAction::List => Operation::ModelList(ModelListRequest { hive_id: hive_id.clone() }),
        ModelAction::Get { id } => Operation::ModelGet(ModelGetRequest { hive_id: hive_id.clone(), id: id.clone() }),
        ModelAction::Delete { id } => {
            Operation::ModelDelete(ModelDeleteRequest { hive_id: hive_id.clone(), id: id.clone() })
        }
    };
    
    // TEAM-384: Connect directly to hive (no queen)
    // Queen doesn't handle model operations anyway, it just forwards to hive
    let hive_url = get_hive_url(&hive_id);
    submit_and_stream_job_to_hive(&hive_url, operation).await
}
