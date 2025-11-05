//! Hive job operations using job-client
//!
//! TEAM-380: Split from hive.rs - this file handles job operations (worker/model management)
//!
//! This module uses the shared job-client crate to communicate with rbee-hive's
//! job server endpoint (http://localhost:7835/v1/jobs).
//!
//! Operations handled here:
//! - Worker process operations (spawn, list, get, delete)
//! - Model operations (download, list, get, delete)
//! - Diagnostic operations (HiveCheck)

use anyhow::Result;
use clap::Subcommand;
use job_client::JobClient;
use operations_contract::{
    Operation, ModelDeleteRequest, ModelDownloadRequest, ModelGetRequest, ModelListRequest,
    WorkerProcessDeleteRequest, WorkerProcessGetRequest, WorkerProcessListRequest,
    WorkerSpawnRequest,
};

/// CLI actions for hive job operations
#[derive(Subcommand)]
pub enum HiveJobsAction {
    /// Spawn a worker process on hive
    WorkerSpawn {
        /// Model to load
        #[arg(long)]
        model: String,
        /// Worker type (cpu, cuda, metal)
        #[arg(long)]
        worker: String,
        /// Device index
        #[arg(long, default_value = "0")]
        device: u32,
    },

    /// List worker processes running on hive
    WorkerList,

    /// Get details of a specific worker process
    WorkerGet {
        /// Process ID
        #[arg(long)]
        pid: u32,
    },

    /// Delete (kill) a worker process
    WorkerDelete {
        /// Process ID
        #[arg(long)]
        pid: u32,
    },

    /// Download a model to hive
    ModelDownload {
        /// Model identifier (e.g., "meta-llama/Llama-2-7b")
        #[arg(long)]
        model: String,
    },

    /// List models available on hive
    ModelList,

    /// Get details of a specific model
    ModelGet {
        /// Model ID
        #[arg(long)]
        id: String,
    },

    /// Delete a model from hive
    ModelDelete {
        /// Model ID
        #[arg(long)]
        id: String,
    },

    /// Run hive diagnostic check (SSE streaming test)
    Check,
}

/// Submit a hive job operation and stream results
///
/// TEAM-380: Generic handler for all hive job operations
/// Uses job-client to:
/// 1. POST operation to /v1/jobs
/// 2. Connect to SSE stream
/// 3. Print narration events to stdout
/// 4. Wait for \[DONE\] marker
// TEAM-387: RULE ZERO - Updated to use new API
pub async fn submit_hive_job(operation: Operation, hive_url: &str) -> Result<()> {
    let client = JobClient::new(hive_url);
    
    let (_job_id, stream_future) = client.submit_and_stream(operation, |line| {
        // Print narration events to stdout
        println!("{}", line);
        Ok(())
    }).await?;
    
    // Await the stream future
    stream_future.await?;
    
    Ok(())
}

/// Helper to get hive URL from alias
///
/// TEAM-380: Resolves hive alias to HTTP URL
/// TEAM-384: Now uses SSH config resolver for remote hives
///
/// # Behavior
/// - "localhost" → `http://localhost:7835`
/// - Other aliases → Parse `~/.ssh/config` for host entry, use hostname
///
/// # Example
/// ```rust,ignore
/// // Localhost
/// let url = get_hive_url("localhost");
/// // → "http://localhost:7835"
///
/// // Remote (from ~/.ssh/config)
/// let url = get_hive_url("workstation");
/// // → "http://192.168.1.100:7835"
/// ```
pub fn get_hive_url(alias: &str) -> String {
    use crate::ssh_resolver::resolve_ssh_config;
    
    if alias == "localhost" {
        return "http://localhost:7835".to_string();
    }
    
    // TEAM-384: Resolve via SSH config
    match resolve_ssh_config(alias) {
        Ok(ssh) => format!("http://{}:7835", ssh.hostname),
        Err(_) => {
            // Fallback: assume alias is hostname
            format!("http://{}:7835", alias)
        }
    }
}

/// Handle hive job actions from CLI
///
/// TEAM-380: Converts HiveJobsAction to Operation and submits via job-client
pub async fn handle_hive_jobs(hive_id: String, action: HiveJobsAction) -> Result<()> {
    let hive_url = get_hive_url(&hive_id);

    let operation = match action {
        HiveJobsAction::WorkerSpawn { model, worker, device } => {
            Operation::WorkerSpawn(WorkerSpawnRequest {
                hive_id: hive_id.clone(),
                model,
                worker,
                device,
            })
        }
        HiveJobsAction::WorkerList => {
            Operation::WorkerProcessList(WorkerProcessListRequest {
                hive_id: hive_id.clone(),
            })
        }
        HiveJobsAction::WorkerGet { pid } => {
            Operation::WorkerProcessGet(WorkerProcessGetRequest {
                hive_id: hive_id.clone(),
                pid,
            })
        }
        HiveJobsAction::WorkerDelete { pid } => {
            Operation::WorkerProcessDelete(WorkerProcessDeleteRequest {
                hive_id: hive_id.clone(),
                pid,
            })
        }
        HiveJobsAction::ModelDownload { model } => {
            Operation::ModelDownload(ModelDownloadRequest {
                hive_id: hive_id.clone(),
                model,
            })
        }
        HiveJobsAction::ModelList => {
            Operation::ModelList(ModelListRequest {
                hive_id: hive_id.clone(),
            })
        }
        HiveJobsAction::ModelGet { id } => {
            Operation::ModelGet(ModelGetRequest {
                hive_id: hive_id.clone(),
                id,
            })
        }
        HiveJobsAction::ModelDelete { id } => {
            Operation::ModelDelete(ModelDeleteRequest {
                hive_id: hive_id.clone(),
                id,
            })
        }
        HiveJobsAction::Check => {
            Operation::HiveCheck { alias: hive_id.clone() }
        }
    };

    submit_hive_job(operation, &hive_url).await
}
