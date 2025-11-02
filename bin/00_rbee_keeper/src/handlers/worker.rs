//! Worker command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-274: Updated worker actions for new architecture
//! TEAM-324: Moved WorkerAction and WorkerProcessAction enums here to eliminate duplication
//! TEAM-388: Rewritten to match model.rs pattern with catalog operations

use anyhow::Result;
use clap::Subcommand;
use operations_contract::{
    Operation, WorkerCatalogGetRequest, WorkerCatalogListRequest, WorkerInstallRequest,
    WorkerListInstalledRequest, WorkerProcessDeleteRequest, WorkerProcessGetRequest,
    WorkerProcessListRequest, WorkerRemoveRequest, WorkerSpawnRequest,
}; // TEAM-284: Renamed from rbee_operations

use crate::handlers::hive_jobs::get_hive_url;
use crate::job_client::submit_and_stream_job_to_hive;

#[derive(Subcommand)]
pub enum WorkerAction {
    /// List available workers from catalog server
    /// 
    /// Shows all workers that can be installed from the Hono catalog.
    /// These are workers that can be downloaded, built, and installed.
    #[command(visible_alias = "catalog")]
    Available,

    /// List installed worker binaries
    /// 
    /// Shows workers that have been downloaded and installed to ~/.cache/rbee/workers/
    #[command(visible_alias = "ls")]
    List,

    /// Show details of a specific worker
    /// 
    /// Shows details from either the catalog (if not installed) or from the
    /// installed workers directory (if installed).
    #[command(visible_alias = "show")]
    Get {
        /// Worker ID (e.g., "llm-worker-rbee-cpu")
        worker_id: String,
    },

    /// Download, build, and install a worker from catalog
    /// 
    /// This will:
    /// 1. Download PKGBUILD from catalog server
    /// 2. Build the worker binary
    /// 3. Install to ~/.cache/rbee/workers/
    #[command(visible_alias = "install")]
    Download {
        /// Worker ID from catalog (e.g., "llm-worker-rbee-cpu")
        worker_id: String,
    },

    /// Remove an installed worker binary
    /// 
    /// Removes the worker from ~/.cache/rbee/workers/
    #[command(visible_alias = "rm")]
    Remove {
        /// Worker ID to remove (e.g., "llm-worker-rbee-cpu")
        worker_id: String,
    },

    /// Spawn a worker process with a model
    /// 
    /// Starts a worker process that loads the specified model.
    /// The worker must already be installed.
    Spawn {
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

    /// Worker process management (local ps on hive)
    #[command(subcommand)]
    Process(WorkerProcessAction),
}

#[derive(Subcommand)]
pub enum WorkerProcessAction {
    /// List worker processes (local ps)
    List,
    /// Get worker process details by PID
    Get { pid: u32 },
    /// Delete (kill) worker process by PID
    Delete { pid: u32 },
}

pub async fn handle_worker(hive_id: String, action: WorkerAction) -> Result<()> {
    // TEAM-388: Build operation
    let operation = match &action {
        WorkerAction::Available => Operation::WorkerCatalogList(WorkerCatalogListRequest {
            hive_id: hive_id.clone(),
        }),
        WorkerAction::List => Operation::WorkerListInstalled(WorkerListInstalledRequest {
            hive_id: hive_id.clone(),
        }),
        WorkerAction::Get { worker_id } => {
            // Try installed first, fall back to catalog
            Operation::WorkerInstalledGet(WorkerCatalogGetRequest {
                hive_id: hive_id.clone(),
                worker_id: worker_id.clone(),
            })
        }
        WorkerAction::Download { worker_id } => {
            Operation::WorkerInstall(WorkerInstallRequest {
                hive_id: hive_id.clone(),
                worker_id: worker_id.clone(),
            })
        }
        WorkerAction::Remove { worker_id } => Operation::WorkerRemove(WorkerRemoveRequest {
            hive_id: hive_id.clone(),
            worker_id: worker_id.clone(),
        }),
        WorkerAction::Spawn {
            model,
            worker,
            device,
        } => Operation::WorkerSpawn(WorkerSpawnRequest {
            hive_id: hive_id.clone(),
            model: model.clone(),
            worker: worker.clone(),
            device: *device,
        }),
        WorkerAction::Process(proc_action) => match proc_action {
            WorkerProcessAction::List => {
                Operation::WorkerProcessList(WorkerProcessListRequest {
                    hive_id: hive_id.clone(),
                })
            }
            WorkerProcessAction::Get { pid } => {
                Operation::WorkerProcessGet(WorkerProcessGetRequest {
                    hive_id: hive_id.clone(),
                    pid: *pid,
                })
            }
            WorkerProcessAction::Delete { pid } => {
                Operation::WorkerProcessDelete(WorkerProcessDeleteRequest {
                    hive_id: hive_id.clone(),
                    pid: *pid,
                })
            }
        },
    };

    // TEAM-388: Connect directly to hive (no queen)
    let hive_url = get_hive_url(&hive_id);
    submit_and_stream_job_to_hive(&hive_url, operation).await
}
