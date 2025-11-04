//! Worker catalog and process operations
//!
//! TEAM-388: Extracted from job_router.rs
//!
//! Handles:
//! - WorkerCatalogList: List available workers from Hono catalog
//! - WorkerCatalogGet: Get worker details from Hono catalog
//! - WorkerInstalledGet: Get installed worker details
//! - WorkerInstall: Install worker from catalog
//! - WorkerRemove: Remove installed worker
//! - WorkerListInstalled: List installed workers
//! - WorkerSpawn: Start a worker process
//! - WorkerProcessList/Get/Delete: Manage running worker processes

use anyhow::Result;
use observability_narration_core::n;
use operations_contract::Operation;
use rbee_hive_artifact_catalog::{Artifact, ArtifactCatalog}; // TEAM-388: Needed for trait methods
use rbee_hive_worker_catalog::WorkerCatalog;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Handle worker-related operations
///
/// TEAM-388: Extracted from job_router.rs for better organization
pub async fn handle_worker_operation(
    operation: &Operation,
    worker_catalog: Arc<WorkerCatalog>,
    job_id: &str,
    get_cancel_token: impl FnOnce() -> Option<CancellationToken>,
) -> Result<()> {
    match operation {
        Operation::WorkerCatalogList(request) => {
            handle_worker_catalog_list(request).await
        }
        Operation::WorkerCatalogGet(request) => {
            handle_worker_catalog_get(request).await
        }
        Operation::WorkerInstalledGet(request) => {
            handle_worker_installed_get(request, worker_catalog).await
        }
        Operation::WorkerInstall(request) => {
            handle_worker_install(request, worker_catalog, job_id, get_cancel_token).await
        }
        Operation::WorkerRemove(request) => {
            handle_worker_remove(request, worker_catalog).await
        }
        Operation::WorkerListInstalled(request) => {
            handle_worker_list_installed(request, worker_catalog).await
        }
        Operation::WorkerSpawn(request) => {
            handle_worker_spawn(request, worker_catalog, job_id).await
        }
        Operation::WorkerProcessList(request) => {
            handle_worker_process_list(request).await
        }
        Operation::WorkerProcessGet(request) => {
            handle_worker_process_get(request).await
        }
        Operation::WorkerProcessDelete(request) => {
            handle_worker_process_delete(request).await
        }
        _ => Err(anyhow::anyhow!("Not a worker operation")),
    }
}

// ============================================================================
// WORKER CATALOG OPERATIONS
// ============================================================================

async fn handle_worker_catalog_list(request: &operations_contract::WorkerCatalogListRequest) -> Result<()> {
    let hive_id = &request.hive_id;
    n!("worker_catalog_list_start", "📋 Listing available workers from catalog (hive '{}')", hive_id);
    
    // Query Hono catalog server
    let catalog_url = "http://localhost:8787/workers";
    n!("worker_catalog_query", "🌐 Querying Hono catalog at {}", catalog_url);
    
    match reqwest::get(catalog_url).await {
        Ok(response) => {
            match response.json::<serde_json::Value>().await {
                Ok(catalog_data) => {
                    let empty_vec = vec![];
                    let workers = catalog_data["workers"]
                        .as_array()
                        .unwrap_or(&empty_vec);
                    
                    n!("worker_catalog_list_ok", "✅ Listed {} available workers from catalog", workers.len());
                    
                    // TEAM-388: Create simplified, user-friendly table with only essential info
                    let simplified: Vec<serde_json::Value> = workers.iter().map(|w| {
                        serde_json::json!({
                            "id": w["id"],
                            "name": w["name"],
                            "type": w["worker_type"],
                            "platforms": w["platforms"]
                                .as_array()
                                .map(|arr| arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", "))
                                .unwrap_or_else(|| "unknown".to_string()),
                            "description": w["description"]
                        })
                    }).collect();
                    
                    n!("worker_catalog_list_table", table: &simplified);
                    Ok(())
                }
                Err(e) => {
                    n!("worker_catalog_list_parse_error", "❌ Failed to parse catalog response: {}", e);
                    Err(anyhow::anyhow!("Failed to parse catalog response: {}", e))
                }
            }
        }
        Err(e) => {
            n!("worker_catalog_list_error", "❌ Failed to query Hono catalog: {}", e);
            n!("worker_catalog_list_hint", "💡 Make sure Hono catalog server is running on port 8787");
            Err(anyhow::anyhow!("Failed to query Hono catalog at {}: {}", catalog_url, e))
        }
    }
}

async fn handle_worker_catalog_get(request: &operations_contract::WorkerCatalogGetRequest) -> Result<()> {
    let hive_id = &request.hive_id;
    let worker_id = &request.worker_id;
    n!("worker_catalog_get_start", "🔍 Getting worker '{}' from catalog (hive '{}')", worker_id, hive_id);
    
    // Query Hono catalog server for specific worker
    let catalog_url = format!("http://localhost:8787/workers/{}", worker_id);
    n!("worker_catalog_get_query", "🌐 Querying Hono catalog at {}", catalog_url);
    
    match reqwest::get(&catalog_url).await {
        Ok(response) => {
            if response.status().is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(worker_data) => {
                        n!("worker_catalog_get_ok", "✅ Found worker '{}' in catalog", worker_id);
                        n!("worker_catalog_get_json", "{}", worker_data.to_string());
                        Ok(())
                    }
                    Err(e) => {
                        n!("worker_catalog_get_parse_error", "❌ Failed to parse worker data: {}", e);
                        Err(anyhow::anyhow!("Failed to parse worker data: {}", e))
                    }
                }
            } else if response.status().as_u16() == 404 {
                n!("worker_catalog_get_not_found", "❌ Worker '{}' not found in catalog", worker_id);
                Err(anyhow::anyhow!("Worker '{}' not found in catalog", worker_id))
            } else {
                n!("worker_catalog_get_error", "❌ Catalog server returned status: {}", response.status());
                Err(anyhow::anyhow!("Catalog server returned status: {}", response.status()))
            }
        }
        Err(e) => {
            n!("worker_catalog_get_error", "❌ Failed to query Hono catalog: {}", e);
            n!("worker_catalog_get_hint", "💡 Make sure Hono catalog server is running on port 8787");
            Err(anyhow::anyhow!("Failed to query Hono catalog at {}: {}", catalog_url, e))
        }
    }
}

async fn handle_worker_installed_get(
    request: &operations_contract::WorkerCatalogGetRequest,
    worker_catalog: Arc<WorkerCatalog>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    let worker_id = &request.worker_id;
    n!("worker_installed_get_start", "🔍 Getting installed worker '{}' (hive '{}')", worker_id, hive_id);
    
    // Find worker in catalog
    let workers = worker_catalog.list();
    let worker = workers.iter().find(|w| w.id() == worker_id)
        .ok_or_else(|| anyhow::anyhow!("Worker '{}' not found in catalog", worker_id))?;
    
    let response = serde_json::json!({
        "id": worker.id(),
        "name": worker.name(),
        "worker_type": format!("{:?}", worker.worker_type),
        "platform": format!("{:?}", worker.platform),
        "version": worker.version,
        "size": worker.size(),
        "path": worker.path().display().to_string(),
        "added_at": worker.added_at.to_rfc3339(),
    });
    
    n!("worker_installed_get_ok", "✅ Found installed worker '{}'", worker_id);
    n!("worker_installed_get_json", "{}", response.to_string());
    Ok(())
}

async fn handle_worker_install(
    request: &operations_contract::WorkerInstallRequest,
    worker_catalog: Arc<WorkerCatalog>,
    job_id: &str,
    get_cancel_token: impl FnOnce() -> Option<CancellationToken>,
) -> Result<()> {
    n!(
        "worker_install_start",
        "📦 Installing worker '{}' on hive '{}'",
        request.worker_id,
        request.hive_id
    );

    // TEAM-388: Get cancellation token from job registry
    let cancel_token = get_cancel_token()
        .ok_or_else(|| anyhow::anyhow!("Job not found in registry"))?;

    crate::worker_install::handle_worker_install(
        request.worker_id.clone(),
        worker_catalog,
        cancel_token,
    )
    .await?;

    n!(
        "worker_install_complete",
        "✅ Worker '{}' installation complete",
        request.worker_id
    );
    Ok(())
}

async fn handle_worker_remove(
    request: &operations_contract::WorkerRemoveRequest,
    worker_catalog: Arc<WorkerCatalog>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    let worker_id = &request.worker_id;
    n!("worker_remove_start", "🗑️  Removing worker '{}' from hive '{}'", worker_id, hive_id);
    
    // Check if worker exists before attempting removal
    if !worker_catalog.contains(worker_id) {
        n!("worker_remove_not_found", "❌ Worker '{}' not found in catalog", worker_id);
        return Err(anyhow::anyhow!("Worker '{}' not found in catalog", worker_id));
    }
    
    // Remove from catalog
    worker_catalog.remove(worker_id)?;
    
    n!("worker_remove_ok", "✅ Worker '{}' removed from catalog", worker_id);
    Ok(())
}

async fn handle_worker_list_installed(
    request: &operations_contract::WorkerListInstalledRequest,
    worker_catalog: Arc<WorkerCatalog>,
) -> Result<()> {
    let hive_id = &request.hive_id;
    n!("worker_list_installed_start", "📋 Listing installed workers on hive '{}'", hive_id);
    
    // List all installed workers from catalog
    let workers = worker_catalog.list();
    
    n!("worker_list_installed_count", "Found {} installed workers", workers.len());
    
    // Convert to JSON response for frontend
    let response = serde_json::json!({
        "workers": workers.iter().map(|w| {
            serde_json::json!({
                "id": w.id(),
                "name": w.name(),
                "worker_type": format!("{:?}", w.worker_type),
                "platform": format!("{:?}", w.platform),
                "version": w.version,
                "size": w.size(),
                "path": w.path().display().to_string(),
                "added_at": w.added_at.to_rfc3339(),
            })
        }).collect::<Vec<_>>()
    });
    
    n!("worker_list_installed_ok", "✅ Listed {} installed workers", workers.len());
    n!("worker_list_installed_json", "{}", response.to_string());
    Ok(())
}

async fn handle_worker_spawn(
    request: &operations_contract::WorkerSpawnRequest,
    worker_catalog: Arc<WorkerCatalog>,
    job_id: &str,
) -> Result<()> {
    use lifecycle_local::{start_daemon, HttpDaemonConfig, StartConfig};
    use rbee_hive_worker_catalog::{Platform, WorkerType};

    n!(
        "worker_spawn_start",
        "🚀 Spawning worker '{}' with model '{}' on device {}",
        request.worker,
        request.model,
        request.device
    );

    // Determine worker type from worker string
    // TEAM-404: Updated to use simplified WorkerType enum
    let worker_type = match request.worker.as_str() {
        "cuda" => WorkerType::Cuda,
        "cpu" => WorkerType::Cpu,
        "metal" => WorkerType::Metal,
        _ => return Err(anyhow::anyhow!("Unsupported worker type: {}", request.worker)),
    };

    // Find worker binary in catalog
    let _worker_binary = worker_catalog
        .find_by_type_and_platform(worker_type, Platform::current())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Worker binary not found for {:?}. \
                 Worker binaries must be installed via worker-catalog first!",
                worker_type
            )
        })?;

    // Allocate port
    let port = 9000 + (rand::random::<u16>() % 1000);
    let queen_url = "http://localhost:7833".to_string();
    let worker_id = format!("worker-{}-{}", request.worker, port);

    // Build worker arguments
    let args = vec![
        "--worker-id".to_string(),
        worker_id.clone(),
        "--model".to_string(),
        request.model.clone(),
        "--device".to_string(),
        request.device.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--queen-url".to_string(),
        queen_url,
    ];

    // Start worker with monitoring
    let base_url = format!("http://localhost:{}", port);
    let daemon_config = HttpDaemonConfig::new(&worker_id, &base_url)
        .with_args(args)
        .with_monitor_group("llm")
        .with_monitor_instance(port.to_string());
    
    let config = StartConfig {
        daemon_config,
        job_id: Some(job_id.to_string()),
    };

    let pid = start_daemon(config).await?;

    n!(
        "worker_spawn_complete",
        "✅ Worker '{}' spawned (PID: {}, port: {})",
        worker_id,
        pid,
        port
    );
    Ok(())
}

// ============================================================================
// WORKER PROCESS OPERATIONS
// ============================================================================

async fn handle_worker_process_list(request: &operations_contract::WorkerProcessListRequest) -> Result<()> {
    let hive_id = &request.hive_id;
    n!("worker_proc_list_start", "📋 Listing worker processes on hive '{}'", hive_id);

    // Use ps to list worker processes
    let output = tokio::process::Command::new("ps")
        .args(&["aux"])
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to run ps: {}", e))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let worker_lines: Vec<_> = stdout
        .lines()
        .filter(|line| line.contains("llm-worker") || line.contains("worker-rbee"))
        .collect();

    n!("worker_proc_list_result", "Found {} worker process(es)", worker_lines.len());

    if worker_lines.is_empty() {
        n!("worker_proc_list_empty", "No worker processes found");
    } else {
        for line in worker_lines {
            n!("worker_proc_list_entry", "  {}", line);
        }
    }
    Ok(())
}

async fn handle_worker_process_get(request: &operations_contract::WorkerProcessGetRequest) -> Result<()> {
    let hive_id = &request.hive_id;
    let pid = request.pid;

    n!(
        "worker_proc_get_start",
        "🔍 Getting worker process PID {} on hive '{}'",
        pid,
        hive_id
    );

    // Use ps to get specific process
    let output = tokio::process::Command::new("ps")
        .args(&["-p", &pid.to_string(), "-o", "pid,command"])
        .output()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to run ps: {}", e))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!("Process {} not found", pid));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    n!("worker_proc_get_found", "✅ PID {}: {}", pid, stdout.trim());
    Ok(())
}

async fn handle_worker_process_delete(request: &operations_contract::WorkerProcessDeleteRequest) -> Result<()> {
    let hive_id = &request.hive_id;
    let pid = request.pid;

    n!(
        "worker_proc_del_start",
        "🗑️  Deleting worker process PID {} on hive '{}'",
        pid,
        hive_id
    );

    // Kill process using SIGTERM
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;

        let pid_nix = Pid::from_raw(pid as i32);
        match kill(pid_nix, Signal::SIGTERM) {
            Ok(_) => {
                n!("worker_proc_del_sigterm", "Sent SIGTERM to PID {}", pid);
                // Wait briefly for graceful shutdown
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                // Try SIGKILL if still alive
                let _ = kill(pid_nix, Signal::SIGKILL);
            }
            Err(_) => {
                n!("worker_proc_del_already_dead", "Process {} may already be dead", pid);
            }
        }
    }

    #[cfg(not(unix))]
    {
        return Err(anyhow::anyhow!("Process killing not supported on this platform"));
    }

    n!("worker_proc_del_ok", "✅ Worker process PID {} deleted successfully", pid);
    Ok(())
}
