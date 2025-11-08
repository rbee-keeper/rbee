//! Job routing and operation dispatch for rbee-hive
//!
//! TEAM-261: Mirrors queen-rbee pattern for consistency
//! TEAM-388: Refactored to use modular operation handlers
//!
//! This module handles:
//! - Parsing operation payloads into typed Operation enum
//! - Routing operations to appropriate handlers (operations module)
//! - Job lifecycle management (create, register, execute)

use anyhow::Result;
use job_server::JobRegistry;
use observability_narration_core::n;
use operations_contract::Operation;
use rbee_hive_model_catalog::ModelCatalog;
use rbee_hive_model_provisioner::ModelProvisioner;
use rbee_hive_worker_catalog::WorkerCatalog;
use std::sync::Arc;

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>,
    pub model_provisioner: Arc<ModelProvisioner>,
    pub worker_catalog: Arc<WorkerCatalog>,
    pub port_assigner: port_assigner::PortAssigner,  // TEAM-XXX: Dynamic port allocation
}

pub use jobs_contract::JobResponse;

/// Create a new job and store its payload
///
/// TEAM-261: Server generates job_id, stores payload for deferred execution
pub fn create_job(registry: Arc<JobRegistry<String>>, payload: serde_json::Value) -> String {
    // ============================================================
    // BUG FIX: TEAM-389 | Missing SSE channel creation
    // ============================================================
    // SUSPICION:
    // - TEAM-388 mentioned "Job channel not found" error
    // - Suspected it was pre-existing, but refactoring changed create_job()
    //
    // INVESTIGATION:
    // - Compared job_router_old.rs (line 62) with new job_router.rs
    // - Found sse_sink::create_job_channel() call was DELETED
    // - Old version: Lines 54-67 had channel creation
    // - New version: Lines 35-39 missing channel creation
    // - Checked queen-rbee/src/job_router.rs - it DOES create channels (line 62)
    //
    // ROOT CAUSE:
    // - TEAM-388 refactoring simplified create_job() signature
    // - Changed from async fn returning Result<JobResponse> to sync fn returning String
    // - Accidentally deleted the sse_sink::create_job_channel() call (line 62 old version)
    // - Without channel creation, sse_sink::take_job_receiver() returns None
    // - This triggers "Job channel not found" error in http/jobs.rs line 130
    //
    // FIX:
    // - Restored the missing sse_sink::create_job_channel() call
    // - Used 10000 buffer size (TEAM-378 increased for high-volume ops like cargo build)
    // - Placed AFTER set_payload() to match original order
    //
    // TESTING:
    // - cargo build --bin rbee-hive (compilation check)
    // - ./rbee model ls (should show models, not "Job channel not found")
    // - ./rbee model remove (should work without SSE errors)
    // - ./rbee worker install --name llama-cli (high-volume narration test)
    // ============================================================
    
    let job_id = registry.create_job();
    registry.set_payload(&job_id, payload);
    
    // TEAM-389: Restore SSE channel creation (accidentally removed by TEAM-388)
    // TEAM-378: 10000 buffer for high-volume operations (cargo build produces many messages)
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 10000);
    
    n!("job_create", "Job {} created, waiting for client connection", job_id);
    
    job_id
}

/// Execute a job and stream results via SSE
///
/// TEAM-261: Mirrors queen-rbee pattern
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();
    let state_clone = state.clone();

    job_server::execute_and_stream(
        job_id,
        registry.clone(),
        move |job_id, payload| route_operation(job_id, payload, state_clone.clone()),
        None,
    )
    .await
}

/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
/// TEAM-385: Context now injected by job-server, no manual setup needed!
/// TEAM-388: Delegates to modular operation handlers
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name().to_string();

    n!("route_job", "Executing operation: {}", operation_name);
    
    execute_operation(operation, operation_name, job_id, state).await
}

/// TEAM-388: Execute the actual operation logic (delegates to operation modules)
async fn execute_operation(
    operation: Operation,
    operation_name: String,
    job_id: String,
    state: JobState,
) -> Result<()> {
    // ============================================================================
    // OPERATION ROUTING
    // ============================================================================
    //
    // TEAM-388: Operations are now handled by focused modules:
    // - operations::hive - Hive management operations
    // - operations::worker - Worker catalog and process operations
    // - operations::model - Model catalog and provisioning operations
    //
    match &operation {
        // Hive operations
        Operation::HiveCheck { .. } => {
            rbee_hive::operations::handle_hive_operation(&operation).await?;
        }

        // Worker catalog operations
        Operation::WorkerCatalogList(_)
        | Operation::WorkerCatalogGet(_)
        | Operation::WorkerInstalledGet(_)
        | Operation::WorkerInstall(_)
        | Operation::WorkerRemove(_)
        | Operation::WorkerListInstalled(_)
        | Operation::WorkerSpawn(_)
        | Operation::WorkerProcessList(_)
        | Operation::WorkerProcessGet(_)
        | Operation::WorkerProcessDelete(_) => {
            rbee_hive::operations::handle_worker_operation(
                &operation,
                state.worker_catalog.clone(),
                &state.port_assigner,  // TEAM-XXX: Pass port assigner
                &job_id,
                || state.registry.get_cancellation_token(&job_id),
            )
            .await?;
        }

        // Model operations
        Operation::ModelDownload(_)
        | Operation::ModelList(_)
        | Operation::ModelGet(_)
        | Operation::ModelDelete(_)
        | Operation::ModelLoad(_)
        | Operation::ModelUnload(_) => {
            rbee_hive::operations::handle_model_operation(
                &operation,
                state.model_catalog.clone(),
                state.model_provisioner.clone(),
                &job_id,
                || state.registry.get_cancellation_token(&job_id),
            )
            .await?;
        }

        // ========================================================================
        // INFERENCE REJECTION - CRITICAL ARCHITECTURE NOTE (TEAM-261)
        // ========================================================================
        //
        // ⚠️  INFER SHOULD NOT BE IN HIVE!
        //
        // Why?
        // - Hive only manages worker LIFECYCLE (spawn/stop/list)
        // - Queen handles inference routing DIRECTLY to workers
        // - Queen → Worker is DIRECT HTTP (circumvents hive)
        // - This is INTENTIONAL for performance and simplicity
        //
        // If you see Infer here, something is wrong with the routing in queen-rbee!
        //
        // See: bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md
        //
        Operation::Infer { .. } => {
            return Err(anyhow::anyhow!(
                "Infer operation should NOT be routed to hive! \
                 Queen should route inference directly to workers. \
                 This indicates a routing bug in queen-rbee/src/job_router.rs. \
                 See bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md for details."
            ));
        }

        // Unsupported operations (handled by queen-rbee)
        _ => {
            return Err(anyhow::anyhow!(
                "Operation '{}' is not supported by rbee-hive (should be handled by queen-rbee)",
                operation_name
            ));
        }
    }

    Ok(())
}
