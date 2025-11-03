// TEAM-390: Extracted from llm-worker-rbee/src/heartbeat.rs
// Original: TEAM-164, TEAM-261, TEAM-380
//
//! Worker heartbeat handling
//!
//! Shared between all worker types (LLM, SD, etc.)
//!
//! **What lives here:**
//! - Worker sends heartbeats to queen
//! - Periodic heartbeat task
//! - Worker health status reporting

use anyhow::Result;
use observability_narration_core::n;
use worker_contract::{WorkerHeartbeat, WorkerInfo};

/// Send heartbeat to queen
///
/// **Flow:**
/// 1. Build WorkerHeartbeat with full WorkerInfo
/// 2. Send POST /v1/worker-heartbeat to queen
/// 3. Return acknowledgement
///
/// This is called periodically (e.g., every 30s) to signal worker is alive
pub async fn send_heartbeat_to_queen(
    worker_info: &WorkerInfo,
    queen_url: &str,
) -> Result<()> {
    n!("send_heartbeat", "Sending heartbeat to queen at {}", queen_url);

    let _heartbeat = WorkerHeartbeat::new(worker_info.clone());

    // TODO: Implement HTTP POST to queen
    // POST {queen_url}/v1/worker-heartbeat with heartbeat
    // let client = reqwest::Client::new();
    // client.post(format!("{}/v1/worker-heartbeat", queen_url))
    //     .json(&_heartbeat)
    //     .send()
    //     .await?;

    Ok(())
}

/// Start periodic heartbeat task
///
/// **Flow:**
/// 1. Spawn tokio task
/// 2. Every 30 seconds, send heartbeat to queen with full WorkerInfo
/// 3. Continue until task is cancelled
///
/// This runs in the background for the lifetime of the worker
pub fn start_heartbeat_task(
    worker_info: WorkerInfo,
    queen_url: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));

        loop {
            interval.tick().await;

            if let Err(e) = send_heartbeat_to_queen(&worker_info, &queen_url).await {
                eprintln!("Failed to send heartbeat: {}", e);
            }
        }
    })
}
