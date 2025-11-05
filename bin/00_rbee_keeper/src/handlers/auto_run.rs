// TEAM-416: Auto-run logic for models/workers
// Implements one-click installation from marketplace

use anyhow::Result;
use job_client::JobClient;
use operations_contract::{Operation, ModelDownloadRequest, WorkerSpawnRequest};

/// Auto-run model installation
///
/// Downloads the model and spawns a worker to run it.
/// This is the "one-click install" experience from the marketplace.
///
/// # Arguments
/// * `model_id` - Model identifier (e.g., "meta-llama/Llama-3.2-1B")
/// * `hive_id` - Hive to install on (default: "localhost")
///
/// # Example
/// ```no_run
/// auto_run_model("meta-llama/Llama-3.2-1B".to_string(), "localhost".to_string()).await?;
/// ```
pub async fn auto_run_model(model_id: String, hive_id: String) -> Result<()> {
    println!("🚀 Auto-running model: {}", model_id);
    
    // TEAM-416: Step 1 - Download model from HuggingFace
    let client = JobClient::new("http://localhost:7835"); // rbee-hive port
    
    let download_op = Operation::ModelDownload(ModelDownloadRequest {
        hive_id: hive_id.clone(),
        model: model_id.clone(),
    });
    
    println!("📥 Downloading model...");
    let (_job_id, stream_fut) = client.submit_and_stream(download_op, |line| {
        println!("   {}", line);
        Ok(())
    }).await?;
    stream_fut.await?;
    
    // TEAM-416: Step 2 - Spawn worker with the model
    // Default to CPU worker for maximum compatibility
    let spawn_op = Operation::WorkerSpawn(WorkerSpawnRequest {
        hive_id: hive_id.clone(),
        model: model_id.clone(),
        worker: "cpu".to_string(), // Default to CPU for compatibility
        device: 0, // Default device
    });
    
    println!("🐝 Spawning worker...");
    let (_job_id, stream_fut) = client.submit_and_stream(spawn_op, |line| {
        println!("   {}", line);
        Ok(())
    }).await?;
    stream_fut.await?;
    
    println!("✅ Model ready: {}", model_id);
    Ok(())
}

/// Auto-run worker installation
///
/// Spawns a worker process. Used when installing a worker without a specific model.
///
/// # Arguments
/// * `worker_type` - Worker type (e.g., "cpu", "cuda", "metal")
/// * `hive_id` - Hive to spawn on (default: "localhost")
///
/// # Example
/// ```no_run
/// auto_run_worker("cpu".to_string(), "localhost".to_string()).await?;
/// ```
pub async fn auto_run_worker(worker_type: String, hive_id: String) -> Result<()> {
    println!("🚀 Auto-running worker: {}", worker_type);
    
    let client = JobClient::new("http://localhost:7835"); // rbee-hive port
    
    let spawn_op = Operation::WorkerSpawn(WorkerSpawnRequest {
        hive_id: hive_id.clone(),
        model: "".to_string(), // No specific model
        worker: worker_type.clone(),
        device: 0, // Default device
    });
    
    println!("🐝 Spawning worker...");
    let (_job_id, stream_fut) = client.submit_and_stream(spawn_op, |line| {
        println!("   {}", line);
        Ok(())
    }).await?;
    stream_fut.await?;
    
    println!("✅ Worker ready: {}", worker_type);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // TEAM-416: Unit tests for auto-run functions
    // Note: These are integration tests that require rbee-hive to be running
    // They are marked as #[ignore] to prevent CI failures
    
    #[tokio::test]
    #[ignore = "requires rbee-hive running"]
    async fn test_auto_run_model() {
        let result = auto_run_model(
            "meta-llama/Llama-3.2-1B".to_string(),
            "localhost".to_string()
        ).await;
        
        // Should succeed if rbee-hive is running
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    #[ignore = "requires rbee-hive running"]
    async fn test_auto_run_worker() {
        let result = auto_run_worker(
            "cpu".to_string(),
            "localhost".to_string()
        ).await;
        
        // Should succeed if rbee-hive is running
        assert!(result.is_ok());
    }
}
