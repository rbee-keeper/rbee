// TEAM-488: ROCm GPU worker binary
//!
//! Uses AMD ROCm for GPU inference with strict device residency.
//! This binary is feature-gated to ROCm backend only.
//!
//! Created by: TEAM-488

use anyhow::Result;
use clap::Parser;
use llm_worker_rbee::{backend::CandleInferenceBackend, setup_worker_with_backend, HttpServer};
use std::net::SocketAddr;

/// CLI arguments for ROCm worker daemon
#[derive(Parser, Debug)]
#[command(name = "llorch-rocm-candled")]
#[command(about = "AMD ROCm GPU Candle-based multi-model worker daemon")]
struct Args {
    /// Worker ID (UUID) - assigned by pool-managerd
    #[arg(long)]
    worker_id: String,

    /// Model file path (GGUF or `SafeTensors` format)
    #[arg(long)]
    model: String,

    /// Model reference (e.g., "hf:tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    #[arg(long)]
    model_ref: String,

    /// HTTP server port - assigned by pool-managerd
    #[arg(long)]
    port: u16,

    /// Hive URL - where to send heartbeats
    #[arg(long)]
    hive_url: String,

    /// ROCm device ID (default: 0)
    #[arg(long, default_value = "0")]
    rocm_device: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    // Initialize tracing (JSON format for structured logging)
    tracing_subscriber::fmt().with_target(false).json().init();

    let args = Args::parse();

    tracing::info!(
        worker_id = %args.worker_id,
        model = %args.model,
        port = args.port,
        rocm_device = args.rocm_device,
        backend = "rocm",
        "Starting llorch-rocm-candled"
    );

    // ============================================================
    // STEP 1: Load model to ROCm GPU
    // ============================================================
    // TEAM-488: Load model with auto-detected architecture
    // Device is compile-time (ROCm), GPU ID passed at runtime
    tracing::info!(model = %args.model, rocm_device = args.rocm_device, "Loading model to ROCm GPU...");
    let mut backend = CandleInferenceBackend::load(&args.model, args.rocm_device)?;
    tracing::info!("Model loaded successfully on ROCm GPU {}", args.rocm_device);

    // ============================================================
    // STEP 2.5: GPU Warmup
    // ============================================================
    // TEAM-488: Warmup ROCm GPU to eliminate cold start overhead
    backend.warmup()?;
    tracing::info!("ROCm GPU warmup complete - ready for inference");

    // ============================================================
    // STEP 3: Start heartbeat task
    // ============================================================
    tracing::info!("Starting heartbeat task");

    let worker_info = worker_contract::WorkerInfo {
        id: args.worker_id.clone(),
        model_id: args.model_ref.clone(),
        device: format!("rocm:{}", args.rocm_device),
        port: args.port,
        status: worker_contract::WorkerStatus::Ready,
        implementation: "llm-worker-rbee-rocm".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let _heartbeat_handle =
        llm_worker_rbee::heartbeat::start_heartbeat_task(worker_info, args.hive_url.clone());
    tracing::info!("Heartbeat task started (30s interval)");

    // ============================================================
    // STEP 4: Start HTTP server (runs forever)
    // ============================================================
    tracing::info!("Worker ready, starting HTTP server");

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));

    // TEAM-102: Load API token for authentication
    let expected_token = std::env::var("LLORCH_API_TOKEN").unwrap_or_else(|_| {
        tracing::info!("⚠️  LLORCH_API_TOKEN not set - using dev mode (no auth)");
        String::new()
    });

    if !expected_token.is_empty() {
        tracing::info!("✅ API token loaded (authentication enabled)");
    }

    // Setup job-based architecture
    let router = setup_worker_with_backend(backend, expected_token);
    let server = HttpServer::new(addr, router).await?;

    tracing::info!("llorch-rocm-candled ready on port {} (ROCm GPU {})", args.port, args.rocm_device);

    // Run forever (until killed by pool-managerd)
    server.run().await?;

    Ok(())
}
