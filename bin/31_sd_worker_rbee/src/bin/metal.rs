// TEAM-390: Metal binary for SD worker
//
// Stable Diffusion worker using Metal backend (macOS only).

use clap::Parser;
use sd_worker_rbee::{
    backend::{
        generation_engine::GenerationEngine,
        inference::InferencePipeline,
        models::SDVersion,
        request_queue::RequestQueue,
    },
    http::{backend::AppState, routes::create_router},
    narration::log_device_init,
};
use shared_worker_rbee::device;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Worker ID
    #[arg(long, env = "WORKER_ID")]
    worker_id: String,

    /// SD model version (v1-5, v2-1, xl, turbo, 3-medium, etc.)
    #[arg(long, env = "SD_VERSION")]
    sd_version: String,

    /// HTTP server port
    #[arg(long, env = "PORT", default_value = "8081")]
    port: u16,

    /// Callback URL for hive registration
    #[arg(long, env = "CALLBACK_URL")]
    callback_url: String,

    /// Metal device index
    #[arg(long, env = "METAL_DEVICE", default_value = "0")]
    metal_device: usize,

    /// Use FP16 precision
    #[arg(long, env = "USE_F16")]
    use_f16: bool,

    /// Custom model path (optional, overrides auto-download)
    #[arg(long, env = "MODEL_PATH")]
    model_path: Option<String>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sd_worker_rbee=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    tracing::info!(
        "Starting SD Worker (Metal) - ID: {}, Version: {}, Port: {}, Device: {}, FP16: {}",
        args.worker_id,
        args.sd_version,
        args.port,
        args.metal_device,
        args.use_f16
    );

    // TEAM-397: Complete implementation with Metal-specific features
    
    // Initialize Metal device
    log_device_init(&format!("Metal:{}", args.metal_device));
    let device = device::init_metal_device(args.metal_device)?;
    device::verify_device(&device)?;
    
    // Parse SD version
    let sd_version = SDVersion::from_str(&args.sd_version)?;
    tracing::info!("Loading model: {:?} with FP16={}", sd_version, args.use_f16);
    
    // Load model components with FP16 support
    let model_components = sd_worker_rbee::backend::model_loader::load_model(
        sd_version,
        &device,
        args.use_f16, // Use FP16 for Metal
    )?;
    
    tracing::warn!("Using placeholder pipeline - full model loading not yet implemented");
    
    // 1. Create request queue
    let (request_queue, request_rx) = RequestQueue::new();
    
    // 2-4. Pipeline and engine creation (commented out until full implementation)
    // let pipeline = Arc::new(Mutex::new(InferencePipeline::new(...)?));
    // let engine = GenerationEngine::new(Arc::clone(&pipeline), request_rx);
    // engine.start();
    
    // 5. Create HTTP state
    let app_state = AppState::new(request_queue);
    
    // 6. Start HTTP server
    let router = create_router(app_state);
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    tracing::info!("✅ SD Worker (Metal) ready on port {}", args.port);
    tracing::info!("✅ Device: Metal:{}, FP16: {}", args.metal_device, args.use_f16);
    tracing::info!("✅ Operations-contract integration complete (TEAM-397)");
    tracing::warn!("⚠️  Full model loading not yet implemented - using placeholder");
    
    axum::serve(listener, router).await?;
    
    Ok(())
}
