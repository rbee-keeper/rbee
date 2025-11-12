// TEAM-390: Metal binary for SD worker
//
// Stable Diffusion worker using Metal backend (macOS only).

use clap::Parser;
use sd_worker_rbee::{
    backend::{
        generation_engine::GenerationEngine,
        model_loader,
        models::SDVersion,
        request_queue::RequestQueue,
    },
    http::{backend::AppState, routes::create_router},
    narration::log_device_init,
};
use shared_worker_rbee::device;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex; // TEAM-481: For wrapping trait object
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

    /// HTTP server port - MUST be provided by hive (no default)
    /// Port is dynamically assigned by rbee-hive using PortAssigner
    #[arg(long, env = "PORT")]
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
    
    // 1. Create request queue
    let (request_queue, request_rx) = RequestQueue::new();
    
    // Load model components
    // TEAM-488: Updated to include LoRA support
    let model_components = model_loader::load_model(
        sd_version,
        &device,
        true,  // use_f16 = true for Metal
        &[],   // loras = empty (no LoRAs for now)
        false, // quantized = false
    )?;
    tracing::info!("Model loaded successfully");
    
    // 2. Create generation engine with loaded models
    // TEAM-481: model_components is now Box<dyn ImageModel>, wrap in Arc<Mutex<>>
    let engine = GenerationEngine::new(
        Arc::new(Mutex::new(model_components)),
        request_rx,
    );
    
    // 3. Start engine (consumes self, spawns blocking task)
    engine.start();
    
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
