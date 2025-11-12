// TEAM-390: CPU binary for SD worker
//
// Stable Diffusion worker using CPU backend.

use clap::Parser;
use sd_worker_rbee::{
    backend::{
        generation_engine::GenerationEngine, model_loader, models::SDVersion,
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
        "Starting SD Worker (CPU) - ID: {}, Version: {}, Port: {}",
        args.worker_id,
        args.sd_version,
        args.port
    );

    // TEAM-397: Complete implementation following TEAM-396's correct architecture

    // Initialize CPU device
    log_device_init("CPU");
    let device = device::init_cpu_device()?;
    device::verify_device(&device)?;

    // Parse SD version
    let sd_version = SDVersion::from_str(&args.sd_version)?;
    tracing::info!("Loading model: {:?}", sd_version);

    // Load model components (downloads from HuggingFace if needed)
    // TEAM-488: Updated to include LoRA support
    let model_components = model_loader::load_model(
        sd_version,
        &device,
        false, // use_f16 = false for CPU
        &[],   // loras = empty (no LoRAs for now)
        false, // quantized = false for CPU
    )?;
    tracing::info!("Model loaded successfully");

    // 1. Create request queue (returns queue and receiver)
    let (request_queue, request_rx) = RequestQueue::new();

    // 2. Create generation engine with loaded models
    // TEAM-481: model_components is now Box<dyn ImageModel>, wrap in Arc<Mutex<>>
    let engine = GenerationEngine::new(Arc::new(Mutex::new(model_components)), request_rx);

    // 3. Start engine (consumes self, spawns blocking task)
    engine.start();

    // 5. Create HTTP state with request_queue
    let app_state = AppState::new(request_queue);

    // 6. Start HTTP server
    let router = create_router(app_state);
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("✅ SD Worker (CPU) ready on port {}", args.port);
    tracing::info!("✅ Model: {:?}", sd_version);
    tracing::info!("✅ Architecture: RequestQueue/GenerationEngine pattern");
    tracing::info!("✅ Generation engine started and ready for requests");

    // Run HTTP server
    axum::serve(listener, router).await?;

    Ok(())
}
