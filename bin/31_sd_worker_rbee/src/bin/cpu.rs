// TEAM-390: CPU binary for SD worker
//
// Stable Diffusion worker using CPU backend.

use clap::Parser;
use sd_worker_rbee::{narration::log_device_init};
// TEAM-396: create_router will be used after model loading implemented
use shared_worker_rbee::device;
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

    // Initialize CPU device
    log_device_init("CPU");
    let _device = device::init_cpu_device()?;
    device::verify_device(&_device)?;

    // TEAM-396: Fixed to match LLM worker pattern
    // TODO: Load SD model and create pipeline (TEAM-397)
    // For now, show correct architecture:
    
    // 1. Create request queue (returns queue and receiver)
    // let (request_queue, request_rx) = sd_worker_rbee::backend::RequestQueue::new();
    
    // 2. Load model and create pipeline
    // let pipeline = Arc::new(Mutex::new(InferencePipeline::new(...)?));
    
    // 3. Create generation engine with dependency injection
    // let engine = sd_worker_rbee::backend::GenerationEngine::new(
    //     Arc::clone(&pipeline),
    //     request_rx,
    // );
    
    // 4. Start engine (consumes self)
    // engine.start();
    
    // 5. Create HTTP state with request_queue
    // let app_state = AppState { request_queue };
    
    // 6. Start HTTP server
    // let router = create_router(app_state);
    // let listener = tokio::net::TcpListener::bind(...).await?;
    // axum::serve(listener, router).await?;
    
    tracing::error!("Model loading not yet implemented");
    tracing::info!("✅ RequestQueue/GenerationEngine architecture fixed (TEAM-396)");
    tracing::info!("❌ Model loading needed for actual startup (TEAM-397)");
    anyhow::bail!("Model loading not yet implemented - architecture is now correct")
}
