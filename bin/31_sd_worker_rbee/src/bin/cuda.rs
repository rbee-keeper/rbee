// TEAM-390: CUDA binary for SD worker
//
// Stable Diffusion worker using CUDA backend.

use clap::Parser;
use sd_worker_rbee::{http::create_router, narration::log_device_init};
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

    /// CUDA device index
    #[arg(long, env = "CUDA_DEVICE", default_value = "0")]
    cuda_device: usize,

    /// Enable flash attention (Ampere/Ada/Hopper GPUs only)
    #[arg(long, env = "USE_FLASH_ATTN")]
    use_flash_attn: bool,

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
        "Starting SD Worker (CUDA) - ID: {}, Version: {}, Port: {}, Device: {}, Flash-Attn: {}, FP16: {}",
        args.worker_id,
        args.sd_version,
        args.port,
        args.cuda_device,
        args.use_flash_attn,
        args.use_f16
    );

    // Initialize CUDA device
    log_device_init(&format!("CUDA:{}", args.cuda_device));
    let _device = device::init_cuda_device(args.cuda_device)?;
    device::verify_device(&_device)?;

    // TODO: Load SD model
    // TODO: Initialize backend with flash-attn/fp16 options
    // TODO: Register with hive

    // Create HTTP router
    let app = create_router();

    // Start server
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;
    tracing::info!("SD Worker listening on {}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}
