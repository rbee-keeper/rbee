//! # sd-worker-rbee
//!
//! **Stable Diffusion inference worker for rbee**
//!
//! TEAM-390: Candle-based Stable Diffusion worker that generates images from text prompts.
//! Spawned by `rbee-hive` and provides HTTP endpoints for image generation.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │       sd-worker-rbee Worker          │
//! ├─────────────────────────────────────┤
//! │  HTTP Server (Axum)                 │
//! │  ├─ /health                         │
//! │  ├─ /execute (text-to-image)        │
//! │  ├─ /img2img                        │
//! │  └─ /inpaint                        │
//! ├─────────────────────────────────────┤
//! │  Generation Engine                  │
//! │  ├─ Request Queue                   │
//! │  ├─ Model Loader                    │
//! │  └─ Sampling Pipeline               │
//! ├─────────────────────────────────────┤
//! │  Candle Backend                     │
//! │  ├─ CLIP Text Encoder               │
//! │  ├─ UNet / MMDiT / FLUX             │
//! │  ├─ VAE (Encoder/Decoder)           │
//! │  └─ Scheduler (DDIM/Euler/etc)      │
//! ├─────────────────────────────────────┤
//! │  Candle (ML Framework)              │
//! │  └─ Device (CPU/CUDA/Metal)         │
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - **Text-to-image** generation
//! - **Image-to-image** transformation
//! - **Inpainting** support
//! - **Multiple backends** (CPU, CUDA, Metal)
//! - **Streaming progress** via SSE
//! - **Multiple SD models** (1.5, 2.1, XL, Turbo, SD3, FLUX)
//! - **LoRA** support for model customization
//! - **Modular scheduler** architecture (DDIM, Euler, DPM++, etc.)
//!
//! ## Modules
//!
//! - [`backend`] - Core inference engine and model implementations
//! - [`http`] - HTTP server and API endpoints
//! - [`jobs`] - Job handlers for different generation types
//! - [`job_router`] - Routes incoming jobs to appropriate handlers
//! - [`device`] - Device management (CPU/CUDA/Metal)
//! - [`error`] - Error types and result aliases
//! - [`narration`] - Structured logging for observability
//!
//! ## Example Usage
//!
//! This worker is spawned by `rbee-hive` with configuration:
//!
//! ```bash
//! sd-worker-rbee \
//!   --port 8080 \
//!   --model-path ~/.cache/rbee/models/stable-diffusion-v1-5 \
//!   --device cuda \
//!   --worker-id worker-123
//! ```
//!
//! ## Building
//!
//! ```bash
//! # CPU variant
//! cargo build --release --no-default-features --features cpu
//!
//! # CUDA variant (requires CUDA toolkit)
//! cargo build --release --no-default-features --features cuda
//!
//! # Metal variant (macOS only)
//! cargo build --release --no-default-features --features metal
//! ```
//!
//! TEAM-482: Sampler/scheduler separation for flexible noise schedules

#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::similar_names,
    clippy::too_many_lines,
    // TEAM-482: Disable documentation lints - focus on code quality
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::missing_const_for_fn
)]

/// Core inference engine and model implementations
///
/// Contains the Candle-based Stable Diffusion backend with support for:
/// - Model loading and management
/// - Generation pipeline
/// - Schedulers and samplers
/// - LoRA support
/// - Image utilities
pub mod backend;

/// Device management (CPU/CUDA/Metal)
///
/// Re-exports device initialization from `shared-worker-rbee`.
pub mod device;

/// Error types and result aliases
///
/// Provides [`Error`] enum and [`Result<T>`] type alias.
pub mod error;

/// HTTP server and API endpoints
///
/// Axum-based HTTP server with endpoints for:
/// - `/health` - Health check
/// - `/execute` - Text-to-image generation
/// - `/img2img` - Image-to-image transformation
/// - `/inpaint` - Inpainting
pub mod http;

/// Routes incoming jobs to appropriate handlers
///
/// TEAM-487: Job routing logic separated from handlers
pub mod job_router;

/// Job handlers for different generation types
///
/// TEAM-487: Job handlers separated from router
pub mod jobs;

/// Structured logging for observability
///
/// Uses `narration-core` for structured event logging.
pub mod narration;

pub use error::{Error, Result};
