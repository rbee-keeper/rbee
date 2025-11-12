//! # Backend - Stable Diffusion Inference Engine
//!
//! TEAM-390: Candle-based Stable Diffusion backend with support for multiple models and backends.
//!
//! ## Architecture
//!
//! This module provides a trait-based architecture for Stable Diffusion inference:
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │         ImageModel Trait            │  ← Unified interface
//! ├─────────────────────────────────────┤
//! │  StableDiffusionModel               │  ← SD 1.5, 2.1, XL
//! │  SD3Model                           │  ← SD 3.0, 3.5
//! │  FluxModel                          │  ← FLUX.1
//! └─────────────────────────────────────┘
//!         ↓
//! ┌─────────────────────────────────────┐
//! │      Generation Engine              │  ← Request queue + generation
//! ├─────────────────────────────────────┤
//! │  Sampling Pipeline                  │  ← Text → Latents → Image
//! │  Scheduler (DDIM/Euler/DPM++)       │  ← Noise schedules
//! │  LoRA Support                       │  ← Model customization
//! └─────────────────────────────────────┘
//! ```
//!
//! ## Key Modules
//!
//! - [`traits`] - Core traits (`ImageModel`, `ModelCapabilities`)
//! - [`models`] - Model implementations (SD, SD3, FLUX)
//! - [`generation_engine`] - Request queue and generation orchestration
//! - [`model_loader`] - Model loading from HuggingFace Hub
//! - [`sampling`] - Sampling pipeline (text → latents → image)
//! - [`schedulers`] - Noise schedulers (DDIM, Euler, DPM++, etc.)
//! - [`lora`] - LoRA support for model customization
//! - [`image_utils`] - Image encoding/decoding utilities
//! - [`request_queue`] - Request queue for sequential generation
//! - [`ids`] - Type-safe IDs for models and requests
//!
//! ## RULE ZERO Compliance
//!
//! TEAM-397: This module follows RULE ZERO principles:
//! - Direct Candle types (no custom wrappers)
//! - Functions over structs where possible
//! - Trait objects for polymorphism
//! - No backwards-compatible shims
//!
//! Removed: `clip.rs`, `vae.rs`, `inference.rs` (custom wrappers)
//! Added: Direct Candle model usage via traits

// TEAM-488: Trait-based architecture for clean model abstraction
pub mod traits;

// TEAM-488: Model implementations (self-contained)
pub mod models;

// TEAM-488: Unified infrastructure
pub mod generation_engine;
pub mod model_loader;

// TEAM-487: LoRA support for model customization
pub mod lora;

// TEAM-392: Inference pipeline modules
pub mod sampling;

// TEAM-481: Modular scheduler architecture (sampler + noise schedule)
pub mod schedulers;

// TEAM-481: Type-safe IDs for models and requests
pub mod ids;

// TEAM-488: Utilities (used by jobs and other modules)
pub mod image_utils;
pub mod request_queue;

// TEAM-488: Trait exports for public API
pub use traits::{ImageModel, ModelCapabilities};
