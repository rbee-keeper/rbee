// TEAM-488: ModelComponents - direct Candle types (RULE ZERO)
//
// This struct holds the loaded Stable Diffusion model components.
// All types are direct from Candle - no wrappers, no abstractions.
//
// Based on reference/candle/candle-examples/examples/stable-diffusion/

use candle_core::{DType, Device};
use candle_transformers::models::stable_diffusion;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::backend::models::SDVersion;

/// Stable Diffusion model components
///
/// TEAM-397: Direct Candle types - RULE ZERO compliance
/// No wrappers, no custom abstractions, just Candle types
pub struct ModelComponents {
    /// Model version (v1-5, v2-1, xl, turbo, etc.)
    pub version: SDVersion,

    /// Device (CPU, CUDA, Metal)
    pub device: Device,

    /// Data type (F16, F32, BF16)
    pub dtype: DType,

    /// UNet (denoising network) - Direct Candle type
    pub unet: stable_diffusion::unet_2d::UNet2DConditionModel,

    /// VAE (encoder/decoder) - Direct Candle type
    pub vae: stable_diffusion::vae::AutoEncoderKL,

    /// Scheduler - Our implementation (trait object for flexibility)
    pub scheduler: Box<dyn crate::backend::scheduler::Scheduler>,

    /// CLIP tokenizer
    pub tokenizer: Tokenizer,

    /// CLIP config
    pub clip_config: stable_diffusion::clip::Config,

    /// CLIP weights path
    pub clip_weights: PathBuf,

    /// VAE scaling factor
    pub vae_scale: f64,
}

impl ModelComponents {
    /// Get model version
    pub fn version(&self) -> &SDVersion {
        &self.version
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
