// TEAM-390: Stable Diffusion model definitions and version selection
//
// Defines supported SD models and their configurations.

use crate::error::Result;
use candle_core::Device;

pub mod sd_config;

/// Supported Stable Diffusion model versions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SDVersion {
    /// Stable Diffusion 1.5 (512x512)
    V1_5,
    /// Stable Diffusion 1.5 Inpainting
    V1_5Inpaint,
    /// Stable Diffusion 2.1 (768x768)
    V2_1,
    /// Stable Diffusion 2.1 Inpainting
    V2Inpaint,
    /// Stable Diffusion XL (1024x1024)
    XL,
    /// Stable Diffusion XL Inpainting
    XLInpaint,
    /// Stable Diffusion XL Turbo (4-step)
    Turbo,
}

impl SDVersion {
    /// Get HuggingFace repository for this model version
    pub fn repo(&self) -> &'static str {
        match self {
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::V1_5Inpaint => "stable-diffusion-v1-5/stable-diffusion-inpainting",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V2Inpaint => "stabilityai/stable-diffusion-2-inpainting",
            Self::XL => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::XLInpaint => "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    /// Get default image dimensions for this model
    pub fn default_size(&self) -> (usize, usize) {
        match self {
            Self::V1_5 | Self::V1_5Inpaint => (512, 512),
            Self::V2_1 | Self::V2Inpaint => (768, 768),
            Self::XL | Self::XLInpaint | Self::Turbo => (1024, 1024),
        }
    }

    /// Get default number of inference steps
    pub fn default_steps(&self) -> usize {
        match self {
            Self::Turbo => 4, // Turbo is optimized for 4 steps
            _ => 20,
        }
    }

    /// Get default guidance scale
    pub fn default_guidance_scale(&self) -> f64 {
        match self {
            Self::Turbo => 0.0, // Turbo doesn't use guidance
            _ => 7.5,
        }
    }

    /// Check if this is an inpainting model
    pub fn is_inpainting(&self) -> bool {
        matches!(self, Self::V1_5Inpaint | Self::V2Inpaint | Self::XLInpaint)
    }

    /// Check if this is an XL-based model
    pub fn is_xl(&self) -> bool {
        matches!(self, Self::XL | Self::XLInpaint | Self::Turbo)
    }

    /// Parse from string (e.g., "v1-5", "xl", "turbo")
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "v1-5" | "v1.5" | "1.5" => Ok(Self::V1_5),
            "v1-5-inpaint" | "v1.5-inpaint" => Ok(Self::V1_5Inpaint),
            "v2-1" | "v2.1" | "2.1" => Ok(Self::V2_1),
            "v2-inpaint" | "v2.1-inpaint" => Ok(Self::V2Inpaint),
            "xl" => Ok(Self::XL),
            "xl-inpaint" => Ok(Self::XLInpaint),
            "turbo" => Ok(Self::Turbo),
            _ => Err(crate::error::Error::InvalidInput(format!(
                "Unknown SD version: {}. Supported: v1-5, v2-1, xl, turbo",
                s
            ))),
        }
    }
}

/// Model file paths for different components
#[derive(Debug, Clone, Copy)]
pub enum ModelFile {
    Tokenizer,
    Tokenizer2, // For XL models
    Clip,
    Clip2, // For XL models
    Unet,
    Vae,
}

impl ModelFile {
    /// Get the file path for this component
    pub fn path(&self, _version: SDVersion, use_f16: bool) -> &'static str {
        match self {
            Self::Tokenizer => "tokenizer/tokenizer_config.json",
            Self::Tokenizer2 => "tokenizer_2/tokenizer_config.json",
            Self::Clip => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
            Self::Clip2 => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
            Self::Unet => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
            Self::Vae => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    /// Get tokenizer repository (different for some models)
    pub fn tokenizer_repo(version: SDVersion) -> &'static str {
        match version {
            SDVersion::V1_5 | SDVersion::V1_5Inpaint | SDVersion::V2_1 | SDVersion::V2Inpaint => {
                "openai/clip-vit-base-patch32"
            }
            SDVersion::XL | SDVersion::XLInpaint | SDVersion::Turbo => {
                "openai/clip-vit-large-patch14"
            }
        }
    }
}

/// Model components loaded into memory
/// TEAM-397: RULE ZERO - Direct Candle types, NO wrappers
pub struct ModelComponents {
    pub version: SDVersion,
    pub device: Device,
    pub dtype: candle_core::DType,
    
    // ✅ Direct Candle types (no wrappers)
    pub tokenizer: tokenizers::Tokenizer,
    pub clip_config: candle_transformers::models::stable_diffusion::clip::Config,
    pub clip_weights: std::path::PathBuf,
    pub unet: candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel,
    pub vae: candle_transformers::models::stable_diffusion::vae::AutoEncoderKL,
    pub scheduler: crate::backend::scheduler::DDIMScheduler,  // TEAM-397: Use our scheduler
    pub vae_scale: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        assert_eq!(SDVersion::from_str("v1-5").unwrap(), SDVersion::V1_5);
        assert_eq!(SDVersion::from_str("xl").unwrap(), SDVersion::XL);
        assert_eq!(SDVersion::from_str("turbo").unwrap(), SDVersion::Turbo);
        assert!(SDVersion::from_str("invalid").is_err());
    }

    #[test]
    fn test_default_sizes() {
        assert_eq!(SDVersion::V1_5.default_size(), (512, 512));
        assert_eq!(SDVersion::V2_1.default_size(), (768, 768));
        assert_eq!(SDVersion::XL.default_size(), (1024, 1024));
    }

    #[test]
    fn test_is_xl() {
        assert!(!SDVersion::V1_5.is_xl());
        assert!(SDVersion::XL.is_xl());
        assert!(SDVersion::Turbo.is_xl());
    }
}
