// TEAM-390: Stable Diffusion model definitions and version selection
// TEAM-482: Added architecture constants for type-safe model identifiers
//
// Defines supported SD models and their configurations.

// TEAM-488: Self-contained model implementations
pub mod flux;
pub mod stable_diffusion;

// TEAM-482: Shared helpers to avoid code duplication
pub mod shared;

/// TEAM-482: Architecture constants - single source of truth for model identifiers
///
/// Adopted from LLM Worker for type safety and maintainability.
/// Benefits:
/// - Compile-time typo detection
/// - Single source of truth
/// - Easy refactoring (change in one place)
/// - IDE autocomplete support
pub mod arch {
    /// Stable Diffusion architecture identifier
    pub const STABLE_DIFFUSION: &str = "stable-diffusion";

    /// FLUX architecture identifier
    pub const FLUX: &str = "flux";

    /// Model variant constants
    pub mod variants {
        // Stable Diffusion variants
        pub const SD_1_5: &str = "sd1.5";
        pub const SD_1_5_INPAINT: &str = "sd1.5-inpaint";
        pub const SD_2_1: &str = "sd2.1";
        pub const SD_2_INPAINT: &str = "sd2-inpaint";
        pub const SD_XL: &str = "sdxl";
        pub const SD_XL_INPAINT: &str = "sdxl-inpaint";
        pub const SD_TURBO: &str = "sdxl-turbo";

        // FLUX variants
        pub const FLUX_DEV: &str = "flux-dev";
        pub const FLUX_SCHNELL: &str = "flux-schnell";
    }
}

/// Supported Stable Diffusion model versions
/// TEAM-483: Added FLUX support
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

    // TEAM-483: FLUX models
    /// FLUX.1-dev (50 steps, guidance-distilled, best quality)
    FluxDev,
    /// FLUX.1-schnell (4 steps, fast, good quality)
    FluxSchnell,
}

impl SDVersion {
    /// Get `HuggingFace` repository for this model version
    #[must_use]
    pub fn repo(&self) -> &'static str {
        match self {
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::V1_5Inpaint => "stable-diffusion-v1-5/stable-diffusion-inpainting",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V2Inpaint => "stabilityai/stable-diffusion-2-inpainting",
            Self::XL => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::XLInpaint => "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            Self::Turbo => "stabilityai/sdxl-turbo",
            Self::FluxDev => "black-forest-labs/FLUX.1-dev",
            Self::FluxSchnell => "black-forest-labs/FLUX.1-schnell",
        }
    }

    /// Get default image dimensions for this model
    #[must_use]
    pub fn default_size(&self) -> (usize, usize) {
        match self {
            Self::V1_5 | Self::V1_5Inpaint => (512, 512),
            Self::V2_1 | Self::V2Inpaint => (768, 768),
            // TEAM-482: Merged identical arms (clippy::match_same_arms)
            Self::XL | Self::XLInpaint | Self::Turbo | Self::FluxDev | Self::FluxSchnell => {
                (1024, 1024)
            }
        }
    }

    /// Get default number of inference steps
    #[must_use]
    pub fn default_steps(&self) -> usize {
        match self {
            // TEAM-482: Merged identical arms (clippy::match_same_arms)
            Self::Turbo | Self::FluxSchnell => 4, // Turbo/Schnell optimized for 4 steps
            Self::FluxDev => 50,                  // Dev needs more steps for quality
            _ => 20,
        }
    }

    /// Get default guidance scale
    #[must_use]
    pub fn default_guidance_scale(&self) -> f64 {
        match self {
            // TEAM-482: Merged identical arms (clippy::match_same_arms)
            Self::Turbo | Self::FluxSchnell => 0.0, // Turbo/Schnell don't use guidance
            Self::FluxDev => 3.5,                   // FLUX uses lower guidance than SD
            _ => 7.5,
        }
    }

    /// Check if this is an inpainting model
    #[must_use]
    pub fn is_inpainting(&self) -> bool {
        matches!(self, Self::V1_5Inpaint | Self::V2Inpaint | Self::XLInpaint)
    }

    /// Check if this is an XL-based model
    #[must_use]
    pub fn is_xl(&self) -> bool {
        matches!(self, Self::XL | Self::XLInpaint | Self::Turbo)
    }

    /// Check if this is a FLUX model
    #[must_use]
    pub fn is_flux(&self) -> bool {
        matches!(self, Self::FluxDev | Self::FluxSchnell)
    }

    // TEAM-399: Config methods for model initialization
    // Based on reference/candle/candle-transformers/src/models/stable_diffusion/mod.rs

    /// Get CLIP config for this model version
    #[must_use]
    pub fn clip_config(&self) -> candle_transformers::models::stable_diffusion::clip::Config {
        use candle_transformers::models::stable_diffusion::clip::Config;
        match self {
            Self::V1_5 | Self::V1_5Inpaint => Config::v1_5(),
            Self::V2_1 | Self::V2Inpaint => Config::v2_1(),
            Self::XL | Self::XLInpaint | Self::Turbo => Config::sdxl(),
            Self::FluxDev | Self::FluxSchnell => panic!("FLUX models don't use SD CLIP config"),
        }
    }

    /// Get `UNet` config for this model version
    /// Manually constructed like in `StableDiffusionConfig::v1_5()`
    #[must_use]
    pub fn unet_config(
        &self,
    ) -> candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModelConfig {
        use candle_transformers::models::stable_diffusion::unet_2d::{
            BlockConfig, UNet2DConditionModelConfig,
        };

        let bc = |out_channels, use_cross_attn, attention_head_dim| BlockConfig {
            out_channels,
            use_cross_attn,
            attention_head_dim,
        };

        match self {
            Self::V1_5 | Self::V1_5Inpaint => {
                // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json
                UNet2DConditionModelConfig {
                    blocks: vec![
                        bc(320, Some(1), 8),
                        bc(640, Some(1), 8),
                        bc(1280, Some(1), 8),
                        bc(1280, None, 8),
                    ],
                    center_input_sample: false,
                    cross_attention_dim: 768,
                    downsample_padding: 1,
                    flip_sin_to_cos: true,
                    freq_shift: 0.,
                    layers_per_block: 2,
                    mid_block_scale_factor: 1.,
                    norm_eps: 1e-5,
                    norm_num_groups: 32,
                    sliced_attention_size: None,
                    use_linear_projection: false,
                }
            }
            Self::V2_1 | Self::V2Inpaint => {
                // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/unet/config.json
                UNet2DConditionModelConfig {
                    blocks: vec![
                        bc(320, Some(1), 8),
                        bc(640, Some(1), 8),
                        bc(1280, Some(1), 8),
                        bc(1280, None, 8),
                    ],
                    center_input_sample: false,
                    cross_attention_dim: 1024,
                    downsample_padding: 1,
                    flip_sin_to_cos: true,
                    freq_shift: 0.,
                    layers_per_block: 2,
                    mid_block_scale_factor: 1.,
                    norm_eps: 1e-5,
                    norm_num_groups: 32,
                    sliced_attention_size: None,
                    use_linear_projection: true,
                }
            }
            Self::XL | Self::XLInpaint | Self::Turbo => {
                // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/config.json
                UNet2DConditionModelConfig {
                    blocks: vec![bc(320, Some(2), 8), bc(640, Some(2), 8), bc(1280, Some(10), 8)],
                    center_input_sample: false,
                    cross_attention_dim: 2048,
                    downsample_padding: 1,
                    flip_sin_to_cos: true,
                    freq_shift: 0.,
                    layers_per_block: 2,
                    mid_block_scale_factor: 1.,
                    norm_eps: 1e-5,
                    norm_num_groups: 32,
                    sliced_attention_size: None,
                    use_linear_projection: true,
                }
            }
            Self::FluxDev | Self::FluxSchnell => {
                panic!("FLUX models don't use UNet config")
            }
        }
    }

    /// Get VAE config for this model version
    /// Manually constructed like in `StableDiffusionConfig::v1_5()`
    #[must_use]
    pub fn vae_config(
        &self,
    ) -> candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig {
        use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

        match self {
            Self::V1_5 | Self::V1_5Inpaint | Self::V2_1 | Self::V2Inpaint => {
                // https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json
                AutoEncoderKLConfig {
                    block_out_channels: vec![128, 256, 512, 512],
                    layers_per_block: 2,
                    latent_channels: 4,
                    norm_num_groups: 32,
                    use_quant_conv: true,
                    use_post_quant_conv: true,
                }
            }
            Self::XL | Self::XLInpaint | Self::Turbo => {
                // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/vae/config.json
                AutoEncoderKLConfig {
                    block_out_channels: vec![128, 256, 512, 512],
                    layers_per_block: 2,
                    latent_channels: 4,
                    norm_num_groups: 32,
                    use_quant_conv: true,
                    use_post_quant_conv: true,
                }
            }
            Self::FluxDev | Self::FluxSchnell => {
                panic!("FLUX models don't use SD VAE config")
            }
        }
    }

    /// Parse from string (e.g., "v1-5", "xl", "turbo", "flux-dev", "flux-schnell")
    ///
    /// TEAM-482: Custom parsing logic (not std::str::FromStr trait)
    /// We use a custom method name to avoid confusion with the trait
    pub fn parse_version(s: &str) -> crate::error::Result<Self> {
        match s.to_lowercase().as_str() {
            "v1-5" | "v1.5" | "1.5" => Ok(Self::V1_5),
            "v1-5-inpaint" | "v1.5-inpaint" => Ok(Self::V1_5Inpaint),
            "v2-1" | "v2.1" | "2.1" => Ok(Self::V2_1),
            "v2-inpaint" | "v2.1-inpaint" => Ok(Self::V2Inpaint),
            "xl" => Ok(Self::XL),
            "xl-inpaint" => Ok(Self::XLInpaint),
            "turbo" => Ok(Self::Turbo),
            "flux-dev" | "flux.1-dev" | "flux1-dev" => Ok(Self::FluxDev),
            "flux-schnell" | "flux.1-schnell" | "flux1-schnell" => Ok(Self::FluxSchnell),
            _ => Err(crate::error::Error::InvalidInput(format!(
                "Unknown SD version: {s}. Supported: v1-5, v2-1, xl, turbo, flux-dev, flux-schnell"
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
    #[must_use]
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
    #[must_use]
    pub fn tokenizer_repo(version: SDVersion) -> &'static str {
        match version {
            SDVersion::V1_5 | SDVersion::V1_5Inpaint | SDVersion::V2_1 | SDVersion::V2Inpaint => {
                "openai/clip-vit-base-patch32"
            }
            SDVersion::XL | SDVersion::XLInpaint | SDVersion::Turbo => {
                "openai/clip-vit-large-patch14"
            }
            SDVersion::FluxDev | SDVersion::FluxSchnell => {
                panic!("FLUX models use different tokenizers")
            }
        }
    }
}

// TEAM-488: ModelComponents moved to stable_diffusion::ModelComponents
// Use stable_diffusion module instead

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        assert_eq!(SDVersion::parse_version("v1-5").unwrap(), SDVersion::V1_5);
        assert_eq!(SDVersion::parse_version("xl").unwrap(), SDVersion::XL);
        assert_eq!(SDVersion::parse_version("turbo").unwrap(), SDVersion::Turbo);
        assert!(SDVersion::parse_version("invalid").is_err());
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
