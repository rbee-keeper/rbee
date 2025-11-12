// TEAM-488: Stable Diffusion model implementation
//
// Self-contained implementation of Stable Diffusion models:
// - SD 1.5, SD 2.1, SDXL, SDXL Turbo
// - Text-to-image, image-to-image, inpainting
// - LoRA support
//
// This module is organized for clarity:
// - components.rs: ModelComponents (direct Candle types)
// - loader.rs: Model loading from HuggingFace
// - generator.rs: Generation functions (txt2img, img2img, inpaint)
// - config.rs: SD-specific configuration

mod components;
mod config;
mod generation;
mod loader;

pub use components::ModelComponents;
pub use config::SDConfig;
pub use loader::{load_model_simple, load_stable_diffusion_with_lora};

use crate::backend::traits::{GenerationRequest, ImageModel, ModelCapabilities};
use crate::error::Result;
use image::DynamicImage;

/// Stable Diffusion model with ImageModel trait implementation
pub struct StableDiffusionModel {
    components: ModelComponents,
    capabilities: ModelCapabilities,
}

impl StableDiffusionModel {
    /// Create new SD model from loaded components
    pub fn new(components: ModelComponents) -> Self {
        // Determine capabilities based on version
        let is_inpaint = components.version.is_inpainting();
        let default_steps = components.version.default_steps();
        
        let capabilities = ModelCapabilities {
            img2img: true,
            inpainting: is_inpaint,
            lora: true,
            controlnet: false, // Not implemented yet
            default_size: components.version.default_size(),
            supported_sizes: vec![
                (512, 512),
                (768, 768),
                (1024, 1024),
            ],
            default_steps,
            supports_guidance: true,
        };

        Self {
            components,
            capabilities,
        }
    }
}

impl ImageModel for StableDiffusionModel {
    fn model_type(&self) -> &str {
        "stable-diffusion"
    }

    fn model_variant(&self) -> &str {
        use crate::backend::models::SDVersion;
        match self.components.version {
            SDVersion::V1_5 | SDVersion::V1_5Inpaint => "v1-5",
            SDVersion::V2_1 | SDVersion::V2Inpaint => "v2-1",
            SDVersion::XL | SDVersion::XLInpaint => "xl",
            SDVersion::Turbo => "turbo",
            _ => "unknown",
        }
    }

    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    fn generate<F>(
        &mut self,
        request: &GenerationRequest,
        progress_callback: F,
    ) -> Result<DynamicImage>
    where
        F: FnMut(usize, usize, Option<DynamicImage>),
    {
        // Dispatch based on operation type
        match (request.input_image.as_ref(), request.mask.as_ref()) {
            (Some(img), Some(mask)) => {
                if !self.supports_inpainting() {
                    return Err(crate::error::Error::InvalidInput(
                        format!(
                            "Model {:?} does not support inpainting. Use an inpainting-specific model.",
                            self.components.version
                        )
                    ));
                }
                generation::inpaint(&self.components, request, img, mask, progress_callback)
            }
            (Some(img), None) => {
                generation::img2img(&self.components, request, img, progress_callback)
            }
            (None, _) => {
                generation::txt2img(&self.components, request, progress_callback)
            }
        }
    }
}
