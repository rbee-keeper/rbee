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
            supported_sizes: vec![(512, 512), (768, 768), (1024, 1024)],
            default_steps,
            supports_guidance: true,
        };

        Self { components, capabilities }
    }
}

/// TEAM-482: Implement ImageModel for StableDiffusionModel
///
/// Provides text-to-image, image-to-image, and inpainting capabilities
/// using the Stable Diffusion architecture (v1.5, v2.1, XL, Turbo).
///
/// Adopted patterns from LLM Worker:
/// - Sealed trait (API stability)
/// - Static lifetimes (zero-cost)
/// - Inline hints (performance)
/// - Architecture constants (type safety)
impl ImageModel for StableDiffusionModel {
    #[inline]
    fn model_type(&self) -> &'static str {
        super::arch::STABLE_DIFFUSION
    }

    #[inline]
    fn model_variant(&self) -> &'static str {
        use crate::backend::models::SDVersion;
        match self.components.version {
            SDVersion::V1_5 => super::arch::variants::SD_1_5,
            SDVersion::V1_5Inpaint => super::arch::variants::SD_1_5_INPAINT,
            SDVersion::V2_1 => super::arch::variants::SD_2_1,
            SDVersion::V2Inpaint => super::arch::variants::SD_2_INPAINT,
            SDVersion::XL => super::arch::variants::SD_XL,
            SDVersion::XLInpaint => super::arch::variants::SD_XL_INPAINT,
            SDVersion::Turbo => super::arch::variants::SD_TURBO,
            _ => "unknown", // FLUX models shouldn't reach here
        }
    }

    #[inline]
    fn capabilities(&self) -> &ModelCapabilities {
        &self.capabilities
    }

    fn generate(
        &mut self,
        request: &GenerationRequest,
        mut progress_callback: Box<dyn FnMut(usize, usize, Option<DynamicImage>) + Send>,
    ) -> Result<DynamicImage> {
        // TEAM-481: Unbox the callback and pass it to generation functions
        // Dispatch based on operation type
        match (request.input_image.as_ref(), request.mask.as_ref()) {
            (Some(img), Some(mask)) => {
                if !self.supports_inpainting() {
                    return Err(crate::error::Error::InvalidInput(format!(
                        "Model {:?} does not support inpainting. Use an inpainting-specific model.",
                        self.components.version
                    )));
                }
                generation::inpaint(&self.components, request, img, mask, |step, total, preview| {
                    progress_callback(step, total, preview)
                })
            }
            (Some(img), None) => {
                generation::img2img(&self.components, request, img, |step, total, preview| {
                    progress_callback(step, total, preview)
                })
            }
            (None, _) => generation::txt2img(&self.components, request, |step, total, preview| {
                progress_callback(step, total, preview)
            }),
        }
    }
}
