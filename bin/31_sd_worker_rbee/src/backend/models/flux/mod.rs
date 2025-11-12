// TEAM-488: FLUX model module
//
// Self-contained FLUX implementation mirroring stable_diffusion structure
// TEAM-488: FLUX model
// Self-contained module for FLUX.1-dev and FLUX.1-schnell

mod components;
mod config;
pub mod generation;
mod loader;

pub use components::ModelComponents;
pub use config::FluxConfig;
pub use loader::load_model;

// Re-export generation functions
pub use generation::txt2img;

use crate::backend::traits::{GenerationRequest, ImageModel, ModelCapabilities};
use crate::error::Result;
use image::DynamicImage;

/// FLUX model with ImageModel trait implementation
/// TEAM-488: Wraps ModelComponents and implements the unified trait
pub struct FluxModel {
    components: ModelComponents,
    capabilities: ModelCapabilities,
}

impl FluxModel {
    /// Create new FLUX model from loaded components
    pub fn new(components: ModelComponents) -> Self {
        let (width, height) = components.version.default_size();

        let capabilities = ModelCapabilities {
            img2img: false,    // FLUX doesn't support img2img in Candle
            inpainting: false, // FLUX doesn't support inpainting in Candle
            lora: false,       // FLUX LoRA not yet supported in Candle
            controlnet: false, // FLUX ControlNet not yet supported in Candle
            default_size: (width, height),
            supported_sizes: vec![(1024, 1024), (768, 1024), (1024, 768)], // FLUX typical sizes
            default_steps: components.version.default_steps(),
            supports_guidance: matches!(
                components.version,
                crate::backend::models::SDVersion::FluxDev
            ),
        };

        Self { components, capabilities }
    }
}

/// TEAM-482: Implement ImageModel for FluxModel
///
/// Provides text-to-image capabilities using the FLUX architecture.
/// Note: FLUX models don't support img2img or inpainting due to Candle limitations.
///
/// Adopted patterns from LLM Worker:
/// - Sealed trait (API stability)
/// - Static lifetimes (zero-cost)
/// - Inline hints (performance)
/// - Architecture constants (type safety)
impl ImageModel for FluxModel {
    #[inline]
    fn model_type(&self) -> &'static str {
        super::arch::FLUX
    }

    #[inline]
    fn model_variant(&self) -> &'static str {
        use crate::backend::models::SDVersion;
        match self.components.version {
            SDVersion::FluxDev => super::arch::variants::FLUX_DEV,
            SDVersion::FluxSchnell => super::arch::variants::FLUX_SCHNELL,
            _ => "unknown", // SD models shouldn't reach here
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
        // TEAM-481: Unbox the callback and pass it to generation function
        // FLUX only supports txt2img
        if request.input_image.is_some() {
            return Err(crate::error::Error::InvalidInput(
                "FLUX models don't support img2img or inpainting (Candle limitation)".to_string(),
            ));
        }

        // Call FLUX txt2img generation
        txt2img(&mut self.components, request, |step, total, preview| {
            progress_callback(step, total, preview)
        })
    }
}
