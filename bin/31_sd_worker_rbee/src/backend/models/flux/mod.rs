// TEAM-488: FLUX model module
//
// Self-contained FLUX implementation mirroring stable_diffusion structure
// TEAM-488: FLUX model
// Self-contained module for FLUX.1-dev and FLUX.1-schnell

mod components;
mod config;
mod loader;
pub mod generation;

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
            img2img: false,  // FLUX doesn't support img2img in Candle
            inpainting: false,  // FLUX doesn't support inpainting in Candle
            lora: false,  // FLUX LoRA not yet supported in Candle
            controlnet: false,  // FLUX ControlNet not yet supported in Candle
            default_size: (width, height),
            supported_sizes: vec![(1024, 1024), (768, 1024), (1024, 768)],  // FLUX typical sizes
            default_steps: components.version.default_steps(),
            supports_guidance: matches!(components.version, crate::backend::models::SDVersion::FluxDev),
        };
        
        Self {
            components,
            capabilities,
        }
    }
}

impl ImageModel for FluxModel {
    fn model_type(&self) -> &str {
        "flux"
    }

    fn model_variant(&self) -> &str {
        use crate::backend::models::SDVersion;
        match self.components.version {
            SDVersion::FluxDev => "flux-dev",
            SDVersion::FluxSchnell => "flux-schnell",
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
        // FLUX only supports txt2img
        if request.input_image.is_some() {
            return Err(crate::error::Error::InvalidInput(
                "FLUX models don't support img2img or inpainting (Candle limitation)".to_string(),
            ));
        }
        
        // Call FLUX txt2img generation
        txt2img(&mut self.components, request, progress_callback)
    }
}
