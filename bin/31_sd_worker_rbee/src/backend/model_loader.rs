// TEAM-488: Unified model loader for SD and FLUX
//
// Loads models based on SDVersion and returns trait objects

use crate::backend::lora::LoRAConfig;
use crate::backend::models::{flux, stable_diffusion, SDVersion};
use crate::backend::traits::ImageModel;
use crate::error::Result;
use candle_core::Device;

/// Model enum that can hold either SD or FLUX models
///
/// TEAM-488: Concrete enum instead of trait object (avoids object safety issues)
pub enum LoadedModel {
    StableDiffusion(stable_diffusion::StableDiffusionModel),
    Flux(flux::FluxModel),
}

impl ImageModel for LoadedModel {
    fn model_type(&self) -> &str {
        match self {
            Self::StableDiffusion(m) => m.model_type(),
            Self::Flux(m) => m.model_type(),
        }
    }

    fn model_variant(&self) -> &str {
        match self {
            Self::StableDiffusion(m) => m.model_variant(),
            Self::Flux(m) => m.model_variant(),
        }
    }

    fn capabilities(&self) -> &crate::backend::traits::ModelCapabilities {
        match self {
            Self::StableDiffusion(m) => m.capabilities(),
            Self::Flux(m) => m.capabilities(),
        }
    }

    fn generate<F>(
        &mut self,
        request: &crate::backend::traits::GenerationRequest,
        progress_callback: F,
    ) -> Result<image::DynamicImage>
    where
        F: FnMut(usize, usize, Option<image::DynamicImage>),
    {
        match self {
            Self::StableDiffusion(m) => m.generate(request, progress_callback),
            Self::Flux(m) => m.generate(request, progress_callback),
        }
    }
}

/// Load a model (SD or FLUX) based on version
///
/// Returns a LoadedModel enum that implements ImageModel.
/// The model stays loaded in memory as long as the daemon runs.
///
/// # Arguments
/// * `version` - Model version (V1_5, XL, FluxDev, etc.)
/// * `device` - Compute device (CPU, CUDA, Metal)
/// * `use_f16` - Use FP16 precision (recommended for GPU)
/// * `loras` - LoRA configurations (SD only, ignored for FLUX)
/// * `quantized` - Use quantized models (FLUX only, ignored for SD)
///
/// # Returns
/// LoadedModel enum (StableDiffusion or Flux variant)
pub fn load_model(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    loras: &[LoRAConfig],
    quantized: bool,
) -> Result<LoadedModel> {
    tracing::info!("Loading model: {:?} (f16={}, quantized={})", version, use_f16, quantized);
    
    if version.is_flux() {
        // Load FLUX model
        tracing::info!("Loading FLUX model from HuggingFace...");
        let repo = version.repo();
        let components = flux::load_model(repo, version, device, use_f16, quantized)?;
        let model = flux::FluxModel::new(components);
        
        tracing::info!("✅ FLUX model loaded: {}", model.model_variant());
        Ok(LoadedModel::Flux(model))
    } else {
        // Load Stable Diffusion model
        tracing::info!("Loading Stable Diffusion model from HuggingFace...");
        let model = if loras.is_empty() {
            stable_diffusion::load_model_simple(version, device, use_f16)?
        } else {
            stable_diffusion::load_stable_diffusion_with_lora(version, device, use_f16, loras)?
        };
        
        tracing::info!("✅ Stable Diffusion model loaded: {}", model.model_variant());
        Ok(LoadedModel::StableDiffusion(model))
    }
}
