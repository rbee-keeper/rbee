// TEAM-488: Unified model loader for SD and FLUX
// TEAM-481: Now returns Box<dyn ImageModel> for true object safety

use crate::backend::lora::LoRAConfig;
use crate::backend::models::{flux, stable_diffusion, SDVersion};
use crate::backend::traits::ImageModel;
use crate::error::Result;
use candle_core::Device;

/// Load a model (SD or FLUX) based on version
///
/// TEAM-481: Returns Box<dyn ImageModel> for true polymorphism and object safety.
/// The model stays loaded in memory as long as the daemon runs.
///
/// # Arguments
/// * `version` - Model version (`V1_5`, XL, `FluxDev`, etc.)
/// * `device` - Compute device (CPU, CUDA, Metal)
/// * `use_f16` - Use FP16 precision (recommended for GPU)
/// * `loras` - `LoRA` configurations (SD only, ignored for FLUX)
/// * `quantized` - Use quantized models (FLUX only, ignored for SD)
///
/// # Returns
/// Box<dyn ImageModel> - Boxed trait object (either `StableDiffusion` or Flux)
pub fn load_model(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    loras: &[LoRAConfig],
    quantized: bool,
) -> Result<Box<dyn ImageModel>> {
    tracing::info!("Loading model: {:?} (f16={}, quantized={})", version, use_f16, quantized);

    if version.is_flux() {
        // Load FLUX model
        tracing::info!("Loading FLUX model from HuggingFace...");
        let repo = version.repo();
        let components = flux::load_model(repo, version, device, use_f16, quantized)?;
        let model = flux::FluxModel::new(components);

        tracing::info!("✅ FLUX model loaded: {}", model.model_variant());
        Ok(Box::new(model))
    } else {
        // Load Stable Diffusion model
        tracing::info!("Loading Stable Diffusion model from HuggingFace...");
        let model = if loras.is_empty() {
            stable_diffusion::load_model_simple(version, device, use_f16)?
        } else {
            stable_diffusion::load_stable_diffusion_with_lora(version, device, use_f16, loras)?
        };

        tracing::info!("✅ Stable Diffusion model loaded: {}", model.model_variant());
        Ok(Box::new(model))
    }
}
