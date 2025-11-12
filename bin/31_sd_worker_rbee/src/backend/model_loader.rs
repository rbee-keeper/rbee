// TEAM-390: Model loading from HuggingFace Hub
//
// Downloads and loads SD model components using hf-hub and SafeTensors.

use crate::error::{Error, Result};
use crate::narration::{log_model_loading_complete, log_model_loading_start};
use candle_core::Device;
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::time::Instant;

use super::lora::{create_lora_varbuilder, LoRAConfig, LoRAWeights};
use super::models::{flux_loader::FluxComponents, ModelComponents, ModelFile, SDVersion};

/// Model loader for downloading and caching models
pub struct ModelLoader {
    api: Api,
    version: SDVersion,
    use_f16: bool,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(version: SDVersion, use_f16: bool) -> Result<Self> {
        let api = Api::new()
            .map_err(|e| Error::ModelLoading(format!("Failed to create HF API: {}", e)))?;
        Ok(Self { api, version, use_f16 })
    }

    /// Get path to a model file, downloading if necessary
    pub fn get_file(&self, file: ModelFile) -> Result<PathBuf> {
        let repo = match file {
            ModelFile::Tokenizer | ModelFile::Tokenizer2 => ModelFile::tokenizer_repo(self.version),
            _ => self.version.repo(),
        };

        let path = file.path(self.version, self.use_f16);

        let repo = self
            .api
            .model(repo.to_string())
            .get(path)
            .map_err(|e| Error::ModelLoading(format!("Failed to download {}: {}", path, e)))?;

        Ok(repo)
    }

    /// Load all model components
    /// TEAM-399: Full implementation with actual model loading
    /// TEAM-487: Added LoRA support
    pub fn load_components(
        &self,
        device: &Device,
        lora_configs: &[LoRAConfig],
    ) -> Result<ModelComponents> {
        let start = Instant::now();
        log_model_loading_start(&format!("{:?}", self.version));

        // Download all required files
        let tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
        let clip_weights = self.get_file(ModelFile::Clip)?;
        let unet_weights = self.get_file(ModelFile::Unet)?;
        let vae_weights = self.get_file(ModelFile::Vae)?;

        // For XL models, also download second tokenizer and CLIP
        if self.version.is_xl() {
            let _tokenizer2_path = self.get_file(ModelFile::Tokenizer2)?;
            let _clip2_path = self.get_file(ModelFile::Clip2)?;
        }

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| Error::ModelLoading(format!("Failed to load tokenizer: {}", e)))?;

        // Get configs from version
        let clip_config = self.version.clip_config();
        let unet_config = self.version.unet_config();
        let vae_config = self.version.vae_config();

        // Determine dtype
        let dtype = if self.use_f16 { candle_core::DType::F16 } else { candle_core::DType::F32 };

        // Create base VarBuilder from SafeTensors files
        let base_unet_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[unet_weights.clone()], dtype, device)? };

        // TEAM-487: Apply LoRAs if any are configured
        let unet_vb = if !lora_configs.is_empty() {
            tracing::info!("Loading {} LoRAs for UNet", lora_configs.len());

            // Load all LoRA weights
            let mut loras = Vec::new();
            for config in lora_configs {
                tracing::info!("Loading LoRA: {} (strength: {})", config.path, config.strength);
                let lora_weights = LoRAWeights::load(&config.path, device)?;
                loras.push((lora_weights, config.strength));
            }

            // Create VarBuilder with LoRAs merged
            create_lora_varbuilder(base_unet_vb, loras)?
        } else {
            base_unet_vb
        };

        let vae_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[vae_weights.clone()], dtype, device)? };

        // Load UNet (from Candle) - now with LoRAs applied!
        let unet =
            candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel::new(
                unet_vb,
                4,     // in_channels
                4,     // out_channels
                false, // use_flash_attn (set to false for CPU, can be enabled for CUDA)
                unet_config,
            )?;

        // Load VAE (from Candle)
        let vae = candle_transformers::models::stable_diffusion::vae::AutoEncoderKL::new(
            vae_vb, 3, // in_channels (RGB)
            3, // out_channels (RGB)
            vae_config,
        )?;

        // Create scheduler (our implementation)
        // TEAM-404: Box the scheduler to use as trait object
        let scheduler = Box::new(crate::backend::scheduler::DDIMScheduler::new(
            1000, // num_train_timesteps
            self.version.default_steps(),
        ));

        let components = ModelComponents {
            version: self.version,
            device: device.clone(),
            dtype,
            tokenizer,
            clip_config,
            clip_weights,
            unet,
            vae,
            scheduler,
            vae_scale: 0.18215,
        };

        let elapsed = start.elapsed().as_millis() as u64;
        log_model_loading_complete(&format!("{:?}", self.version), elapsed);

        Ok(components)
    }
}

/// Loaded model (either Stable Diffusion or FLUX)
/// TEAM-483: Added FLUX support
pub enum LoadedModel {
    /// Stable Diffusion model (v1.5, v2.1, XL, Turbo, etc.)
    StableDiffusion(ModelComponents),
    /// FLUX model (dev or schnell)
    Flux(FluxComponents),
}

/// Load a model (Stable Diffusion or FLUX)
/// TEAM-483: Added FLUX support
/// TEAM-487: Added LoRA support (SD only)
pub fn load_model(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    lora_configs: &[LoRAConfig],
    quantized: bool,
) -> Result<LoadedModel> {
    if version.is_flux() {
        // Load FLUX model
        tracing::info!("Loading FLUX model: {:?}", version);
        
        // Get model path from HuggingFace cache
        let api = Api::new()
            .map_err(|e| Error::ModelLoading(format!("Failed to create HF API: {}", e)))?;
        let repo = api.model(version.repo().to_string());
        
        // Download a file to get the cache directory
        let model_file = if quantized {
            match version {
                SDVersion::FluxSchnell => "flux1-schnell.gguf",
                SDVersion::FluxDev => "flux1-dev.gguf",
                _ => unreachable!(),
            }
        } else {
            match version {
                SDVersion::FluxSchnell => "flux1-schnell.safetensors",
                SDVersion::FluxDev => "flux1-dev.safetensors",
                _ => unreachable!(),
            }
        };
        
        let _file_path = repo
            .get(model_file)
            .map_err(|e| Error::ModelLoading(format!("Failed to download {}: {}", model_file, e)))?;
        
        // Get the cache directory (parent of the file)
        let cache_dir = repo
            .get("ae.safetensors")
            .map_err(|e| Error::ModelLoading(format!("Failed to download VAE: {}", e)))?
            .parent()
            .ok_or_else(|| Error::ModelLoading("Failed to get cache directory".to_string()))?
            .to_path_buf();
        
        let components = FluxComponents::load(
            cache_dir
                .to_str()
                .ok_or_else(|| Error::ModelLoading("Invalid cache path".to_string()))?,
            version,
            device,
            use_f16,
            quantized,
        )?;
        
        Ok(LoadedModel::Flux(components))
    } else {
        // Load Stable Diffusion model
        tracing::info!("Loading Stable Diffusion model: {:?}", version);
        let loader = ModelLoader::new(version, use_f16)?;
        let components = loader.load_components(device, lora_configs)?;
        Ok(LoadedModel::StableDiffusion(components))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_model_loader_creation() {
        let loader = ModelLoader::new(SDVersion::V1_5, false);
        assert!(loader.is_ok());
    }
}
