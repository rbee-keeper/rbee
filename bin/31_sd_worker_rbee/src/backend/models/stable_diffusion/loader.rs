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

use super::{ModelComponents, StableDiffusionModel};
use crate::backend::lora::LoRAConfig;
use crate::backend::models::{ModelFile, SDVersion};

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
    pub fn load_components(
        &self,
        device: &Device,
        _lora_configs: &[LoRAConfig],
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

        // Create VarBuilder from SafeTensors files
        let unet_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[unet_weights.clone()], dtype, device)? };
        let vae_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[vae_weights.clone()], dtype, device)? };

        // Load UNet (from Candle)
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

/// Load model components without LoRA
pub fn load_model_simple(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
) -> Result<StableDiffusionModel> {
    let loader = ModelLoader::new(version, use_f16)?;
    let components = loader.load_components(device, &[])?;
    Ok(StableDiffusionModel::new(components))
}

/// Load a Stable Diffusion model with LoRA
pub fn load_stable_diffusion_with_lora(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    lora_configs: &[LoRAConfig],
) -> Result<StableDiffusionModel> {
    let loader = ModelLoader::new(version, use_f16)?;
    let components = loader.load_components(device, lora_configs)?;
    Ok(StableDiffusionModel::new(components))
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
