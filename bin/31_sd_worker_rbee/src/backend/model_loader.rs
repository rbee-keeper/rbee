// TEAM-390: Model loading from HuggingFace Hub
//
// Downloads and loads SD model components using hf-hub and SafeTensors.

use crate::error::{Error, Result};
use crate::narration::{log_model_loading_complete, log_model_loading_start};
use candle_core::Device;
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use std::time::Instant;

use super::models::{ModelComponents, ModelFile, SDVersion};

/// Model loader for downloading and caching models
pub struct ModelLoader {
    api: Api,
    version: SDVersion,
    use_f16: bool,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(version: SDVersion, use_f16: bool) -> Result<Self> {
        let api = Api::new().map_err(|e| Error::ModelLoading(format!("Failed to create HF API: {}", e)))?;
        Ok(Self {
            api,
            version,
            use_f16,
        })
    }

    /// Get path to a model file, downloading if necessary
    pub fn get_file(&self, file: ModelFile) -> Result<PathBuf> {
        let repo = match file {
            ModelFile::Tokenizer | ModelFile::Tokenizer2 => {
                ModelFile::tokenizer_repo(self.version)
            }
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
    pub fn load_components(&self, device: &Device) -> Result<ModelComponents> {
        let start = Instant::now();
        log_model_loading_start(&format!("{:?}", self.version));

        // Download all required files
        let _tokenizer_path = self.get_file(ModelFile::Tokenizer)?;
        let _clip_path = self.get_file(ModelFile::Clip)?;
        let _unet_path = self.get_file(ModelFile::Unet)?;
        let _vae_path = self.get_file(ModelFile::Vae)?;

        // For XL models, also download second tokenizer and CLIP
        if self.version.is_xl() {
            let _tokenizer2_path = self.get_file(ModelFile::Tokenizer2)?;
            let _clip2_path = self.get_file(ModelFile::Clip2)?;
        }

        // TODO: Actually load the models using candle-transformers
        // For now, just create placeholder
        let components = ModelComponents::new(self.version, device.clone(), self.use_f16);

        let elapsed = start.elapsed().as_millis() as u64;
        log_model_loading_complete(&format!("{:?}", self.version), elapsed);

        Ok(components)
    }
}

/// Load a Stable Diffusion model
pub fn load_model(
    version: SDVersion,
    device: &Device,
    use_f16: bool,
) -> Result<ModelComponents> {
    let loader = ModelLoader::new(version, use_f16)?;
    loader.load_components(device)
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
