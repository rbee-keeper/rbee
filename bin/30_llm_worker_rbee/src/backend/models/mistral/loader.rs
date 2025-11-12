// TEAM-482: Created during loader/component separation
//! Mistral model loader
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::Config;
use std::path::Path;

use super::MistralModel;

impl MistralModel {
    /// Load Mistral model from `SafeTensors`
    ///
    /// TEAM-017: Candle-idiomatic pattern
    /// TEAM-482: Moved to separate loader module
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (parent, safetensor_files) = crate::backend::models::find_safetensors_files(path)?;

        // Parse config.json
        let config_path = parent.join("config.json");
        let config: Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
        )
        .context("Failed to parse Mistral config.json")?;

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = candle_transformers::models::mistral::Model::new(&config, vb)
            .context("Failed to load Mistral model")?;

        let vocab_size = config.vocab_size;

        tracing::info!(
            architecture = "mistral",
            hidden_size = config.hidden_size,
            num_layers = config.num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Mistral model"
        );

        // TEAM-482: Mistral capabilities (cache reset not yet implemented)
        let capabilities = crate::backend::models::ModelCapabilities {
            uses_position: true,
            supports_cache_reset: false, // Not yet implemented
            max_context_length: 32768,
            supports_streaming: true,
            architecture_family: crate::backend::models::arch::MISTRAL,
            is_quantized: false,
        };

        Ok(Self::new(model, vocab_size, capabilities))
    }
}
