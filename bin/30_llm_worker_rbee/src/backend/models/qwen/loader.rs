// TEAM-482: Created during loader/component separation
//! Qwen model loader
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::Config;
use std::path::Path;

use super::QwenModel;

impl QwenModel {
    /// Load Qwen model from `SafeTensors`
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
        .context("Failed to parse Qwen config.json")?;

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = candle_transformers::models::qwen2::ModelForCausalLM::new(&config, vb)
            .context("Failed to load Qwen model")?;

        let vocab_size = config.vocab_size;

        tracing::info!(
            architecture = "qwen",
            hidden_size = config.hidden_size,
            num_layers = config.num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Qwen model"
        );

        // TEAM-482: Qwen capabilities (cache reset not yet implemented)
        let capabilities = crate::backend::models::ModelCapabilities {
            uses_position: true,
            supports_cache_reset: false, // Not yet implemented
            max_context_length: 32768,
            supports_streaming: true,
            architecture_family: crate::backend::models::arch::QWEN,
            is_quantized: false,
        };

        Ok(Self::new(model, vocab_size, capabilities))
    }
}
