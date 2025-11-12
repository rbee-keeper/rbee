// TEAM-483: Mixtral MoE model loader
//
// Created by: TEAM-483
// Purpose: Load Mixtral-8x7B MoE from safetensors
// Reference: candle-transformers/src/models/mixtral.rs

use super::components::MixtralModel;
use crate::backend::models::helpers::{create_varbuilder, find_safetensors_files, load_config};
use crate::backend::models::ModelCapabilities;
use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::mixtral::{Config, Model as MixtralCandle};
use serde_json::Value;
use std::path::Path;

impl MixtralModel {
    /// Load Mixtral MoE model from safetensors directory
    ///
    /// TEAM-483: Loads Mixtral-8x7B and other Mixtral MoE models
    ///
    /// # Arguments
    /// * `path` - Path to model directory containing config.json and safetensors files
    /// * `device` - Device to load model on (CPU/CUDA)
    ///
    /// # Returns
    /// Loaded MixtralModel ready for inference
    ///
    /// # Errors
    /// Returns error if config is invalid or safetensors files are missing
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        tracing::info!("Loading Mixtral MoE model from {:?}", path);

        // TEAM-483: Use helper to find safetensors files
        let (parent, safetensor_files) = find_safetensors_files(path)?;

        // TEAM-483: Load config as JSON first to extract private fields
        let config_json: Value = load_config(&parent, "Mixtral")?;

        // Extract MoE-specific parameters from config JSON
        // TEAM-483: Config fields are pub(crate), so we extract from JSON like DeepSeek
        let num_experts = config_json["num_local_experts"]
            .as_u64()
            .context("Missing num_local_experts in config")? as usize;
        let experts_per_tok = config_json["num_experts_per_tok"]
            .as_u64()
            .context("Missing num_experts_per_tok in config")?
            as usize;

        tracing::info!(
            "Mixtral MoE config: {} experts, top-{} routing",
            num_experts,
            experts_per_tok
        );

        // Extract metadata from config JSON
        // TEAM-483: Config fields are pub(crate), extract from JSON
        let vocab_size =
            config_json["vocab_size"].as_u64().context("Missing vocab_size in config")? as usize;
        let max_position_embeddings = config_json["max_position_embeddings"]
            .as_u64()
            .context("Missing max_position_embeddings in config")?
            as usize;

        // Now deserialize to Mixtral Config
        let config: Config =
            serde_json::from_value(config_json).context("Failed to parse Mixtral config")?;

        // TEAM-483: Use helper to create VarBuilder
        let vb = create_varbuilder(&safetensor_files, device)?;

        // Create Mixtral model
        let model = MixtralCandle::new(&config, vb).context("Failed to create Mixtral model")?;

        // EOS token ID - Mixtral uses same as Mistral (2)
        let eos_token_id = 2;

        // TEAM-483: Mixtral doesn't have clear_kv_cache in candle's implementation
        // Set supports_cache_reset to false
        let capabilities = crate::backend::models::ModelCapabilities::standard(
            crate::backend::models::arch::MIXTRAL,
            32768,
            dtype.unwrap_or(candle_core::DType::F32),
        );

        tracing::info!(
            "Loaded Mixtral MoE: vocab_size={}, max_pos={}, experts={}, top_k={}",
            vocab_size,
            max_position_embeddings,
            num_experts,
            experts_per_tok
        );

        Ok(Self::new(
            model,
            eos_token_id,
            vocab_size,
            num_experts,
            experts_per_tok,
            device.clone(),
            capabilities,
        ))
    }
}
