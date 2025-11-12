// TEAM-482: DeepSeek model loader
//
// Created by: TEAM-482
// Purpose: Load DeepSeek models from safetensors files
// Reference: candle-transformers/src/models/deepseek2.rs

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::deepseek2::{DeepSeekV2, DeepSeekV2Config};
use serde_json::Value;
use std::path::Path;

use super::DeepSeekModel;
use crate::backend::models::helpers::{create_varbuilder, find_safetensors_files, load_config};

impl DeepSeekModel {
    /// Load DeepSeek model from safetensors
    ///
    /// TEAM-482: Uses helper functions for clean, DRY implementation
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        tracing::info!(path = ?path, "Loading DeepSeek model");

        // TEAM-482: Use helper to find safetensors files
        let (parent, safetensor_files) = find_safetensors_files(path)?;

        // TEAM-482: Load config as JSON first to extract private fields
        let config_json: Value = load_config(&parent, "DeepSeek")?;

        // Extract metadata from JSON (fields are private in DeepSeekV2Config)
        let vocab_size = config_json["vocab_size"]
            .as_u64()
            .context("Missing vocab_size in config.json")? as usize;
        let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(2) as u32; // Default to 2 if not specified
        let max_position_embeddings =
            config_json["max_position_embeddings"].as_u64().unwrap_or(2048) as usize;

        // Now deserialize to DeepSeekV2Config
        let config: DeepSeekV2Config =
            serde_json::from_value(config_json).context("Failed to parse DeepSeek config")?;

        // TEAM-482: Use helper to create VarBuilder
        let vb = create_varbuilder(&safetensor_files, device)?;

        // Build model
        let model = DeepSeekV2::new(&config, vb).context("Failed to load DeepSeek model")?;

        tracing::info!(architecture = "deepseek", vocab_size = vocab_size, "Loaded DeepSeek model");

        // TEAM-482: DeepSeek capabilities
        let capabilities = crate::backend::models::ModelCapabilities::standard(crate::backend::models::arch::DEEPSEEK, max_position_embeddings, dtype.unwrap_or(candle_core::DType::F32));

        Ok(Self::new(model, eos_token_id, vocab_size, device.clone(), capabilities))
    }
}
