// TEAM-482: Created during loader/component separation
//! Phi model loader
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::Config;
use std::path::Path;

use super::PhiModel;

impl PhiModel {
    /// Load Phi model from `SafeTensors`
    ///
    /// TEAM-017: Candle-idiomatic pattern
    /// TEAM-482: Moved to separate loader module
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        let (parent, safetensor_files) = crate::backend::models::find_safetensors_files(path)?;

        // Parse config.json
        let config_path = parent.join("config.json");
        let config: Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
        )
        .context("Failed to parse Phi config.json")?;

        // Create VarBuilder and load model
        let dtype = dtype.unwrap_or(DType::F32);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = candle_transformers::models::phi::Model::new(&config, vb)
            .context("Failed to load Phi model")?;

        // TEAM-017: Extract vocab_size from JSON since fields are private
        let config_json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to reopen config.json at {config_path:?}"))?,
        )?;

        let vocab_size = config_json["vocab_size"].as_u64().context("missing vocab_size")? as usize;
        let hidden_size =
            config_json["hidden_size"].as_u64().context("missing hidden_size")? as usize;
        let num_hidden_layers = config_json["num_hidden_layers"]
            .as_u64()
            .context("missing num_hidden_layers")? as usize;

        tracing::info!(
            architecture = "phi",
            hidden_size = hidden_size,
            num_layers = num_hidden_layers,
            vocab_size = vocab_size,
            "Loaded Phi model"
        );

        // TEAM-482: Phi has special capabilities - doesn't use position, manages cache internally
        let capabilities = crate::backend::models::ModelCapabilities::standard(crate::backend::models::arch::PHI, 2048, dtype);

        Ok(Self::new(model, vocab_size, capabilities))
    }
}
