// TEAM-482: Gemma model loader
//
// Created by: TEAM-482
// Purpose: Load Gemma models from safetensors files
// Reference: candle-transformers/src/models/gemma.rs

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::gemma::{Config, Model};
use std::path::Path;

use super::GemmaModel;
use crate::backend::models::helpers::{create_varbuilder, find_safetensors_files, load_config};

impl GemmaModel {
    /// Load Gemma model from safetensors
    ///
    /// TEAM-482: Uses helper functions for clean, DRY implementation
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        tracing::info!(path = ?path, "Loading Gemma model");

        // TEAM-482: Use helper to find safetensors files
        let (parent, safetensor_files) = find_safetensors_files(path)?;

        // TEAM-482: Use helper to load config
        let config: Config = load_config(&parent, "Gemma")?;

        // TEAM-482: Use helper to create VarBuilder
        let vb = create_varbuilder(&safetensor_files, device)?;

        // TEAM-487: Enable flash attention when available (CUDA builds)
        let use_flash_attn = cfg!(feature = "flash-attn");
        let model = Model::new(use_flash_attn, &config, vb).context("Failed to load Gemma model")?;

        // Extract metadata from config
        let vocab_size = config.vocab_size;
        let eos_token_id = 1; // TEAM-482: Gemma uses EOS token ID 1 (different from Llama's 2)

        tracing::info!(
            architecture = "gemma",
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            use_flash_attn = use_flash_attn,  // TEAM-487: Log flash attention status
            "Loaded Gemma model"
        );

        // TEAM-482: Gemma capabilities
        let capabilities = crate::backend::models::ModelCapabilities::standard(crate::backend::models::arch::GEMMA, config.max_position_embeddings, dtype.unwrap_or(candle_core::DType::F32));

        Ok(Self::new(model, eos_token_id, vocab_size, device.clone(), capabilities))
    }
}
