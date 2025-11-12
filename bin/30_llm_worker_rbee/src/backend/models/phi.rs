// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Phi model implementation

//! Phi model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::phi::{Config, Model};
use std::path::Path;

/// Phi model wrapper
///
/// TEAM-017: Wraps candle-transformers Phi with its natural interface
/// TEAM-482: Added capabilities (Phi has special requirements)
pub struct PhiModel {
    model: Model,
    vocab_size: usize,
    capabilities: super::ModelCapabilities,
}

impl PhiModel {
    /// Load Phi model from `SafeTensors`
    ///
    /// TEAM-017: Candle-idiomatic pattern
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let (parent, safetensor_files) = super::find_safetensors_files(path)?;

        // Parse config.json
        let config_path = parent.join("config.json");
        let config: Config = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
        )
        .context("Failed to parse Phi config.json")?;

        // Create VarBuilder and load model
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, device)? };
        let model = Model::new(&config, vb).context("Failed to load Phi model")?;

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
        let capabilities = super::ModelCapabilities {
            uses_position: false,  // Phi doesn't use position parameter
            supports_cache_reset: false,  // Phi manages cache internally
            max_context_length: 2048,  // Phi default context
            supports_streaming: true,
            architecture_family: super::arch::PHI,
            is_quantized: false,
        };

        Ok(Self { model, vocab_size, capabilities })
    }

    /// Forward pass using Phi's natural interface
    ///
    /// TEAM-017: Phi doesn't use position parameter, manages cache internally
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids).context("Phi forward pass failed")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        // Phi typically uses token ID 50256 for EOS (GPT-2 tokenizer)
        50256
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// TEAM-482: Implement ModelTrait for PhiModel
///
/// Note: Phi's forward pass doesn't use position parameter, so we ignore it.
/// This demonstrates how the trait pattern handles model-specific differences.
impl super::ModelTrait for PhiModel {
    #[inline]
    fn forward(&mut self, input_ids: &Tensor, _position: usize) -> Result<Tensor> {
        // Phi doesn't use position - it manages cache internally
        self.forward(input_ids)
    }

    #[inline]
    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        super::arch::PHI
    }

    #[inline]
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    #[inline]
    fn reset_cache(&mut self) -> Result<()> {
        // Phi manages cache internally, no reset needed
        tracing::debug!("Phi manages cache internally, no reset needed");
        Ok(())
    }
    
    #[inline]
    fn capabilities(&self) -> &super::ModelCapabilities {
        &self.capabilities
    }
}
