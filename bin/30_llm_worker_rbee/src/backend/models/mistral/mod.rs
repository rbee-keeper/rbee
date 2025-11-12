// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Mistral model implementation

//! Mistral model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::mistral::{Config, Model};
use std::path::Path;

/// Mistral model wrapper
///
/// TEAM-017: Wraps candle-transformers Mistral with its natural interface
/// TEAM-482: Added capabilities
#[derive(Debug)]
pub struct MistralModel {
    model: Model,
    vocab_size: usize,
    capabilities: crate::backend::models::ModelCapabilities,
}

impl MistralModel {
    /// Load Mistral model from `SafeTensors`
    ///
    /// TEAM-017: Candle-idiomatic pattern
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
        let model = Model::new(&config, vb).context("Failed to load Mistral model")?;

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
            supports_cache_reset: false,  // Not yet implemented
            max_context_length: 32768,
            supports_streaming: true,
            architecture_family: crate::backend::models::arch::MISTRAL,
            is_quantized: false,
        };

        Ok(Self { model, vocab_size, capabilities })
    }

    /// Forward pass using Mistral's natural interface
    ///
    /// TEAM-017: Uses position parameter
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).context("Mistral forward pass failed")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        // Mistral typically uses token ID 2 for EOS
        2
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// TEAM-482: Implement ModelTrait for MistralModel
impl crate::backend::models::ModelTrait for MistralModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::MISTRAL
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn reset_cache(&mut self) -> Result<()> {
        // TEAM-482: Mistral cache reset not yet implemented
        // Return error instead of silent failure
        anyhow::bail!("Cache reset not yet implemented for Mistral")
    }
    
    #[inline]
    fn capabilities(&self) -> &crate::backend::models::ModelCapabilities {
        &self.capabilities
    }
}
