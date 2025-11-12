// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Llama model implementation
// TEAM-482: Refactored into components and loader modules

mod components;
mod loader;

pub use components::LlamaModel;

use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use candle_transformers::models::llama::{Cache, LlamaEosToks};

impl LlamaModel {
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        tracing::info!(
            position = position,
            input_shape = ?input_ids.dims(),
            input_device = ?input_ids.device(),
            input_dtype = ?input_ids.dtype(),
            "Llama forward pass starting"
        );

        match self.model.forward(input_ids, position, &mut self.cache) {
            Ok(logits) => {
                tracing::info!(
                    output_shape = ?logits.dims(),
                    output_device = ?logits.device(),
                    output_dtype = ?logits.dtype(),
                    "Llama forward pass completed successfully"
                );
                Ok(logits)
            }
            Err(e) => {
                tracing::error!(
                    error = %e,
                    error_debug = ?e,
                    position = position,
                    input_shape = ?input_ids.dims(),
                    input_device = ?input_ids.device(),
                    "Llama forward pass failed - Candle error details"
                );

                if format!("{e:?}").contains("shape") {
                    tracing::error!("Shape mismatch detected in forward pass");
                } else if format!("{e:?}").contains("device") {
                    tracing::error!("Device mismatch detected in forward pass");
                } else if format!("{e:?}").contains("dtype") {
                    tracing::error!("DType mismatch detected in forward pass");
                }

                Err(e).context("Llama forward pass failed")
            }
        }
    }

    pub fn eos_token_id(&self) -> u32 {
        match self.config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => id,
            Some(LlamaEosToks::Multiple(ref ids)) => ids.first().copied().unwrap_or(2),
            None => 2,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        let dtype = DType::F32;
        self.cache = Cache::new(true, dtype, &self.config, &self.device)
            .context("Failed to recreate cache")?;
        tracing::debug!("Cache reset complete - ready for new request");
        Ok(())
    }
}

impl crate::backend::models::ModelTrait for LlamaModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    fn eos_tokens(&self) -> crate::backend::traits::EosTokens {
        // TEAM-486: Llama models support multiple EOS tokens (Llama 3)
        match &self.config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => crate::backend::traits::EosTokens::single(*id),
            Some(LlamaEosToks::Multiple(ids)) => crate::backend::traits::EosTokens::multiple(ids.clone()),
            None => crate::backend::traits::EosTokens::single(2), // Fallback
        }
    }

    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::LLAMA
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn reset_cache(&mut self) -> Result<()> {
        self.reset_cache()
    }

    fn capabilities(&self) -> &crate::backend::models::ModelCapabilities {
        &self.capabilities
    }
}
