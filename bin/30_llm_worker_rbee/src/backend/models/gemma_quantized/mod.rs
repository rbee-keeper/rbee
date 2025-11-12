// TEAM-409: Created 2025-11-05 - Quantized Gemma GGUF support
// TEAM-482: Refactored into components/loader pattern

//\! Quantized Gemma model wrapper for GGUF files
//\!
//\! Created by: TEAM-409
//\! Refactored by: TEAM-482 (split into components/loader)
//\! Purpose: Load and run GGUF quantized Gemma models (`Q4_K_M`, `Q5_K_M`, etc.)
//\!
//\! Based on candle-transformers quantized_gemma3 implementation

mod components;
mod loader;

pub use components::QuantizedGemmaModel;

use anyhow::Result;
use candle_core::Tensor;

impl QuantizedGemmaModel {
    /// Forward pass through the model
    ///
    /// TEAM-409: Delegates to candle's quantized model
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Reset KV cache
    ///
    /// TEAM-409: Gemma doesn't expose clear_kv_cache in the public API
    /// The model manages its own cache internally
    pub fn reset_cache(&mut self) -> Result<()> {
        tracing::warn!("Cache reset not implemented for Gemma (managed internally)");
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for QuantizedGemmaModel
impl crate::backend::models::ModelTrait for QuantizedGemmaModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::GEMMA_QUANTIZED
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn reset_cache(&mut self) -> Result<()> {
        self.reset_cache()
    }

    #[inline]
    fn capabilities(&self) -> &crate::backend::models::ModelCapabilities {
        &self.capabilities
    }
}
