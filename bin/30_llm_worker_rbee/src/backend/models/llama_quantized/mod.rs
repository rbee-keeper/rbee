// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Quantized Llama GGUF support
// TEAM-482: Refactored into components/loader pattern

//! Quantized Llama model wrapper for GGUF files
//!
//! Created by: TEAM-036
//! Modified by: TEAM-088 (added comprehensive narration for debugging)
//! Refactored by: TEAM-482 (split into components/loader)
//! Purpose: Load and run GGUF quantized models (`Q4_K_M`, `Q5_K_M`, etc.)

mod components;
mod loader;

pub use components::QuantizedLlamaModel;

use anyhow::Result;
use candle_core::Tensor;

impl QuantizedLlamaModel {
    /// Forward pass through the model
    ///
    /// TEAM-036: Delegates to candle's quantized model
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

    /// Reset KV cache to clear history
    ///
    /// TEAM-036: Quantized models manage cache internally per layer
    /// Cache is automatically cleared on position=0, so no explicit reset needed
    pub fn reset_cache(&mut self) -> Result<()> {
        // Quantized models in candle reset cache automatically when position=0
        // The kv_cache in each layer is set to None when index_pos == 0
        tracing::debug!("Quantized model cache will reset on next position=0 forward pass");
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for QuantizedLlamaModel
impl crate::backend::models::ModelTrait for QuantizedLlamaModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    fn eos_tokens(&self) -> crate::backend::traits::EosTokens {
        // TEAM-485: Single EOS token (default for most models)
        crate::backend::traits::EosTokens::single(self.eos_token_id())
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::LLAMA_QUANTIZED
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
