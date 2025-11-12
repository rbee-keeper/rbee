// TEAM-482: Gemma model module
//
// Created by: TEAM-482
// Purpose: Gemma safetensors support (completes existing GGUF support)
// Reference: candle-transformers/src/models/gemma.rs

mod components;
mod loader;

pub use components::GemmaModel;

use anyhow::Result;
use candle_core::Tensor;

impl GemmaModel {
    /// Forward pass through the model
    ///
    /// TEAM-482: Delegates to candle's Gemma model
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get EOS token ID
    ///
    /// TEAM-482: Gemma uses EOS token ID 1 (different from Llama's 2)
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Reset KV cache
    ///
    /// TEAM-482: Gemma supports cache clearing
    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for GemmaModel
impl crate::backend::models::ModelTrait for GemmaModel {
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
        crate::backend::models::arch::GEMMA
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
