// TEAM-482: Quantized DeepSeek model module
//
// Created by: TEAM-482
// Purpose: DeepSeek-R1 / DeepSeek-V2 GGUF support

mod components;
mod loader;

pub use components::QuantizedDeepSeekModel;

use anyhow::Result;
use candle_core::Tensor;

impl QuantizedDeepSeekModel {
    /// Forward pass through the model
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
    pub fn reset_cache(&mut self) -> Result<()> {
        tracing::debug!("Quantized DeepSeek model cache will reset on next position=0 forward pass");
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for QuantizedDeepSeekModel
impl crate::backend::models::ModelTrait for QuantizedDeepSeekModel {
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
        crate::backend::models::arch::DEEPSEEK_QUANTIZED
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
