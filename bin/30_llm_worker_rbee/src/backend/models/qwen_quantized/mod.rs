// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Quantized Qwen GGUF support
// TEAM-482: Refactored into components/loader pattern

//\! Quantized Qwen model wrapper for GGUF files
//\!
//\! Created by: TEAM-090
//\! Refactored by: TEAM-482 (split into components/loader)
//\! Purpose: Load and run GGUF quantized Qwen models

mod components;
mod loader;

pub use components::QuantizedQwenModel;

use anyhow::Result;
use candle_core::Tensor;

impl QuantizedQwenModel {
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn reset_cache(&mut self) -> Result<()> {
        tracing::debug!("Quantized Qwen model cache will reset on next position=0 forward pass");
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for QuantizedQwenModel
impl crate::backend::models::ModelTrait for QuantizedQwenModel {
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
        crate::backend::models::arch::QWEN_QUANTIZED
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
