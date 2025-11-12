// TEAM-482: DeepSeek model module
//
// Created by: TEAM-482
// Purpose: DeepSeek-R1 / DeepSeek-V2 safetensors support
// Reference: candle-transformers/src/models/deepseek2.rs

mod components;
mod loader;

pub use components::DeepSeekModel;

use anyhow::Result;
use candle_core::Tensor;

impl DeepSeekModel {
    /// Forward pass through the model
    ///
    /// TEAM-482: Delegates to candle's DeepSeek model
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
    /// TEAM-482: DeepSeek supports cache clearing
    pub fn reset_cache(&mut self) -> Result<()> {
        self.model.clear_kv_cache();
        Ok(())
    }
}

/// TEAM-482: Implement ModelTrait for DeepSeekModel
impl crate::backend::models::ModelTrait for DeepSeekModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::DEEPSEEK
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
