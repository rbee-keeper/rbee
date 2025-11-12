// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Qwen model implementation
// TEAM-482: Refactored into components/loader pattern

//! Qwen model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)
//! Refactored by: TEAM-482 (split into components/loader)

mod components;
mod loader;

pub use components::QwenModel;

use anyhow::{Context, Result};
use candle_core::Tensor;

impl QwenModel {
    /// Forward pass using Qwen's natural interface
    ///
    /// TEAM-017: Uses position parameter
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).context("Qwen forward pass failed")
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        // Qwen typically uses token ID 151643 for EOS
        151643
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

/// TEAM-482: Implement ModelTrait for QwenModel
impl crate::backend::models::ModelTrait for QwenModel {
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.forward(input_ids, position)
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::QWEN
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn reset_cache(&mut self) -> Result<()> {
        // TEAM-482: Qwen cache reset not yet implemented
        // Return error instead of silent failure
        anyhow::bail!("Cache reset not yet implemented for Qwen")
    }
    
    #[inline]
    fn capabilities(&self) -> &crate::backend::models::ModelCapabilities {
        &self.capabilities
    }
}
