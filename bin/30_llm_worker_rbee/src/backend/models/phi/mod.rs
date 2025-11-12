// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Phi model implementation
// TEAM-482: Refactored into components/loader pattern

//! Phi model wrapper
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (removed trait, using enum pattern)
//! Refactored by: TEAM-482 (split into components/loader)

mod components;
mod loader;

pub use components::PhiModel;

use anyhow::{Context, Result};
use candle_core::Tensor;

impl PhiModel {
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
impl crate::backend::models::ModelTrait for PhiModel {
    #[inline]
    fn forward(&mut self, input_ids: &Tensor, _position: usize) -> Result<Tensor> {
        // Phi doesn't use position - it manages cache internally
        self.forward(input_ids)
    }

    #[inline]
    fn eos_token_id(&self) -> u32 {
        self.eos_token_id()
    }

    fn eos_tokens(&self) -> crate::backend::traits::EosTokens {
        // TEAM-485: Single EOS token (default for most models)
        crate::backend::traits::EosTokens::single(self.eos_token_id())
    }

    #[inline]
    fn architecture(&self) -> &'static str {
        crate::backend::models::arch::PHI
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
    fn capabilities(&self) -> &crate::backend::models::ModelCapabilities {
        &self.capabilities
    }
}
