// TEAM-483: Mixtral MoE model module
//
// Created by: TEAM-483
// Purpose: Mixtral-8x7B MoE safetensors support
// Reference: candle-transformers/src/models/mixtral.rs
//
// # Architecture
// Mixtral is a sparse Mixture of Experts (MoE) model based on Mistral architecture:
// - Multiple expert networks (typically 8 experts)
// - Top-k routing (typically top-2 experts per token)
// - Sparse activation (only selected experts process each token)
// - Sliding window attention
// - RoPE embeddings
//
// # Key Differences from Mistral
// - Uses SparseMoeBlock instead of standard MLP
// - Router network selects which experts to activate
// - More parameters but same computational cost (sparse activation)

mod components;
mod loader;

pub use components::MixtralModel;

use anyhow::Result;
use candle_core::Tensor;

impl MixtralModel {
    /// Forward pass through the model
    ///
    /// TEAM-483: Delegates to candle's Mixtral model
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model
            .forward(input_ids, position)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of experts in MoE
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get number of experts activated per token
    pub fn experts_per_tok(&self) -> usize {
        self.experts_per_tok
    }

    /// Reset KV cache
    ///
    /// TEAM-483: Mixtral doesn't support cache clearing in candle's implementation
    /// Returns error to indicate this is not supported
    pub fn reset_cache(&mut self) -> Result<()> {
        anyhow::bail!("Mixtral does not support cache reset")
    }
}

/// TEAM-483: Implement ModelTrait for MixtralModel
impl crate::backend::models::ModelTrait for MixtralModel {
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
        crate::backend::models::arch::MIXTRAL
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
