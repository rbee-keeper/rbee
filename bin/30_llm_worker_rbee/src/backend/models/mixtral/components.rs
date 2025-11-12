// TEAM-483: Mixtral MoE model components
//
// Created by: TEAM-483
// Purpose: Mixtral-8x7B MoE safetensors support
// Reference: candle-transformers/src/models/mixtral.rs

use candle_core::Device;
use candle_transformers::models::mixtral::Model as MixtralCandle;

/// Mixtral MoE model wrapper for safetensors files
///
/// TEAM-483: Wraps candle-transformers `mixtral` with safetensors support
/// Supports Mixtral-8x7B and other Mixtral MoE architectures
///
/// # MoE Architecture
/// - Multiple expert networks (typically 8)
/// - Top-k routing (typically top-2)
/// - Sparse activation (only selected experts process each token)
/// - Based on Mistral architecture with MoE layers
pub struct MixtralModel {
    pub(super) model: MixtralCandle,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) num_experts: usize,
    pub(super) experts_per_tok: usize,
    pub(super) device: Device,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl MixtralModel {
    /// Create a new MixtralModel instance
    ///
    /// TEAM-483: Constructor for internal use by loader
    pub(super) fn new(
        model: MixtralCandle,
        eos_token_id: u32,
        vocab_size: usize,
        num_experts: usize,
        experts_per_tok: usize,
        device: Device,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self {
            model,
            eos_token_id,
            vocab_size,
            num_experts,
            experts_per_tok,
            device,
            capabilities,
        }
    }
}
