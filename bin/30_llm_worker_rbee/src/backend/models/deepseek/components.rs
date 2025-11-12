// TEAM-482: DeepSeek model components
//
// Created by: TEAM-482
// Purpose: DeepSeek-R1 / DeepSeek-V2 safetensors support
// Reference: candle-transformers/src/models/deepseek2.rs

use candle_core::Device;
use candle_transformers::models::deepseek2::DeepSeekV2;

/// DeepSeek model wrapper for safetensors files
///
/// TEAM-482: Wraps candle-transformers `deepseek2` with safetensors support
/// Supports DeepSeek-R1 and DeepSeek-V2 architectures
pub struct DeepSeekModel {
    pub(super) model: DeepSeekV2,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    // TEAM-486: Device stored for future cache reset implementation (see llama/mod.rs reset_cache)
    #[allow(dead_code)]
    pub(super) device: Device,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl DeepSeekModel {
    /// Create a new DeepSeekModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: DeepSeekV2,
        eos_token_id: u32,
        vocab_size: usize,
        device: Device,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self { model, eos_token_id, vocab_size, device, capabilities }
    }
}
