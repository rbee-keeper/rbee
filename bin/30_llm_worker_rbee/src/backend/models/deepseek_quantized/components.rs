// TEAM-482: Quantized DeepSeek model components
//
// Created by: TEAM-482
// Purpose: DeepSeek-R1 / DeepSeek-V2 GGUF support

use candle_transformers::models::quantized_llama::ModelWeights;

/// Quantized DeepSeek model wrapper for GGUF files
///
/// TEAM-482: Uses quantized_llama loader (DeepSeek GGUF uses Llama format)
/// Similar to how Mistral GGUF uses the Llama quantized loader
#[derive(Debug)]
pub struct QuantizedDeepSeekModel {
    pub(super) model: ModelWeights,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl QuantizedDeepSeekModel {
    /// Create a new QuantizedDeepSeekModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: ModelWeights,
        eos_token_id: u32,
        vocab_size: usize,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self { model, eos_token_id, vocab_size, capabilities }
    }
}
