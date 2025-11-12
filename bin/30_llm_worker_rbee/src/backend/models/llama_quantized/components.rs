// TEAM-482: Created during loader/component separation
//! Quantized Llama model components
//!
//! Created by: TEAM-036
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::quantized_llama::ModelWeights;

/// Quantized Llama model wrapper for GGUF files
///
/// TEAM-036: Wraps candle-transformers `quantized_llama` with GGUF support
/// TEAM-482: Added capabilities
#[derive(Debug)]
pub struct QuantizedLlamaModel {
    pub(super) model: ModelWeights,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl QuantizedLlamaModel {
    /// Create a new QuantizedLlamaModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: ModelWeights,
        eos_token_id: u32,
        vocab_size: usize,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self {
            model,
            eos_token_id,
            vocab_size,
            capabilities,
        }
    }
}
