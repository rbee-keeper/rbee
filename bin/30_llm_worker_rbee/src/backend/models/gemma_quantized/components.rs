// TEAM-482: Created during loader/component separation
//! Quantized Gemma model components
//!
//! Created by: TEAM-409
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::quantized_gemma3::ModelWeights;

/// Quantized Gemma model wrapper for GGUF files
///
/// TEAM-409: Wraps candle-transformers `quantized_gemma3` with GGUF support
/// TEAM-482: Added capabilities
#[derive(Debug)]
pub struct QuantizedGemmaModel {
    pub(super) model: ModelWeights,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl QuantizedGemmaModel {
    /// Create a new QuantizedGemmaModel instance
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
