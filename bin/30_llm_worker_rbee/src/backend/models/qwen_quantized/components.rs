// TEAM-482: Created during loader/component separation
//! Quantized Qwen model components
//!
//! Created by: TEAM-090
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::quantized_qwen2::ModelWeights;

/// Quantized Qwen model wrapper for GGUF files
///
/// TEAM-090: Wraps candle-transformers `quantized_qwen2` with GGUF support
/// TEAM-482: Added capabilities
pub struct QuantizedQwenModel {
    pub(super) model: ModelWeights,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl QuantizedQwenModel {
    /// Create a new QuantizedQwenModel instance
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
