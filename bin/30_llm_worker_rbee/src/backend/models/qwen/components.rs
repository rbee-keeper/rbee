// TEAM-482: Created during loader/component separation
//! Qwen model components
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::qwen2::ModelForCausalLM;

/// Qwen model wrapper
///
/// TEAM-017: Wraps candle-transformers Qwen2 with its natural interface
/// TEAM-482: Added capabilities
#[derive(Debug)]
pub struct QwenModel {
    pub(super) model: ModelForCausalLM,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl QwenModel {
    /// Create a new QwenModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: ModelForCausalLM,
        vocab_size: usize,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self {
            model,
            vocab_size,
            capabilities,
        }
    }
}
