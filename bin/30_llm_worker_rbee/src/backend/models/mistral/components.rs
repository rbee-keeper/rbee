// TEAM-482: Created during loader/component separation
//! Mistral model components
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::mistral::Model;

/// Mistral model wrapper
///
/// TEAM-017: Wraps candle-transformers Mistral with its natural interface
/// TEAM-482: Added capabilities
#[derive(Debug)]
pub struct MistralModel {
    pub(super) model: Model,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl MistralModel {
    /// Create a new MistralModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: Model,
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
