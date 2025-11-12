// TEAM-482: Created during loader/component separation
//! Phi model components
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-482 (split into components/loader)

use candle_transformers::models::phi::Model;

/// Phi model wrapper
///
/// TEAM-017: Wraps candle-transformers Phi with its natural interface
/// TEAM-482: Added capabilities (Phi has special requirements)
pub struct PhiModel {
    pub(super) model: Model,
    pub(super) vocab_size: usize,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl PhiModel {
    /// Create a new PhiModel instance
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
