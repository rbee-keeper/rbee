// TEAM-482: Llama model components
//
// Extracted from mod.rs for better organization.
// Contains the model struct and its data structures.

use candle_core::Device;
use candle_transformers::models::llama::{Cache, Config, Llama};

/// Llama model wrapper
///
/// TEAM-017: Wraps candle-transformers Llama with its natural interface
/// TEAM-021: Added device field to support cache reset
/// TEAM-482: Added capabilities for runtime feature detection
#[derive(Debug)]
pub struct LlamaModel {
    pub(super) model: Llama,
    pub(super) cache: Cache,
    pub(super) config: Config,
    pub(super) vocab_size: usize,
    pub(super) device: Device,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl LlamaModel {
    /// Create a new LlamaModel instance
    ///
    /// TEAM-482: Constructor for use by loader
    pub(super) fn new(
        model: Llama,
        cache: Cache,
        config: Config,
        vocab_size: usize,
        device: Device,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self {
            model,
            cache,
            config,
            vocab_size,
            device,
            capabilities,
        }
    }
}
