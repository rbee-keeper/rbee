// TEAM-482: Gemma model components
//
// Created by: TEAM-482
// Purpose: Gemma safetensors support (completes existing GGUF support)
// Reference: candle-transformers/src/models/gemma.rs

use candle_core::Device;
use candle_transformers::models::gemma::Model;

/// Gemma model wrapper for safetensors files
///
/// TEAM-482: Wraps candle-transformers `gemma` with safetensors support
/// Completes Gemma support (we already have GGUF via gemma_quantized)
pub struct GemmaModel {
    pub(super) model: Model,
    pub(super) eos_token_id: u32,
    pub(super) vocab_size: usize,
    pub(super) device: Device,
    pub(super) capabilities: crate::backend::models::ModelCapabilities,
}

impl GemmaModel {
    /// Create a new GemmaModel instance
    ///
    /// TEAM-482: Constructor for internal use by loader
    pub(super) fn new(
        model: Model,
        eos_token_id: u32,
        vocab_size: usize,
        device: Device,
        capabilities: crate::backend::models::ModelCapabilities,
    ) -> Self {
        Self { model, eos_token_id, vocab_size, device, capabilities }
    }
}
