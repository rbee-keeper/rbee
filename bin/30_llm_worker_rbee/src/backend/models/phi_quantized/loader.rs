// TEAM-482: Created during loader/component separation
//! Quantized Phi model loader
//!
//! Created by: TEAM-090
//! Refactored by: TEAM-482 (split into components/loader, now uses helper functions)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_phi3::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedPhiModel;
use crate::backend::models::helpers::{extract_eos_token_id, extract_vocab_size, load_gguf_content};

impl QuantizedPhiModel {
    /// Load quantized Phi model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Refactored to use helper functions
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Phi model");

        n!("gguf_load_start", "Loading GGUF Phi model from {}", path.display());

        // TEAM-482: Use helper functions
        let (mut file, content) = load_gguf_content(path)?;
        let vocab_size = extract_vocab_size(&content, "phi")?;
        let eos_token_id = extract_eos_token_id(&content, 32000);

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF Phi metadata loaded"
        );

        let model = ModelWeights::from_gguf(false, content, &mut file, device)
            .with_context(|| "Failed to load Phi model weights from GGUF")?;

        n!(
            "gguf_load_complete",
            "GGUF Phi model loaded (vocab={}, eos={})",
            vocab_size,
            eos_token_id
        );

        // TEAM-482: Quantized Phi capabilities
        let capabilities = crate::backend::models::ModelCapabilities::quantized(
            crate::backend::models::arch::PHI,
            2048,
        );

        Ok(Self::new(model, eos_token_id, vocab_size, capabilities))
    }
}
