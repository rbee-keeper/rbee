// TEAM-482: Created during loader/component separation
//! Quantized Phi model loader
//!
//! Created by: TEAM-090
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_phi3::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedPhiModel;

impl QuantizedPhiModel {
    /// Load quantized Phi model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Moved to separate loader module
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Phi model");

        n!("gguf_load_start", "Loading GGUF Phi model from {}", path.display());

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file at {path:?}"))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {path:?}"))?;

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("phi.vocab_size")
            .or_else(|| content.metadata.get("llama.vocab_size"))
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                content.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                    _ => None,
                })
            })
            .with_context(|| "Cannot determine vocab_size from GGUF metadata")?;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32000);

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
