// TEAM-482: Created during loader/component separation
//! Quantized Gemma model loader
//!
//! Created by: TEAM-409
//! Refactored by: TEAM-482 (split into components/loader, now uses helper functions)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_gemma3::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedGemmaModel;
use crate::backend::models::helpers::{
    extract_eos_token_id, extract_vocab_size, load_gguf_content,
};

impl QuantizedGemmaModel {
    /// Load quantized Gemma model from GGUF file
    ///
    /// TEAM-409: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Refactored to use helper functions
    /// TEAM-486: dtype parameter is ignored - GGUF files have fixed quantization format
    ///
    /// # Arguments
    /// * `dtype` - Ignored for GGUF files (quantization format is in file metadata)
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        // TEAM-486: Validate dtype parameter - warn if user tries to override GGUF format
        if let Some(requested_dtype) = dtype {
            tracing::warn!(
                requested_dtype = ?requested_dtype,
                "dtype parameter ignored for GGUF files - quantization format is fixed in file metadata"
            );
        }
        tracing::info!(path = ?path, "Loading GGUF Gemma model");

        n!("gguf_load_start", "Loading GGUF Gemma model from {}", path.display());

        // TEAM-482: Use helper functions
        let (mut file, content) = load_gguf_content(path)?;

        // TEAM-409: Narrate metadata inspection
        n!(
            "gguf_inspect_metadata",
            "Inspecting GGUF metadata ({} keys found)",
            content.metadata.len()
        );

        // List available keys for debugging
        let available_keys: Vec<String> =
            content.metadata.keys().map(std::string::ToString::to_string).collect();
        tracing::debug!(keys = ?available_keys, "Available GGUF metadata keys");

        // TEAM-482: Use helper functions to extract metadata
        let vocab_size = extract_vocab_size(&content, "gemma")?;
        // TEAM-409: Gemma uses EOS token ID 1 by default (different from Llama's 2)
        let eos_token_id = extract_eos_token_id(&content, 1); // Default EOS token for Gemma (different from Llama!)

        n!(
            "gguf_gemma_metadata_loaded",
            "Gemma GGUF metadata: vocab={}, eos={}, tensors={}",
            vocab_size,
            eos_token_id,
            content.tensor_infos.len()
        );

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "Gemma GGUF metadata loaded"
        );

        n!(
            "gguf_gemma_load_weights",
            "Loading {} tensors from Gemma GGUF",
            content.tensor_infos.len()
        );

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device).with_context(|| {
            n!("gguf_gemma_weights_failed", "Failed to load Gemma model weights from GGUF");
            "Failed to load Gemma model weights from GGUF"
        })?;

        n!(
            "gguf_gemma_load_complete",
            "Gemma GGUF model loaded (vocab={}, eos={})",
            vocab_size,
            eos_token_id
        );

        tracing::info!("Gemma GGUF model loaded successfully");

        // TEAM-482: Quantized Gemma capabilities
        let capabilities = crate::backend::models::ModelCapabilities::quantized(crate::backend::models::arch::GEMMA_QUANTIZED, 8192, candle_core::DType::F32); // TEAM-485: Quantized models use native dtype from GGUF

        Ok(Self::new(model, eos_token_id, vocab_size as usize, capabilities))
    }
}
