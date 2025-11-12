// TEAM-482: Created during loader/component separation
//! Quantized Llama model loader
//!
//! Created by: TEAM-036
//! Modified by: TEAM-088 (added comprehensive narration for debugging)
//! Refactored by: TEAM-482 (split into components/loader, now uses helper functions)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedLlamaModel;
use crate::backend::models::helpers::{extract_eos_token_id, extract_vocab_size, load_gguf_content};

impl QuantizedLlamaModel {
    /// Load quantized Llama model from GGUF file
    ///
    /// TEAM-036: Loads GGUF files using candle's quantized model support
    /// TEAM-088: Added comprehensive narration for debugging
    /// TEAM-482: Refactored to use helper functions
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF model");

        // TEAM-088: Narrate GGUF loading start
        n!("gguf_load_start", "Loading GGUF model from {}", path.display());

        // TEAM-482: Use helper function to load GGUF content
        let (mut file, content) = load_gguf_content(path)?;

        // TEAM-088: Narrate metadata inspection
        n!(
            "gguf_inspect_metadata",
            "Inspecting GGUF metadata ({} keys found)",
            content.metadata.len()
        );

        // TEAM-088: List all available metadata keys for debugging
        let available_keys: Vec<String> =
            content.metadata.keys().map(std::string::ToString::to_string).collect();
        tracing::debug!(keys = ?available_keys, "Available GGUF metadata keys");

        // TEAM-482: Use helper functions to extract metadata
        let vocab_size = extract_vocab_size(&content, "llama")?;
        let eos_token_id = extract_eos_token_id(&content, 2); // Default EOS token for Llama

        // TEAM-088: Narrate successful metadata extraction
        n!(
            "gguf_metadata_loaded",
            "GGUF metadata: vocab={}, eos={}, tensors={}",
            vocab_size,
            eos_token_id,
            content.tensor_infos.len()
        );

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF metadata loaded"
        );

        // TEAM-088: Narrate model weight loading
        n!("gguf_load_weights", "Loading {} tensors from GGUF", content.tensor_infos.len());

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device).with_context(|| {
            // TEAM-088: Narrate weight loading failure
            n!("gguf_weights_failed", "Failed to load model weights from GGUF");
            "Failed to load model weights from GGUF"
        })?;

        // TEAM-088: Narrate successful load
        n!("gguf_load_complete", "GGUF model loaded (vocab={}, eos={})", vocab_size, eos_token_id);

        tracing::info!("GGUF model loaded successfully");

        // TEAM-482: Quantized model capabilities
        // TEAM-485: Quantized models use native dtype from GGUF
        let capabilities = crate::backend::models::ModelCapabilities::quantized(
            crate::backend::models::arch::LLAMA,
            2048, // Default GGUF context
            candle_core::DType::F32
        );

        Ok(Self::new(model, eos_token_id, vocab_size, capabilities))
    }
}
