// TEAM-482: Created during loader/component separation
//! Quantized Gemma model loader
//!
//! Created by: TEAM-409
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_gemma3::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedGemmaModel;

impl QuantizedGemmaModel {
    /// Load quantized Gemma model from GGUF file
    ///
    /// TEAM-409: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Moved to separate loader module
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading Gemma GGUF model");

        // TEAM-409: Narrate GGUF loading start
        n!("gguf_gemma_load_start", "Loading Gemma GGUF model from {}", path.display());

        // Open GGUF file
        let mut file = std::fs::File::open(path).with_context(|| {
            n!("gguf_gemma_open_failed", "Failed to open Gemma GGUF file: {}", path.display());
            format!("Failed to open Gemma GGUF file at {path:?}")
        })?;

        n!("gguf_gemma_file_opened", "Gemma GGUF file opened, reading content");

        // Read GGUF content
        let content =
            candle_core::quantized::gguf_file::Content::read(&mut file).with_context(|| {
                n!(
                    "gguf_gemma_parse_failed",
                    "Failed to parse Gemma GGUF content from {}",
                    path.display()
                );
                format!("Failed to read Gemma GGUF content from {path:?}")
            })?;

        n!(
            "gguf_gemma_inspect_metadata",
            "Inspecting Gemma GGUF metadata ({} keys found)",
            content.metadata.len()
        );

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("gemma.vocab_size")
            .or_else(|| content.metadata.get("llama.vocab_size")) // Fallback for compatibility
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                // Fallback: count tokens in tokenizer array
                content.metadata
                    .get("tokenizer.ggml.tokens")
                    .and_then(|v| match v {
                        candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                        _ => None,
                    })
            })
            .with_context(|| {
                let available_keys: Vec<String> = content.metadata.keys().map(std::string::ToString::to_string).collect();
                n!("gguf_gemma_metadata_missing", "Cannot determine vocab_size from Gemma GGUF metadata");

                format!(
                    "Cannot determine vocab_size: missing gemma.vocab_size, llama.vocab_size, and tokenizer.ggml.tokens. \
                     Available keys: [{}]. This GGUF file may be incomplete or corrupted.",
                    available_keys.join(", ")
                )
            })?;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(1); // Default EOS token for Gemma (different from Llama!)

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
        let capabilities = crate::backend::models::ModelCapabilities::quantized(
            crate::backend::models::arch::GEMMA_QUANTIZED,
            8192,
        );

        Ok(Self::new(model, eos_token_id, vocab_size as usize, capabilities))
    }
}
