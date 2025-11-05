// TEAM-409: Created 2025-11-05 - Quantized Gemma GGUF support

//! Quantized Gemma model wrapper for GGUF files
//!
//! Created by: TEAM-409
//! Purpose: Load and run GGUF quantized Gemma models (`Q4_K_M`, `Q5_K_M`, etc.)
//!
//! Based on candle-transformers quantized_gemma3 implementation

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma3::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

/// Quantized Gemma model wrapper for GGUF files
///
/// TEAM-409: Wraps candle-transformers `quantized_gemma3` with GGUF support
#[derive(Debug)]
pub struct QuantizedGemmaModel {
    model: ModelWeights,
    eos_token_id: u32,
    vocab_size: usize,
}

impl QuantizedGemmaModel {
    /// Load quantized Gemma model from GGUF file
    ///
    /// TEAM-409: Loads GGUF files using candle's quantized model support
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
                n!("gguf_gemma_parse_failed", "Failed to parse Gemma GGUF content from {}", path.display());
                format!("Failed to read Gemma GGUF content from {path:?}")
            })?;

        n!("gguf_gemma_inspect_metadata", "Inspecting Gemma GGUF metadata ({} keys found)", content.metadata.len());

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
            })?
            as usize;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(1); // Default EOS token for Gemma (different from Llama!)

        n!("gguf_gemma_metadata_loaded", "Gemma GGUF metadata: vocab={}, eos={}, tensors={}", vocab_size, eos_token_id, content.tensor_infos.len());

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "Gemma GGUF metadata loaded"
        );

        n!("gguf_gemma_load_weights", "Loading {} tensors from Gemma GGUF", content.tensor_infos.len());

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device).with_context(|| {
            n!("gguf_gemma_weights_failed", "Failed to load Gemma model weights from GGUF");
            "Failed to load Gemma model weights from GGUF"
        })?;

        n!("gguf_gemma_load_complete", "Gemma GGUF model loaded (vocab={}, eos={})", vocab_size, eos_token_id);

        tracing::info!("Gemma GGUF model loaded successfully");

        Ok(Self { model, eos_token_id, vocab_size })
    }

    /// Forward pass through the model
    ///
    /// TEAM-409: Delegates to candle's quantized model
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        self.model.forward(input_ids, position).map_err(|e| anyhow::anyhow!("{}", e))
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Reset KV cache
    ///
    /// TEAM-409: Gemma doesn't expose clear_kv_cache in the public API
    /// The model manages its own cache internally
    pub fn reset_cache(&mut self) -> Result<()> {
        tracing::warn!("Cache reset not implemented for Gemma (managed internally)");
        Ok(())
    }
}
