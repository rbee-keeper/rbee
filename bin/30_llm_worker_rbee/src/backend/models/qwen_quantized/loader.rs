// TEAM-482: Created during loader/component separation
//! Quantized Qwen model loader
//!
//! Created by: TEAM-090
//! Refactored by: TEAM-482 (split into components/loader)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedQwenModel;

impl QuantizedQwenModel {
    /// Load quantized Qwen model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Moved to separate loader module
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Qwen model");

        n!("gguf_load_start", "Loading GGUF Qwen model from {}", path.display());

        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open GGUF file at {path:?}"))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("Failed to read GGUF content from {path:?}"))?;

        // Extract metadata
        let vocab_size = content
            .metadata
            .get("qwen.vocab_size")
            .or_else(|| content.metadata.get("qwen2.vocab_size"))
            .or_else(|| content.metadata.get("llama.vocab_size"))
            .and_then(|v| v.to_u32().ok())
            .or_else(|| {
                content.metadata.get("tokenizer.ggml.tokens").and_then(|v| match v {
                    candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                    _ => None,
                })
            })
            .with_context(|| "Cannot determine vocab_size from GGUF metadata")?
            as usize;

        let eos_token_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(151643);

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF Qwen metadata loaded"
        );

        let model = ModelWeights::from_gguf(content, &mut file, device)
            .with_context(|| "Failed to load Qwen model weights from GGUF")?;

        n!("gguf_load_complete", "GGUF Qwen model loaded (vocab={}, eos={})", vocab_size, eos_token_id);

        // TEAM-482: Quantized Qwen capabilities
        let capabilities = crate::backend::models::ModelCapabilities::quantized(
            crate::backend::models::arch::QWEN,
            32768,
        );

        Ok(Self::new(model, eos_token_id, vocab_size, capabilities))
    }
}
