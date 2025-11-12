// TEAM-482: Created during loader/component separation
//! Quantized Qwen model loader
//!
//! Created by: TEAM-090
//! Refactored by: TEAM-482 (split into components/loader, now uses helper functions)

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_qwen2::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedQwenModel;
use crate::backend::models::helpers::{extract_eos_token_id, extract_vocab_size, load_gguf_content};

impl QuantizedQwenModel {
    /// Load quantized Qwen model from GGUF file
    ///
    /// TEAM-090: Loads GGUF files using candle's quantized model support
    /// TEAM-482: Refactored to use helper functions
    pub fn load(path: &Path, device: &Device, dtype: Option<candle_core::DType>) -> Result<Self> {
        tracing::info!(path = ?path, "Loading GGUF Qwen model");

        n!("gguf_load_start", "Loading GGUF Qwen model from {}", path.display());

        // TEAM-482: Use helper functions
        let (mut file, content) = load_gguf_content(path)?;
        let vocab_size = extract_vocab_size(&content, "qwen")?;
        let eos_token_id = extract_eos_token_id(&content, 151643);

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
        let capabilities = crate::backend::models::ModelCapabilities::quantized(crate::backend::models::arch::QWEN, 32768, candle_core::DType::F32); // TEAM-485: Quantized models use native dtype from GGUF

        Ok(Self::new(model, eos_token_id, vocab_size, capabilities))
    }
}
