// TEAM-482: Quantized DeepSeek model loader
//
// Created by: TEAM-482
// Purpose: Load DeepSeek models from GGUF files
// Note: DeepSeek GGUF files use the Llama quantized format

use anyhow::{Context, Result};
use candle_core::Device;
use candle_transformers::models::quantized_llama::ModelWeights;
use observability_narration_core::n;
use std::path::Path;

use super::QuantizedDeepSeekModel;
use crate::backend::models::helpers::{extract_eos_token_id, extract_vocab_size, load_gguf_content};

impl QuantizedDeepSeekModel {
    /// Load quantized DeepSeek model from GGUF file
    ///
    /// TEAM-482: Uses helper functions for clean, DRY implementation
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
        tracing::info!(path = ?path, "Loading GGUF DeepSeek model");

        n!("gguf_load_start", "Loading GGUF DeepSeek model from {}", path.display());

        // TEAM-482: Use helper functions (DRY!)
        let (mut file, content) = load_gguf_content(path)?;
        let vocab_size = extract_vocab_size(&content, "deepseek")?;
        let eos_token_id = extract_eos_token_id(&content, 2); // Default EOS token

        tracing::info!(
            vocab_size = vocab_size,
            eos_token_id = eos_token_id,
            tensors = content.tensor_infos.len(),
            "GGUF DeepSeek metadata loaded"
        );

        // Load model weights from GGUF
        let model = ModelWeights::from_gguf(content, &mut file, device)
            .with_context(|| "Failed to load DeepSeek model weights from GGUF")?;

        tracing::info!("GGUF DeepSeek model loaded successfully");

        // TEAM-482: Quantized DeepSeek capabilities
        // TEAM-485: Quantized models use native dtype from GGUF
        let capabilities = crate::backend::models::ModelCapabilities::quantized(
            crate::backend::models::arch::DEEPSEEK_QUANTIZED,
            2048, // Default GGUF context
            candle_core::DType::F32
        );

        Ok(Self::new(model, eos_token_id, vocab_size, capabilities))
    }
}
