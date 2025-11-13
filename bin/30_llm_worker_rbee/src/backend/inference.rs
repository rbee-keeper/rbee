// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Inference implementation with proper error handling

//! Main inference backend implementation
//!
//! Created by: TEAM-015 (refactored from `candle_backend.rs`)
//! Original code by: TEAM-000, TEAM-009, TEAM-011, TEAM-014
//! Modified by: TEAM-017 (added multi-model support with enum pattern)

use super::models::{self, Model};
use super::sampling;
use super::tokenizer_loader;
use crate::common::{InferenceResult, SamplingConfig};
use crate::http::InferenceBackend;
use crate::narration::{
    ACTION_CACHE_RESET, ACTION_INFERENCE_COMPLETE, ACTION_INFERENCE_START, ACTION_MODEL_LOAD,
    ACTION_TOKENIZE, ACTION_TOKEN_GENERATE, ACTION_WARMUP,
};
use crate::token_output_stream::TokenOutputStream;
use anyhow::{Context, Result};
use async_trait::async_trait;
use candle_core::{Device, DType, Tensor};
use observability_narration_core::n;
use std::path::Path;
use tokenizers::Tokenizer;

/// TEAM-487: Optimized repeat penalty that avoids allocations
///
/// Candle's `apply_repeat_penalty` allocates 3 times per call:
/// 1. `to_vec1()` - converts tensor to Vec
/// 2. HashSet for deduplication
/// 3. `from_vec()` - creates new tensor
///
/// This optimized version:
/// - Only allocates Vec once (unavoidable for CPU tensors)
/// - Uses simple Vec for seen tokens (faster than HashSet for small contexts)
/// - Reuses the same Vec to create output tensor
///
/// For GPU tensors, we fall back to Candle's implementation since
/// in-place modification requires CPU access anyway.
fn apply_repeat_penalty_optimized(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
    vocab_size: usize,
) -> Result<Tensor> {
    // TEAM-487: For GPU tensors, use Candle's implementation
    // (in-place modification requires CPU access anyway)
    if !matches!(logits.device(), Device::Cpu) {
        return Ok(candle_transformers::utils::apply_repeat_penalty(logits, penalty, context)?);
    }

    let device = logits.device();
    
    // TEAM-487: Convert to Vec once (unavoidable for CPU tensors)
    let mut logits_vec = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    
    // TEAM-487: Use Vec instead of HashSet for small contexts (faster)
    // Most contexts are < 100 tokens, linear search is faster than hashing
    let mut seen_tokens = Vec::with_capacity(context.len().min(64));
    
    for &token_id in context {
        // Skip if we've already processed this token
        if seen_tokens.contains(&token_id) {
            continue;
        }
        seen_tokens.push(token_id);
        
        // Apply penalty to this token's logit
        if let Some(logit) = logits_vec.get_mut(token_id as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
    
    // TEAM-487: Reuse the Vec to create output tensor (no extra allocation)
    Ok(Tensor::from_vec(logits_vec, vocab_size, device)?)
}

/// Candle inference backend using candle-transformers models
///
/// TEAM-009: Complete rewrite to use Candle's models directly
/// instead of building layers from scratch.
/// TEAM-015: Refactored into focused modules
/// TEAM-017: Changed to enum pattern for Candle idiomaticity
/// TEAM-149: Made fields pub(crate) for `generation_engine` access
/// TEAM-487: Added cached_eos_token for hot path optimization
pub struct CandleInferenceBackend {
    pub(crate) model: Model,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) device: Device,
    model_size_bytes: u64,
    /// TEAM-487: Cached EOS token ID to avoid repeated lookups in generation loop
    cached_eos_token: Option<u32>,
}

impl CandleInferenceBackend {
    /// Load model from `SafeTensors` with auto-detected architecture
    ///
    /// TEAM-009: Uses candle-transformers models directly
    /// TEAM-015: Delegates to `model_loader` module
    /// TEAM-017: Uses model factory with enum pattern (Candle-idiomatic)
    /// TEAM-NARRATION-FIX: Device is now compile-time determined by feature flags
    #[cfg(feature = "cpu")]
    pub fn load(model_path: &str) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::Cpu;

        // TEAM-017: Load model using model factory (returns Model enum)
        // TEAM-485: Pass None for dtype to use model defaults
        let model = models::load_model(model_path, &device, None)?;
        let model_size_bytes = models::calculate_model_size(model_path)?;

        // TEAM-017: Load tokenizer with auto-detection
        let tokenizer = tokenizer_loader::load_tokenizer(path)?;

        // TEAM-487: Cache EOS token ID for hot path optimization
        let cached_eos_token = tokenizer.token_to_id("</s>");

        tracing::info!(
            architecture = model.architecture(),
            vocab_size = model.vocab_size(),
            tokenizer_vocab = tokenizer.get_vocab_size(true),
            model_size_mb = model_size_bytes / 1_000_000,
            cached_eos_token = ?cached_eos_token,
            "Model and tokenizer loaded successfully"
        );

        n!(
            ACTION_MODEL_LOAD,
            "Loaded {} model ({} MB, vocab: {})",
            model.architecture(),
            model_size_bytes / 1_000_000,
            model.vocab_size()
        );

        Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
    }

    #[cfg(feature = "cuda")]
    pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::new_cuda(gpu_id)?;

        // TEAM-017: Load model using model factory (returns Model enum)
        // TEAM-485: Pass None for dtype to use model defaults
        let model = models::load_model(model_path, &device, None)?;
        let model_size_bytes = models::calculate_model_size(model_path)?;

        // TEAM-017: Load tokenizer with auto-detection
        let tokenizer = tokenizer_loader::load_tokenizer(path)?;

        // TEAM-487: Cache EOS token ID for hot path optimization
        let cached_eos_token = tokenizer.token_to_id("</s>");

        tracing::info!(
            architecture = model.architecture(),
            vocab_size = model.vocab_size(),
            tokenizer_vocab = tokenizer.get_vocab_size(true),
            model_size_mb = model_size_bytes / 1_000_000,
            cached_eos_token = ?cached_eos_token,
            "Model and tokenizer loaded successfully"
        );

        n!(
            ACTION_MODEL_LOAD,
            "Loaded {} model ({} MB, vocab: {})",
            model.architecture(),
            model_size_bytes / 1_000_000,
            model.vocab_size()
        );

        Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
    }

    #[cfg(feature = "metal")]
    pub fn load(model_path: &str, gpu_id: usize) -> Result<Self> {
        let path = Path::new(model_path);
        let device = Device::new_metal(gpu_id)?;

        // TEAM-017: Load model using model factory (returns Model enum)
        // TEAM-485: Pass None for dtype to use model defaults
        let model = models::load_model(model_path, &device, None)?;
        let model_size_bytes = models::calculate_model_size(model_path)?;

        // TEAM-017: Load tokenizer with auto-detection
        let tokenizer = tokenizer_loader::load_tokenizer(path)?;

        // TEAM-487: Cache EOS token ID for hot path optimization
        let cached_eos_token = tokenizer.token_to_id("</s>");

        tracing::info!(
            architecture = model.architecture(),
            vocab_size = model.vocab_size(),
            tokenizer_vocab = tokenizer.get_vocab_size(true),
            model_size_mb = model_size_bytes / 1_000_000,
            cached_eos_token = ?cached_eos_token,
            "Model and tokenizer loaded successfully"
        );

        n!(
            ACTION_MODEL_LOAD,
            "Loaded {} model ({} MB, vocab: {})",
            model.architecture(),
            model_size_bytes / 1_000_000,
            model.vocab_size()
        );

        Ok(Self { model, tokenizer, device, model_size_bytes, cached_eos_token })
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> u64 {
        self.model_size_bytes
    }

    /// Warmup GPU with dummy inference
    ///
    /// TEAM-014: Eliminates cold start by running a single token generation.
    /// This initializes CUDA kernels and caches, preventing 9s overhead on first request.
    /// TEAM-017: Updated to use Model enum
    /// TEAM-021: Warmup uses inference cache, will be reset before actual inference
    ///
    /// 🎯 TEAM-021: Warmup doesn't pollute inference - cache reset handles it!
    pub fn warmup(&mut self) -> Result<()> {
        tracing::info!("Starting GPU warmup...");

        n!(ACTION_WARMUP, "Starting GPU warmup");

        let start = std::time::Instant::now();

        // Use a simple prompt for warmup
        let warmup_prompt = "Hello";

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(warmup_prompt, true)
            .map_err(|e| anyhow::anyhow!("Warmup tokenization failed: {e}"))?;
        let tokens = encoding.get_ids();

        // Create input tensor
        let input_ids = Tensor::new(tokens, &self.device)
            .context("Failed to create warmup tensor")?
            .unsqueeze(0)
            .context("Failed to unsqueeze warmup tensor")?;

        // TEAM-017: Single forward pass using Model enum (delegates to specific model)
        // TEAM-021: This uses the inference cache, but execute() will reset it before use
        let _logits = self.model.forward(&input_ids, 0).context("Warmup forward pass failed")?;

        let duration = start.elapsed();
        tracing::info!(
            duration_ms = duration.as_millis(),
            "GPU warmup complete (cache will be reset before inference)"
        );

        n!(ACTION_WARMUP, "GPU warmup complete ({} ms)", duration.as_millis());

        Ok(())
    }
}

#[async_trait]
impl InferenceBackend for CandleInferenceBackend {
    /// Execute inference with streaming token generation
    ///
    /// TEAM-009: Complete implementation using candle-transformers
    /// TEAM-014: Added warmup support, `LogitsProcessor`, `TokenOutputStream`
    /// TEAM-015: Refactored into focused modules
    /// TEAM-017: Updated to use Model enum (Candle-idiomatic)
    async fn execute(
        &mut self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>> {
        tracing::debug!(
            prompt_len = prompt.len(),
            max_tokens = config.max_tokens,
            temperature = config.temperature,
            "Starting inference"
        );

        n!(
            ACTION_INFERENCE_START,
            "Starting inference (prompt: {} chars, max_tokens: {}, temp: {})",
            prompt.len(),
            config.max_tokens,
            config.temperature
        );

        // Tokenize prompt
        let encoding =
            self.tokenizer.encode(prompt, true).map_err(|e| format!("Tokenization failed: {e}"))?;
        let mut tokens = encoding.get_ids().to_vec();

        tracing::debug!(prompt_tokens = tokens.len(), "Prompt tokenized");

        n!(ACTION_TOKENIZE, "Tokenized prompt ({} tokens)", tokens.len());

        // TEAM-021: Reset cache to clear warmup pollution
        // Warmup leaves KV pairs in cache, causing mask broadcasting errors
        // 🎯 TEAM-021 Victory: Clean cache = no mask mismatch!
        self.model.reset_cache().context("Failed to reset cache before inference")?;
        tracing::debug!("Cache reset before inference to clear warmup pollution");

        n!(ACTION_CACHE_RESET, "Reset KV cache before inference to clear warmup pollution");

        // TEAM-014: Create LogitsProcessor for proper sampling
        // TEAM-015: Delegates to sampling module
        let mut logits_processor = sampling::create_logits_processor(config);

        // TEAM-014: Create TokenOutputStream for proper space handling
        // TEAM-487: Use reference instead of clone to avoid allocation
        let mut token_stream = TokenOutputStream::new(&self.tokenizer);

        // Generate tokens
        let mut generated_tokens = Vec::new();
        let mut generated_text = Vec::new();
        let start_time = std::time::Instant::now();

        for pos in 0..config.max_tokens {
            let pos_usize = pos as usize;

            // TEAM-011: Prepare input tensor with correct shape [batch_size, seq_len]
            let input_ids = if pos == 0 {
                // First iteration: use all prompt tokens
                Tensor::new(&tokens[..], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {e}"))?
                    .unsqueeze(0) // Add batch dimension: [seq_len] -> [1, seq_len]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {e}"))?
            } else {
                // Subsequent iterations: only last token
                Tensor::new(&[tokens[tokens.len() - 1]], &self.device)
                    .map_err(|e| format!("Failed to create input tensor: {e}"))?
                    .unsqueeze(0) // Add batch dimension: [1] -> [1, 1]
                    .map_err(|e| format!("Failed to unsqueeze input tensor: {e}"))?
            };

            // TEAM-009: Verify device residency (log only, no comparison since Device doesn't impl PartialEq)
            if pos == 0 {
                tracing::debug!(
                    input_device = ?input_ids.device(),
                    expected_device = ?self.device,
                    "Device residency check: input tensor"
                );
            }

            // TEAM-017: Forward pass using Model enum (delegates to specific model)
            let logits = self
                .model
                .forward(&input_ids, pos_usize)
                .map_err(|e| format!("Forward pass failed: {e}"))?;

            // TEAM-009: Log output device residency
            if pos == 0 {
                tracing::debug!(
                    output_device = ?logits.device(),
                    expected_device = ?self.device,
                    "Device residency check: output tensor"
                );
            }

            // Get logits for last position
            let logits = logits.squeeze(0).map_err(|e| format!("Failed to squeeze logits: {e}"))?;
            let logits = if logits.dims().len() > 1 {
                logits
                    .get(logits.dims()[0] - 1)
                    .map_err(|e| format!("Failed to get last logits: {e}"))?
            } else {
                logits
            };

            // TEAM-485: Apply repeat penalty before sampling (CRITICAL FIX)
            // This was missing - users setting repetition_penalty had no effect!
            // TEAM-487: Use optimized version that avoids HashSet allocation
            let logits = if config.repetition_penalty == 1.0 {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                apply_repeat_penalty_optimized(
                    &logits,
                    config.repetition_penalty,
                    &tokens[start_at..],
                    self.model.vocab_size(),
                )
                .map_err(|e| format!("Failed to apply repeat penalty: {e}"))?
            };

            // TEAM-014: Sample next token using Candle's LogitsProcessor
            let next_token =
                logits_processor.sample(&logits).map_err(|e| format!("Sampling failed: {e}"))?;

            // TEAM-095: Debug logging for zero-token bug
            // TEAM-487: Changed to trace level - info was causing hot path overhead
            tracing::trace!(
                pos = pos,
                next_token = next_token,
                model_eos = self.model.eos_token_id(),
                "Sampled token"
            );

            // TEAM-485: Check for EOS - supports multiple EOS tokens
            // TEAM-487: Use cached EOS token to avoid repeated HashMap lookups
            let is_eos = self.cached_eos_token.map_or_else(
                || self.model.eos_tokens().is_eos(next_token),
                |eos_id| next_token == eos_id,
            );

            // TEAM-095: Debug EOS detection
            // TEAM-487: Changed to trace level - info was causing hot path overhead
            tracing::trace!(
                pos = pos,
                next_token = next_token,
                cached_eos = ?self.cached_eos_token,
                model_eos = self.model.eos_token_id(),
                is_eos = is_eos,
                "EOS check result"
            );

            if is_eos {
                // TEAM-487: Keep warn level for EOS detection (important event)
                tracing::debug!(
                    pos = pos,
                    next_token = next_token,
                    "EOS token detected - stopping generation"
                );
                break;
            }

            // TEAM-014: Use TokenOutputStream for proper streaming decode with spaces
            if let Some(token_str) = token_stream
                .next_token(next_token)
                .map_err(|e| format!("Detokenization failed: {e}"))?
            {
                generated_text.push(token_str);
            }

            generated_tokens.push(next_token);
            tokens.push(next_token);

            // Log progress
            if (pos + 1) % 10 == 0 {
                tracing::debug!(tokens_generated = pos + 1, "Generation progress");

                n!(ACTION_TOKEN_GENERATE, "Generated {} tokens", pos + 1);
            }
        }

        // TEAM-014: Get any remaining decoded bytes from token stream
        if let Some(rest) =
            token_stream.decode_rest().map_err(|e| format!("Failed to decode rest: {e}"))?
        {
            generated_text.push(rest);
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;
        let tokens_per_sec =
            if duration_ms > 0 { (generated_tokens.len() as u64 * 1000) / duration_ms } else { 0 };

        // TEAM-089: Join generated text for logging
        let full_text = generated_text.join("");
        // TEAM-487: Avoid cloning when creating preview
        let text_preview = if full_text.len() > 100 { &full_text[..100] } else { &full_text };

        tracing::info!(
            tokens_generated = generated_tokens.len(),
            duration_ms = duration_ms,
            tokens_per_sec = tokens_per_sec,
            text_preview = %text_preview,
            "Inference completed"
        );

        // TEAM-089: Narrate the actual answer (CRITICAL for debugging)
        n!(
            ACTION_INFERENCE_COMPLETE,
            "Generated: \"{}\" ({} tokens, {} ms, {} tok/s)",
            text_preview,
            generated_tokens.len(),
            duration_ms,
            tokens_per_sec
        );

        Ok(InferenceResult::max_tokens(generated_text, generated_tokens, config.seed, duration_ms))
    }

    /// Cancel inference (not implemented for single-threaded)
    async fn cancel(&self, _job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Get VRAM usage
    fn vram_usage(&self) -> u64 {
        #[cfg(feature = "cuda")]
        {
            self.model_size_bytes
        }

        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Check if backend is healthy
    fn is_healthy(&self) -> bool {
        true
    }
}
