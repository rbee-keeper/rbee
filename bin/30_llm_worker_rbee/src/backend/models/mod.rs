// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - Model factory with enum pattern
// TEAM-482: Refactored to trait-based abstraction - makes adding models trivial

//! Model factory - Auto-detect and load models
//!
//! Created by: TEAM-017
//! Refactored by: TEAM-017 (switched to enum pattern for Candle idiomaticity)
//! Modified by: TEAM-036 (added GGUF support for quantized models)
//! Modified by: TEAM-090 (added quantized versions for all architectures)
//! Refactored by: TEAM-482 (trait-based abstraction to make adding models trivial)

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde_json::Value;
use std::path::Path;

// TEAM-482: Restructured into directories for better organization
pub mod deepseek; // TEAM-482: DeepSeek-R1 / DeepSeek-V2
pub mod llama;
pub mod mistral;
pub mod phi;
pub mod qwen;

// TEAM-482: Quantized models at same level as regular models
pub mod deepseek_quantized; // TEAM-482: DeepSeek GGUF
pub mod gemma_quantized;
pub mod llama_quantized;
pub mod phi_quantized;
pub mod qwen_quantized;

// TEAM-482: Helper functions organized in subdirectory
pub mod helpers;

// TEAM-482: Re-export traits from traits module for convenience
pub use crate::backend::traits::{arch, ModelCapabilities, ModelTrait};

// TEAM-482: Re-export commonly used helper functions
pub use helpers::{
    calculate_model_size, detect_architecture, detect_architecture_from_gguf,
    find_safetensors_files, load_config_json,
};

/// TEAM-482: Enhanced macro with better ergonomics
///
/// This macro eliminates manual match statement updates when adding new models.
/// It delegates method calls to the ModelTrait implementation.
///
/// Features:
/// - Automatic trait method dispatch
/// - Exhaustive pattern matching (compiler enforced)
/// - Zero runtime overhead (monomorphization)
///
/// Usage: `delegate_to_model!(self, method_name, arg1, arg2, ...)`
macro_rules! delegate_to_model {
    // Mutable reference methods (forward, reset_cache)
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Model::Llama(m) => ModelTrait::$method(m, $($arg),*),
            Model::QuantizedLlama(m) => ModelTrait::$method(m, $($arg),*),
            Model::Mistral(m) => ModelTrait::$method(m, $($arg),*),
            Model::Phi(m) => ModelTrait::$method(m, $($arg),*),
            Model::QuantizedPhi(m) => ModelTrait::$method(m, $($arg),*),
            Model::Qwen(m) => ModelTrait::$method(m, $($arg),*),
            Model::QuantizedQwen(m) => ModelTrait::$method(m, $($arg),*),
            Model::QuantizedGemma(m) => ModelTrait::$method(m, $($arg),*),
            Model::DeepSeek(m) => ModelTrait::$method(m, $($arg),*), // TEAM-482
            Model::QuantizedDeepSeek(m) => ModelTrait::$method(m, $($arg),*), // TEAM-482
        }
    };
}

/// Multi-model enum using Candle's idiomatic pattern
///
/// TEAM-017: Each variant wraps a specific model type with its natural interface
/// TEAM-036: Added `QuantizedLlama` for GGUF support
/// TEAM-090: Added quantized versions for Phi and Qwen
/// TEAM-409: Added Gemma GGUF support (Mistral GGUF uses QuantizedLlama)
/// TEAM-482: Reorganized quantized models to same level as regular models + added DeepSeek
pub enum Model {
    DeepSeek(deepseek::DeepSeekModel), // TEAM-482: DeepSeek-R1 / DeepSeek-V2
    Llama(llama::LlamaModel),
    QuantizedLlama(llama_quantized::QuantizedLlamaModel), // Also handles Mistral GGUF
    Mistral(mistral::MistralModel),
    Phi(phi::PhiModel),
    QuantizedPhi(phi_quantized::QuantizedPhiModel),
    Qwen(qwen::QwenModel),
    QuantizedQwen(qwen_quantized::QuantizedQwenModel),
    QuantizedDeepSeek(deepseek_quantized::QuantizedDeepSeekModel), // TEAM-482: DeepSeek GGUF
    QuantizedGemma(gemma_quantized::QuantizedGemmaModel),
}

impl Model {
    /// Forward pass - delegates to the specific model via trait
    ///
    /// TEAM-482: Now uses macro-based delegation to ModelTrait
    /// Adding a new model only requires implementing ModelTrait and adding to the macro
    pub fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor> {
        delegate_to_model!(self, forward, input_ids, position)
    }

    /// Get EOS token ID
    ///
    /// TEAM-482: Macro-based delegation
    pub fn eos_token_id(&self) -> u32 {
        delegate_to_model!(self, eos_token_id)
    }

    /// Get model architecture name
    ///
    /// TEAM-482: Macro-based delegation
    /// Returns a static string for zero-cost abstraction
    pub fn architecture(&self) -> &'static str {
        delegate_to_model!(self, architecture)
    }

    /// Get vocab size
    ///
    /// TEAM-482: Macro-based delegation
    pub fn vocab_size(&self) -> usize {
        delegate_to_model!(self, vocab_size)
    }

    /// Reset KV cache to clear history
    ///
    /// TEAM-482: Macro-based delegation
    /// Models that don't support cache reset will return an error
    pub fn reset_cache(&mut self) -> Result<()> {
        delegate_to_model!(self, reset_cache)
    }

    /// Get model capabilities for runtime feature detection
    ///
    /// TEAM-482: Macro-based delegation
    /// Returns capabilities describing what this model supports
    pub fn capabilities(&self) -> &ModelCapabilities {
        delegate_to_model!(self, capabilities)
    }
}

/// Load model based on detected architecture
///
/// TEAM-017: Factory function that returns Model enum (Candle-idiomatic pattern)
/// TEAM-036: Added GGUF support - detects .gguf files and loads quantized models
/// TEAM-090: Added architecture detection for GGUF files
pub fn load_model(model_path: &str, device: &Device) -> Result<Model> {
    let path = Path::new(model_path);

    // TEAM-090: Check if this is a GGUF file (quantized model)
    if model_path.ends_with(".gguf") {
        // Detect architecture from GGUF metadata
        let architecture = detect_architecture_from_gguf(path)?;

        tracing::info!(
            path = %model_path,
            architecture = %architecture,
            "Detected GGUF file with architecture: {}", architecture
        );

        // Load appropriate quantized model based on architecture
        // TEAM-409: Added Mistral and Gemma GGUF support
        // TEAM-482: Updated to use new module structure + added DeepSeek
        match architecture.as_str() {
            "deepseek" => {
                // TEAM-482: DeepSeek GGUF
                let model = deepseek_quantized::QuantizedDeepSeekModel::load(path, device)?;
                Ok(Model::QuantizedDeepSeek(model))
            }
            "llama" => {
                let model = llama_quantized::QuantizedLlamaModel::load(path, device)?;
                Ok(Model::QuantizedLlama(model))
            }
            "mistral" => {
                // TEAM-409: Mistral GGUF files use the same format as Llama
                // Candle's quantized_llama loader handles both
                let model = llama_quantized::QuantizedLlamaModel::load(path, device)?;
                Ok(Model::QuantizedLlama(model))
            }
            "phi" | "phi3" => {
                let model = phi_quantized::QuantizedPhiModel::load(path, device)?;
                return Ok(Model::QuantizedPhi(model));
            }
            "qwen" | "qwen2" => {
                let model = qwen_quantized::QuantizedQwenModel::load(path, device)?;
                Ok(Model::QuantizedQwen(model))
            }
            "gemma" | "gemma2" | "gemma3" => {
                let model = gemma_quantized::QuantizedGemmaModel::load(path, device)?;
                Ok(Model::QuantizedGemma(model))
            }
            _ => bail!(
                "Unsupported quantized architecture: {} (supported: deepseek, llama, mistral, phi, qwen, gemma)",
                architecture
            ),
        }
    } else {
        // Otherwise, load from safetensors with config.json
        let config_json = load_config_json(path)?;
        let architecture = detect_architecture(&config_json)?;

        tracing::info!(
            architecture = %architecture,
            path = %model_path,
            "Detected model architecture"
        );

        match architecture.as_str() {
            "deepseek" => {
                // TEAM-482: DeepSeek-R1 / DeepSeek-V2
                let model = deepseek::DeepSeekModel::load(path, device)?;
                Ok(Model::DeepSeek(model))
            }
            "llama" => {
                let model = llama::LlamaModel::load(path, device)?;
                Ok(Model::Llama(model))
            }
            "mistral" => {
                let model = mistral::MistralModel::load(path, device)?;
                Ok(Model::Mistral(model))
            }
            "phi" => {
                let model = phi::PhiModel::load(path, device)?;
                Ok(Model::Phi(model))
            }
            "qwen" | "qwen2" => {
                let model = qwen::QwenModel::load(path, device)?;
                Ok(Model::Qwen(model))
            }
            _ => bail!("Unsupported model architecture: {}", architecture),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TEAM-482: Test that ModelTrait is object-safe
    ///
    /// This test verifies that ModelTrait can be used with dynamic dispatch,
    /// enabling plugin architectures and runtime polymorphism.
    #[test]
    fn test_model_trait_is_object_safe() {
        // This compiles only if ModelTrait is object-safe
        // We can't actually create instances without loading models,
        // but the type system will verify object safety at compile time

        fn _takes_trait_object(_model: &dyn ModelTrait) {
            // This function signature proves ModelTrait is object-safe
        }

        fn _returns_boxed_trait() -> Box<dyn ModelTrait> {
            // This return type proves we can use Box<dyn ModelTrait>
            unimplemented!("This is a compile-time test only")
        }

        fn _uses_vec_of_traits(_models: Vec<Box<dyn ModelTrait>>) {
            // This proves we can store trait objects in collections
        }

        // If this test compiles, ModelTrait is object-safe ✅
    }

    #[test]
    fn test_model_capabilities_clone() {
        // Verify ModelCapabilities is Clone (needed for flexibility)
        let caps = ModelCapabilities::standard(arch::LLAMA, 4096);
        let _cloned = caps.clone();
    }
}
