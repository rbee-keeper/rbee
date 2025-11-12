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

pub mod llama;
pub mod mistral;
pub mod phi;
pub mod quantized_gemma;
pub mod quantized_llama;
pub mod quantized_phi;
pub mod quantized_qwen;
pub mod qwen;

/// TEAM-482: Sealed trait pattern - prevents external implementations
///
/// This module ensures only internal model types can implement ModelTrait,
/// maintaining type safety and preventing misuse.
mod sealed {
    pub trait Sealed {}
    
    // Only internal model types can implement Sealed
    impl Sealed for super::llama::LlamaModel {}
    impl Sealed for super::quantized_llama::QuantizedLlamaModel {}
    impl Sealed for super::mistral::MistralModel {}
    impl Sealed for super::phi::PhiModel {}
    impl Sealed for super::quantized_phi::QuantizedPhiModel {}
    impl Sealed for super::qwen::QwenModel {}
    impl Sealed for super::quantized_qwen::QuantizedQwenModel {}
    impl Sealed for super::quantized_gemma::QuantizedGemmaModel {}
}

/// TEAM-482: Model capabilities for runtime feature detection
///
/// Models declare their capabilities, allowing the generation engine
/// to query what features are supported without hardcoded checks.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Whether the model uses position parameter in forward pass
    pub uses_position: bool,
    
    /// Whether the model supports cache reset
    pub supports_cache_reset: bool,
    
    /// Maximum context length (in tokens)
    pub max_context_length: usize,
    
    /// Whether the model supports streaming generation
    pub supports_streaming: bool,
    
    /// Model architecture family (e.g., "llama", "mistral", "phi")
    pub architecture_family: &'static str,
    
    /// Whether the model is quantized (GGUF)
    pub is_quantized: bool,
}

impl ModelCapabilities {
    /// Create capabilities for a standard model
    pub fn standard(architecture: &'static str, max_context: usize) -> Self {
        Self {
            uses_position: true,
            supports_cache_reset: true,
            max_context_length: max_context,
            supports_streaming: true,
            architecture_family: architecture,
            is_quantized: false,
        }
    }
    
    /// Create capabilities for a quantized model
    pub fn quantized(architecture: &'static str, max_context: usize) -> Self {
        Self {
            uses_position: true,
            supports_cache_reset: true,
            max_context_length: max_context,
            supports_streaming: true,
            architecture_family: architecture,
            is_quantized: true,
        }
    }
}

/// TEAM-482: Common interface that all models must implement
///
/// This trait enforces a consistent API across all model implementations,
/// making it trivial to add new models. Simply implement this trait and
/// add one line to the Model enum.
///
/// Design decisions:
/// - `forward` takes `position` parameter (models that don't need it ignore it)
/// - `reset_cache` must be implemented (return error if not supported)
/// - All methods are required (no silent failures)
/// - Sealed trait prevents external implementations (type safety)
/// - Capabilities enable runtime feature detection
/// - **Object-safe**: Can use `Box<dyn ModelTrait>` for true polymorphism
///
/// # Object Safety
///
/// This trait is object-safe, enabling dynamic dispatch:
/// ```ignore
/// let model: Box<dyn ModelTrait> = Box::new(llama_model);
/// let models: Vec<Box<dyn ModelTrait>> = vec![...];
/// ```
pub trait ModelTrait: sealed::Sealed {
    /// Forward pass through the model
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs as a tensor
    /// * `position` - Current position in the sequence (for KV cache)
    ///
    /// # Note
    /// Models that don't use position (like Phi) can ignore this parameter
    fn forward(&mut self, input_ids: &Tensor, position: usize) -> Result<Tensor>;

    /// Get the end-of-sequence token ID
    fn eos_token_id(&self) -> u32;

    /// Get the model architecture name (e.g., "llama", "mistral")
    /// 
    /// # Note
    /// Returns a static string for zero-cost abstraction
    fn architecture(&self) -> &'static str;

    /// Get the vocabulary size
    fn vocab_size(&self) -> usize;

    /// Reset the KV cache to clear history between requests
    ///
    /// # Errors
    /// Return an error if cache reset is not supported for this model
    fn reset_cache(&mut self) -> Result<()>;
    
    /// Get model capabilities for runtime feature detection
    ///
    /// # Returns
    /// Reference to capabilities struct describing what this model supports
    fn capabilities(&self) -> &ModelCapabilities;
}

/// TEAM-482: Architecture name constants for type safety
///
/// Using constants ensures consistency and enables compile-time checks
pub mod arch {
    pub const LLAMA: &str = "llama";
    pub const LLAMA_QUANTIZED: &str = "llama-quantized";
    pub const MISTRAL: &str = "mistral";
    pub const PHI: &str = "phi";
    pub const PHI_QUANTIZED: &str = "phi-quantized";
    pub const QWEN: &str = "qwen";
    pub const QWEN_QUANTIZED: &str = "qwen-quantized";
    pub const GEMMA_QUANTIZED: &str = "gemma-quantized";
}

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
        }
    };
}

/// Multi-model enum using Candle's idiomatic pattern
///
/// TEAM-017: Each variant wraps a specific model type with its natural interface
/// TEAM-036: Added `QuantizedLlama` for GGUF support
/// TEAM-090: Added quantized versions for Phi and Qwen
/// TEAM-409: Added Gemma GGUF support (Mistral GGUF uses QuantizedLlama)
pub enum Model {
    Llama(llama::LlamaModel),
    QuantizedLlama(quantized_llama::QuantizedLlamaModel),  // Also handles Mistral GGUF
    Mistral(mistral::MistralModel),
    Phi(phi::PhiModel),
    QuantizedPhi(quantized_phi::QuantizedPhiModel),
    Qwen(qwen::QwenModel),
    QuantizedQwen(quantized_qwen::QuantizedQwenModel),
    QuantizedGemma(quantized_gemma::QuantizedGemmaModel),
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

/// Detect model architecture from config.json
///
/// TEAM-017: Checks `model_type` and architectures fields
pub fn detect_architecture(config_json: &Value) -> Result<String> {
    // Check "model_type" field
    if let Some(model_type) = config_json.get("model_type").and_then(|v| v.as_str()) {
        return Ok(model_type.to_lowercase());
    }

    // Check "architectures" array
    if let Some(archs) = config_json.get("architectures").and_then(|v| v.as_array()) {
        if let Some(arch) = archs.first().and_then(|v| v.as_str()) {
            let arch_lower = arch.to_lowercase();
            // Normalize architecture names
            if arch_lower.contains("llama") {
                return Ok("llama".to_string());
            } else if arch_lower.contains("mistral") {
                return Ok("mistral".to_string());
            } else if arch_lower.contains("phi") {
                return Ok("phi".to_string());
            } else if arch_lower.contains("qwen") {
                return Ok("qwen".to_string());
            } else if arch_lower.contains("gemma") {
                return Ok("gemma".to_string());
            }
            return Ok(arch_lower);
        }
    }

    bail!("Could not detect model architecture from config.json");
}

/// Scan for safetensors files
///
/// TEAM-017: Candle-idiomatic helper to find safetensors files
pub(super) fn find_safetensors_files(
    path: &Path,
) -> Result<(std::path::PathBuf, Vec<std::path::PathBuf>)> {
    if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        Ok((parent.to_path_buf(), vec![path.to_path_buf()]))
    } else if path.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                files.push(entry_path);
            }
        }
        if files.is_empty() {
            bail!("No safetensors files found at {}", path.display());
        }
        Ok((path.to_path_buf(), files))
    } else {
        bail!("Path must be a .safetensors file or directory");
    }
}

/// Load config.json from model path
///
/// TEAM-017: Helper to load and parse config.json
fn load_config_json(model_path: &Path) -> Result<Value> {
    let parent = if model_path.is_dir() {
        model_path
    } else {
        model_path.parent().unwrap_or_else(|| Path::new("."))
    };

    let config_path = parent.join("config.json");
    let config_json: Value = serde_json::from_reader(
        std::fs::File::open(&config_path)
            .with_context(|| format!("Failed to open config.json at {config_path:?}"))?,
    )
    .context("Failed to parse config.json")?;

    Ok(config_json)
}

/// Detect architecture from GGUF metadata
///
/// TEAM-090: Read GGUF file and extract architecture from general.architecture field
fn detect_architecture_from_gguf(gguf_path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Failed to open GGUF file: {}", gguf_path.display()))?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .with_context(|| format!("Failed to read GGUF content from {}", gguf_path.display()))?;

    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            candle_core::quantized::gguf_file::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .context("Missing general.architecture in GGUF metadata")?;

    Ok(arch)
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
        match architecture.as_str() {
            "llama" => {
                let model = quantized_llama::QuantizedLlamaModel::load(path, device)?;
                Ok(Model::QuantizedLlama(model))
            }
            "mistral" => {
                // TEAM-409: Mistral GGUF files use the same format as Llama
                // Candle's quantized_llama loader handles both
                let model = quantized_llama::QuantizedLlamaModel::load(path, device)?;
                Ok(Model::QuantizedLlama(model))
            }
            "phi" | "phi3" => {
                let model = quantized_phi::QuantizedPhiModel::load(path, device)?;
                Ok(Model::QuantizedPhi(model))
            }
            "qwen" | "qwen2" => {
                let model = quantized_qwen::QuantizedQwenModel::load(path, device)?;
                Ok(Model::QuantizedQwen(model))
            }
            "gemma" | "gemma2" | "gemma3" => {
                let model = quantized_gemma::QuantizedGemmaModel::load(path, device)?;
                Ok(Model::QuantizedGemma(model))
            }
            _ => bail!(
                "Unsupported quantized architecture: {} (supported: llama, mistral, phi, qwen, gemma)",
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

/// Calculate model size in bytes from safetensors or GGUF files
///
/// TEAM-017: Helper to calculate total model size
/// TEAM-036: Added GGUF support
pub fn calculate_model_size(model_path: &str) -> Result<u64> {
    let path = Path::new(model_path);

    // TEAM-036: Handle GGUF files
    if model_path.ends_with(".gguf") {
        let metadata = std::fs::metadata(path)?;
        return Ok(metadata.len());
    }

    // Handle safetensors files
    let safetensor_files =
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            vec![path.to_path_buf()]
        } else if path.is_dir() {
            let mut files = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    files.push(entry_path);
                }
            }
            files
        } else {
            bail!("Path must be a .safetensors, .gguf file or directory");
        };

    if safetensor_files.is_empty() {
        bail!("No safetensors files found at {}", path.display());
    }

    let model_size_bytes: u64 =
        safetensor_files.iter().filter_map(|p| std::fs::metadata(p).ok()).map(|m| m.len()).sum();

    Ok(model_size_bytes)
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
