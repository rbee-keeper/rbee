// TEAM-482: Model trait definition - separated from models for clarity
//
// This file contains the core trait interface that all models must implement,
// along with supporting types like ModelCapabilities and architecture constants.

use anyhow::Result;
use candle_core::Tensor;

/// TEAM-482: Sealed trait pattern - prevents external implementations
///
/// This module ensures only internal model types can implement ModelTrait,
/// maintaining type safety and preventing misuse.
mod sealed {
    pub trait Sealed {}
    
    // Only internal model types can implement Sealed
    // TEAM-482: Updated to use new quantized module structure + added DeepSeek
    impl Sealed for crate::backend::models::deepseek::DeepSeekModel {}
    impl Sealed for crate::backend::models::deepseek_quantized::QuantizedDeepSeekModel {}
    impl Sealed for crate::backend::models::llama::LlamaModel {}
    impl Sealed for crate::backend::models::llama_quantized::QuantizedLlamaModel {}
    impl Sealed for crate::backend::models::mistral::MistralModel {}
    impl Sealed for crate::backend::models::phi::PhiModel {}
    impl Sealed for crate::backend::models::phi_quantized::QuantizedPhiModel {}
    impl Sealed for crate::backend::models::qwen::QwenModel {}
    impl Sealed for crate::backend::models::qwen_quantized::QuantizedQwenModel {}
    impl Sealed for crate::backend::models::gemma_quantized::QuantizedGemmaModel {}
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
    pub const DEEPSEEK: &str = "deepseek"; // TEAM-482: DeepSeek-R1 / DeepSeek-V2
    pub const DEEPSEEK_QUANTIZED: &str = "deepseek-quantized"; // TEAM-482: DeepSeek GGUF
}
