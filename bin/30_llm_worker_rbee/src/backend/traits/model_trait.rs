// TEAM-482: Model trait definition - separated from models for clarity
//
// This file contains the core trait interface that all models must implement,
// along with supporting types like ModelCapabilities and architecture constants.

use anyhow::Result;
use candle_core::Tensor;

/// TEAM-485: EOS token representation - supports single or multiple tokens
///
/// Some models (like Llama 3) have multiple EOS tokens that should all
/// trigger generation stop. This enum matches Candle's pattern.
#[derive(Debug, Clone)]
pub enum EosTokens {
    /// Single EOS token (most common)
    Single(u32),
    /// Multiple EOS tokens (e.g., Llama 3: [128001, 128009])
    Multiple(Vec<u32>),
}

impl EosTokens {
    /// Check if a token is an EOS token
    pub fn is_eos(&self, token: u32) -> bool {
        match self {
            EosTokens::Single(eos) => token == *eos,
            EosTokens::Multiple(eos_list) => eos_list.contains(&token),
        }
    }

    /// Get the primary EOS token (for backward compatibility)
    pub fn primary(&self) -> u32 {
        match self {
            EosTokens::Single(eos) => *eos,
            EosTokens::Multiple(eos_list) => eos_list[0],
        }
    }

    /// Create from a single token
    pub fn single(token: u32) -> Self {
        EosTokens::Single(token)
    }

    /// Create from multiple tokens
    pub fn multiple(tokens: Vec<u32>) -> Self {
        if tokens.len() == 1 {
            EosTokens::Single(tokens[0])
        } else {
            EosTokens::Multiple(tokens)
        }
    }
}

/// TEAM-482: Sealed trait pattern - prevents external implementations
///
/// This module ensures only internal model types can implement ModelTrait,
/// maintaining type safety and preventing misuse.
mod sealed {
    pub trait Sealed {}

    // Only internal model types can implement Sealed
    // TEAM-482: Updated to use new quantized module structure + added DeepSeek + Gemma
    // TEAM-483: Added Mixtral MoE
    impl Sealed for crate::backend::models::deepseek::DeepSeekModel {}
    impl Sealed for crate::backend::models::deepseek_quantized::QuantizedDeepSeekModel {}
    impl Sealed for crate::backend::models::gemma::GemmaModel {}
    impl Sealed for crate::backend::models::llama::LlamaModel {}
    impl Sealed for crate::backend::models::llama_quantized::QuantizedLlamaModel {}
    impl Sealed for crate::backend::models::mistral::MistralModel {}
    impl Sealed for crate::backend::models::mixtral::MixtralModel {} // TEAM-483: Mixtral MoE
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

    /// TEAM-485: DType used by this model
    pub dtype: candle_core::DType,

    /// TEAM-485: Supported dtypes for this model (for validation)
    pub supported_dtypes: &'static [candle_core::DType],
}

/// TEAM-485: Static arrays for supported dtypes (for lifetime correctness)
static SAFETENSORS_SUPPORTED_DTYPES: &[candle_core::DType] = &[
    candle_core::DType::F16,
    candle_core::DType::BF16,
    candle_core::DType::F32,
];

impl ModelCapabilities {
    /// Create capabilities for a standard model
    ///
    /// TEAM-485: Now includes dtype parameter
    pub fn standard(architecture: &'static str, max_context: usize, dtype: candle_core::DType) -> Self {
        Self {
            uses_position: true,
            supports_cache_reset: true,
            max_context_length: max_context,
            supports_streaming: true,
            architecture_family: architecture,
            is_quantized: false,
            dtype,
            // TEAM-485: Safetensors models support F16, BF16, F32
            supported_dtypes: SAFETENSORS_SUPPORTED_DTYPES,
        }
    }

    /// Create capabilities for a quantized model
    ///
    /// TEAM-485: Quantized models have fixed dtype from GGUF file
    /// Note: For quantized models, we use the safetensors list but only the native dtype is actually supported
    pub fn quantized(architecture: &'static str, max_context: usize, dtype: candle_core::DType) -> Self {
        Self {
            uses_position: true,
            supports_cache_reset: true,
            max_context_length: max_context,
            supports_streaming: true,
            architecture_family: architecture,
            is_quantized: true,
            dtype,
            // TEAM-485: Quantized models technically only support their native dtype
            // but we use the same list for simplicity (validation happens elsewhere)
            supported_dtypes: SAFETENSORS_SUPPORTED_DTYPES,
        }
    }

    /// TEAM-485: Check if a dtype is supported by this model
    pub fn supports_dtype(&self, dtype: candle_core::DType) -> bool {
        // For quantized models, only the current dtype is supported
        if self.is_quantized {
            return dtype == self.dtype;
        }
        // For safetensors models, check the list
        self.supported_dtypes.contains(&dtype)
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

    /// Get the end-of-sequence token ID (backward compatible)
    ///
    /// Returns the primary EOS token. For models with multiple EOS tokens,
    /// use `eos_tokens()` instead.
    fn eos_token_id(&self) -> u32 {
        self.eos_tokens().primary()
    }

    /// Get all EOS tokens (supports multiple EOS tokens)
    ///
    /// TEAM-485: Some models (like Llama 3) have multiple EOS tokens.
    /// This method returns all of them.
    fn eos_tokens(&self) -> EosTokens;

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
    pub const MIXTRAL: &str = "mixtral"; // TEAM-483: Mixtral MoE
    pub const PHI: &str = "phi";
    pub const PHI_QUANTIZED: &str = "phi-quantized";
    pub const QWEN: &str = "qwen";
    pub const QWEN_QUANTIZED: &str = "qwen-quantized";
    pub const GEMMA: &str = "gemma"; // TEAM-482: Gemma safetensors
    pub const GEMMA_QUANTIZED: &str = "gemma-quantized";
    pub const DEEPSEEK: &str = "deepseek"; // TEAM-482: DeepSeek-R1 / DeepSeek-V2
    pub const DEEPSEEK_QUANTIZED: &str = "deepseek-quantized"; // TEAM-482: DeepSeek GGUF
}
