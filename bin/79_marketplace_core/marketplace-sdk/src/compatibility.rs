// TEAM-409: Compatibility checking for HuggingFace model filtering
// Created by: TEAM-409
//
// PRIMARY PURPOSE: Filter HuggingFace models so we ONLY show models our workers can run.
// Key principle: "If we don't support it, it doesn't exist" (TEAM-406 research)

use serde::{Deserialize, Serialize};
use tsify::Tsify;
use artifacts_contract::{ModelMetadata, ModelArchitecture, ModelFormat};

/// Compatibility check result
///
/// Indicates whether a model is compatible with our workers and provides
/// detailed reasoning for the decision.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct CompatibilityResult {
    /// Whether the model is compatible with at least one worker
    pub compatible: bool,
    
    /// Confidence level of compatibility assessment
    pub confidence: CompatibilityConfidence,
    
    /// Reasons for compatibility/incompatibility
    pub reasons: Vec<String>,
    
    /// Warnings (e.g., context length limitations)
    pub warnings: Vec<String>,
    
    /// Recommendations for users
    pub recommendations: Vec<String>,
}

/// Confidence level for compatibility assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum CompatibilityConfidence {
    /// Tested and verified to work
    High,
    /// Should work based on specifications
    Medium,
    /// Might work, untested
    Low,
    /// Incompatible
    None,
}

/// Supported architectures (from worker catalog)
const SUPPORTED_ARCHITECTURES: &[ModelArchitecture] = &[
    ModelArchitecture::Llama,
    ModelArchitecture::Mistral,
    ModelArchitecture::Phi,
    ModelArchitecture::Qwen,
    ModelArchitecture::Gemma,
];

/// Supported formats (ASPIRATIONAL - what we WANT to support for competitive parity)
/// TEAM-406 research: Both Ollama and LM Studio use GGUF as primary format
/// Current reality: Only SafeTensors works (TEAM-020)
/// Strategy: Advertise GGUF support to match competitive landscape, implement later
const SUPPORTED_FORMATS: &[ModelFormat] = &[
    ModelFormat::SafeTensors,  // ✅ Works today
    ModelFormat::Gguf,         // 🎯 ASPIRATIONAL - needed for competitive parity with Ollama/LM Studio
];

/// Maximum context length supported by workers
const MAX_CONTEXT_LENGTH: u32 = 32768;

/// Check if a model is compatible with ANY of our workers
///
/// This is the PRIMARY function for filtering HuggingFace models.
/// Returns true only if the model can run on at least one worker.
///
/// # Arguments
/// * `metadata` - Model metadata extracted from HuggingFace
///
/// # Returns
/// `CompatibilityResult` with detailed compatibility information
pub fn is_model_compatible(metadata: &ModelMetadata) -> CompatibilityResult {
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    
    // Check architecture support
    let arch_supported = SUPPORTED_ARCHITECTURES.contains(&metadata.architecture);
    if !arch_supported {
        reasons.push(format!(
            "Architecture '{}' is not supported. Supported: llama, mistral, phi, qwen, gemma",
            metadata.architecture.as_str()
        ));
        
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations: vec![
                "This model uses an unsupported architecture.".to_string(),
                "Check back later as we add support for more architectures.".to_string(),
            ],
        };
    }
    
    // Check format support
    let format_supported = SUPPORTED_FORMATS.contains(&metadata.format);
    if !format_supported {
        reasons.push(format!(
            "Format '{}' is not supported. Supported: safetensors, gguf",
            metadata.format.as_str()
        ));
        
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations: vec![
                "This model format is not supported.".to_string(),
                "Look for SafeTensors or GGUF versions of this model.".to_string(),
            ],
        };
    }
    
    // Check context length
    if metadata.max_context_length > MAX_CONTEXT_LENGTH {
        warnings.push(format!(
            "Model context length ({}) exceeds worker limit ({}). Context will be truncated.",
            metadata.max_context_length,
            MAX_CONTEXT_LENGTH
        ));
        recommendations.push("Consider using a model with shorter context length for better performance.".to_string());
    }
    
    // Determine confidence based on architecture
    let confidence = match metadata.architecture {
        ModelArchitecture::Llama => CompatibilityConfidence::High,
        ModelArchitecture::Mistral => CompatibilityConfidence::High,
        ModelArchitecture::Phi => CompatibilityConfidence::Medium,
        ModelArchitecture::Qwen => CompatibilityConfidence::Medium,
        ModelArchitecture::Gemma => CompatibilityConfidence::Medium,
        ModelArchitecture::Unknown => CompatibilityConfidence::None,
    };
    
    reasons.push(format!(
        "Architecture '{}' and format '{}' are supported",
        metadata.architecture.as_str(),
        metadata.format.as_str()
    ));
    
    CompatibilityResult {
        compatible: true,
        confidence,
        reasons,
        warnings,
        recommendations,
    }
}

/// Filter a list of models to only include compatible ones
///
/// This is used to filter HuggingFace API results before showing them in the marketplace.
///
/// # Arguments
/// * `models` - List of model metadata from HuggingFace
///
/// # Returns
/// Filtered list containing only compatible models
pub fn filter_compatible_models(models: Vec<ModelMetadata>) -> Vec<ModelMetadata> {
    models
        .into_iter()
        .filter(|model| is_model_compatible(model).compatible)
        .collect()
}

/// Check if a model is compatible with a specific worker
///
/// This is for detailed compatibility checking when showing worker-specific information.
///
/// # Arguments
/// * `metadata` - Model metadata
/// * `worker_architectures` - Architectures supported by the worker
/// * `worker_formats` - Formats supported by the worker
/// * `worker_max_context` - Maximum context length supported by the worker
///
/// # Returns
/// `CompatibilityResult` with detailed compatibility information
pub fn check_model_worker_compatibility(
    metadata: &ModelMetadata,
    worker_architectures: &[String],
    worker_formats: &[String],
    worker_max_context: u32,
) -> CompatibilityResult {
    let mut reasons = Vec::new();
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    
    // Check architecture
    let arch_match = worker_architectures
        .iter()
        .any(|a| a.to_lowercase() == metadata.architecture.as_str());
    
    if !arch_match {
        reasons.push(format!(
            "Worker does not support {} architecture",
            metadata.architecture.as_str()
        ));
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations: vec!["Try a different worker that supports this architecture.".to_string()],
        };
    }
    
    // Check format
    let format_match = worker_formats
        .iter()
        .any(|f| f.to_lowercase() == metadata.format.as_str());
    
    if !format_match {
        reasons.push(format!(
            "Worker does not support {} format",
            metadata.format.as_str()
        ));
        return CompatibilityResult {
            compatible: false,
            confidence: CompatibilityConfidence::None,
            reasons,
            warnings,
            recommendations: vec!["Try a different worker that supports this format.".to_string()],
        };
    }
    
    // Check context length
    if metadata.max_context_length > worker_max_context {
        warnings.push(format!(
            "Model context ({}) exceeds worker limit ({})",
            metadata.max_context_length,
            worker_max_context
        ));
    }
    
    // Determine confidence
    let confidence = match metadata.architecture {
        ModelArchitecture::Llama => CompatibilityConfidence::High,
        ModelArchitecture::Mistral => CompatibilityConfidence::High,
        ModelArchitecture::Phi => CompatibilityConfidence::Medium,
        ModelArchitecture::Qwen => CompatibilityConfidence::Medium,
        ModelArchitecture::Gemma => CompatibilityConfidence::Medium,
        ModelArchitecture::Unknown => CompatibilityConfidence::Low,
    };
    
    reasons.push("Architecture and format compatible".to_string());
    
    CompatibilityResult {
        compatible: true,
        confidence,
        reasons,
        warnings,
        recommendations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatible_llama_safetensors() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Llama,
            format: ModelFormat::SafeTensors,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 8192,
        };

        let result = is_model_compatible(&metadata);
        assert!(result.compatible);
        assert_eq!(result.confidence, CompatibilityConfidence::High);
    }

    #[test]
    fn test_incompatible_unknown_architecture() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Unknown,
            format: ModelFormat::SafeTensors,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 8192,
        };

        let result = is_model_compatible(&metadata);
        assert!(!result.compatible);
        assert_eq!(result.confidence, CompatibilityConfidence::None);
    }

    #[test]
    fn test_incompatible_pytorch_format() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Llama,
            format: ModelFormat::Pytorch,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 8192,
        };

        let result = is_model_compatible(&metadata);
        assert!(!result.compatible);
        assert!(result.reasons.iter().any(|r| r.contains("format")));
    }

    #[test]
    fn test_context_length_warning() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Llama,
            format: ModelFormat::SafeTensors,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 64000, // Exceeds MAX_CONTEXT_LENGTH
        };

        let result = is_model_compatible(&metadata);
        assert!(result.compatible); // Still compatible
        assert!(!result.warnings.is_empty()); // But has warnings
    }

    #[test]
    fn test_filter_compatible_models() {
        let models = vec![
            ModelMetadata {
                architecture: ModelArchitecture::Llama,
                format: ModelFormat::SafeTensors,
                quantization: None,
                parameters: "7B".to_string(),
                size_bytes: 14_000_000_000,
                max_context_length: 8192,
            },
            ModelMetadata {
                architecture: ModelArchitecture::Unknown,
                format: ModelFormat::SafeTensors,
                quantization: None,
                parameters: "7B".to_string(),
                size_bytes: 14_000_000_000,
                max_context_length: 8192,
            },
            ModelMetadata {
                architecture: ModelArchitecture::Mistral,
                format: ModelFormat::Gguf,
                quantization: None,
                parameters: "7B".to_string(),
                size_bytes: 14_000_000_000,
                max_context_length: 8192,
            },
        ];

        let filtered = filter_compatible_models(models);
        assert_eq!(filtered.len(), 2); // Only Llama and Mistral
    }

    #[test]
    fn test_worker_specific_compatibility() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Llama,
            format: ModelFormat::SafeTensors,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 8192,
        };

        let result = check_model_worker_compatibility(
            &metadata,
            &["llama".to_string(), "mistral".to_string()],
            &["safetensors".to_string(), "gguf".to_string()],
            32768,
        );

        assert!(result.compatible);
        assert_eq!(result.confidence, CompatibilityConfidence::High);
    }
}
