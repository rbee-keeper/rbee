// TEAM-407: Model metadata types for marketplace compatibility
//! Model metadata types
//!
//! Provides detailed model metadata for marketplace filtering and compatibility checking.
//! These types enable the "If we don't support it, it doesn't exist" philosophy.

use serde::{Deserialize, Serialize};
use tsify::Tsify;

/// Model architecture type
///
/// Represents the neural network architecture of the model.
/// Used for worker compatibility filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// Llama architecture (Meta)
    Llama,
    /// Mistral architecture
    Mistral,
    /// Phi architecture (Microsoft)
    Phi,
    /// Qwen architecture (Alibaba)
    Qwen,
    /// Gemma architecture (Google)
    Gemma,
    /// Unknown or unsupported architecture
    Unknown,
}

impl ModelArchitecture {
    /// Parse architecture from string (case-insensitive)
    pub fn from_str_flexible(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => Self::Llama,
            "mistral" => Self::Mistral,
            "phi" | "phi2" | "phi3" => Self::Phi,
            "qwen" | "qwen2" | "qwen2.5" => Self::Qwen,
            "gemma" | "gemma2" => Self::Gemma,
            _ => Self::Unknown,
        }
    }

    /// Get canonical string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Mistral => "mistral",
            Self::Phi => "phi",
            Self::Qwen => "qwen",
            Self::Gemma => "gemma",
            Self::Unknown => "unknown",
        }
    }
}

/// Model file format
///
/// Represents the storage format of model weights.
/// Different workers support different formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// GGUF format (llama.cpp)
    Gguf,
    /// PyTorch format (.bin files)
    Pytorch,
}

impl ModelFormat {
    /// Parse format from string (case-insensitive)
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "safetensors" => Some(Self::SafeTensors),
            "gguf" => Some(Self::Gguf),
            "pytorch" | "bin" => Some(Self::Pytorch),
            _ => None,
        }
    }

    /// Get canonical string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
            Self::Pytorch => "pytorch",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => ".safetensors",
            Self::Gguf => ".gguf",
            Self::Pytorch => ".bin",
        }
    }
}

/// Quantization type
///
/// Represents the precision/quantization of model weights.
/// Lower precision = smaller size, faster inference, slightly lower quality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Quantization {
    /// 16-bit floating point (high quality)
    Fp16,
    /// 32-bit floating point (full precision)
    Fp32,
    /// 4-bit quantization (method 0)
    Q4_0,
    /// 4-bit quantization (method 1)
    Q4_1,
    /// 5-bit quantization (method 0)
    Q5_0,
    /// 5-bit quantization (method 1)
    Q5_1,
    /// 8-bit quantization
    Q8_0,
}

impl Quantization {
    /// Parse quantization from string (case-insensitive)
    pub fn from_str_flexible(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "FP16" | "F16" => Some(Self::Fp16),
            "FP32" | "F32" => Some(Self::Fp32),
            "Q4_0" | "Q4-0" => Some(Self::Q4_0),
            "Q4_1" | "Q4-1" => Some(Self::Q4_1),
            "Q5_0" | "Q5-0" => Some(Self::Q5_0),
            "Q5_1" | "Q5-1" => Some(Self::Q5_1),
            "Q8_0" | "Q8-0" => Some(Self::Q8_0),
            _ => None,
        }
    }

    /// Get canonical string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fp16 => "FP16",
            Self::Fp32 => "FP32",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
        }
    }

    /// Get approximate bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Fp32 => 32.0,
            Self::Fp16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q5_0 | Self::Q5_1 => 5.0,
            Self::Q4_0 | Self::Q4_1 => 4.0,
        }
    }
}

/// Model metadata for compatibility checking
///
/// Contains all information needed to determine if a model is compatible
/// with available workers and should be shown in the marketplace.
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[serde(rename_all = "camelCase")]
pub struct ModelMetadata {
    /// Model architecture (Llama, Mistral, etc.)
    pub architecture: ModelArchitecture,

    /// Model file format (SafeTensors, GGUF, etc.)
    pub format: ModelFormat,

    /// Quantization type (if applicable)
    pub quantization: Option<Quantization>,

    /// Parameter count (e.g., "7B", "13B", "70B")
    pub parameters: String,

    /// Model size in bytes
    pub size_bytes: u64,

    /// Maximum context length (tokens)
    pub max_context_length: u32,
}

impl ModelMetadata {
    /// Create metadata from HuggingFace model info
    ///
    /// Extracts architecture, format, and other metadata from HF API response.
    pub fn from_huggingface(hf_data: &serde_json::Value) -> Self {
        // Extract architecture from tags or model_id
        let architecture = hf_data["tags"]
            .as_array()
            .and_then(|tags| {
                tags.iter()
                    .filter_map(|t| t.as_str())
                    .find_map(|tag| {
                        let arch = ModelArchitecture::from_str_flexible(tag);
                        if arch != ModelArchitecture::Unknown {
                            Some(arch)
                        } else {
                            None
                        }
                    })
            })
            .unwrap_or(ModelArchitecture::Unknown);

        // Detect format from files
        let format = hf_data["siblings"]
            .as_array()
            .and_then(|files| {
                files.iter().find_map(|file| {
                    let filename = file["rfilename"].as_str()?;
                    if filename.ends_with(".safetensors") {
                        Some(ModelFormat::SafeTensors)
                    } else if filename.ends_with(".gguf") {
                        Some(ModelFormat::Gguf)
                    } else if filename.ends_with(".bin") {
                        Some(ModelFormat::Pytorch)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(ModelFormat::SafeTensors);

        // Extract parameter count from model_id or tags
        let model_id = hf_data["modelId"].as_str().unwrap_or("");
        let parameters = extract_parameter_count(model_id);

        // Get size
        let size_bytes = hf_data["usedStorage"].as_u64().unwrap_or(0);

        // Default context length (can be overridden)
        let max_context_length = 8192;

        Self {
            architecture,
            format,
            quantization: None,
            parameters,
            size_bytes,
            max_context_length,
        }
    }

    /// Check if this model is compatible with a worker
    ///
    /// Returns true if the worker supports this model's architecture and format.
    pub fn is_compatible_with_worker(
        &self,
        worker_architectures: &[String],
        worker_formats: &[String],
    ) -> bool {
        let arch_match = worker_architectures
            .iter()
            .any(|a| a.to_lowercase() == self.architecture.as_str());

        let format_match = worker_formats
            .iter()
            .any(|f| f.to_lowercase() == self.format.as_str());

        arch_match && format_match
    }
}

/// Extract parameter count from model ID
///
/// Examples:
/// - "meta-llama/Llama-2-7b-hf" → "7B"
/// - "mistralai/Mistral-7B-v0.1" → "7B"
/// - "microsoft/phi-2" → "2.7B"
fn extract_parameter_count(model_id: &str) -> String {
    // Common patterns: "7b", "7B", "13b", "70b", etc.
    let lower = model_id.to_lowercase();

    if lower.contains("70b") {
        "70B".to_string()
    } else if lower.contains("13b") {
        "13B".to_string()
    } else if lower.contains("7b") {
        "7B".to_string()
    } else if lower.contains("3b") {
        "3B".to_string()
    } else if lower.contains("1b") {
        "1B".to_string()
    } else if lower.contains("phi-2") {
        "2.7B".to_string()
    } else if lower.contains("phi-3") {
        "3.8B".to_string()
    } else {
        "Unknown".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_parsing() {
        assert_eq!(
            ModelArchitecture::from_str_flexible("llama"),
            ModelArchitecture::Llama
        );
        assert_eq!(
            ModelArchitecture::from_str_flexible("MISTRAL"),
            ModelArchitecture::Mistral
        );
        assert_eq!(
            ModelArchitecture::from_str_flexible("unknown"),
            ModelArchitecture::Unknown
        );
    }

    #[test]
    fn test_format_parsing() {
        assert_eq!(
            ModelFormat::from_str_flexible("safetensors"),
            Some(ModelFormat::SafeTensors)
        );
        assert_eq!(
            ModelFormat::from_str_flexible("GGUF"),
            Some(ModelFormat::Gguf)
        );
        assert_eq!(ModelFormat::from_str_flexible("unknown"), None);
    }

    #[test]
    fn test_quantization_parsing() {
        assert_eq!(
            Quantization::from_str_flexible("fp16"),
            Some(Quantization::Fp16)
        );
        assert_eq!(
            Quantization::from_str_flexible("Q4_0"),
            Some(Quantization::Q4_0)
        );
        assert_eq!(Quantization::from_str_flexible("unknown"), None);
    }

    #[test]
    fn test_parameter_extraction() {
        assert_eq!(extract_parameter_count("meta-llama/Llama-2-7b-hf"), "7B");
        assert_eq!(extract_parameter_count("mistralai/Mistral-7B-v0.1"), "7B");
        assert_eq!(extract_parameter_count("microsoft/phi-2"), "2.7B");
    }

    #[test]
    fn test_compatibility_check() {
        let metadata = ModelMetadata {
            architecture: ModelArchitecture::Llama,
            format: ModelFormat::SafeTensors,
            quantization: None,
            parameters: "7B".to_string(),
            size_bytes: 14_000_000_000,
            max_context_length: 8192,
        };

        // Compatible worker
        assert!(metadata.is_compatible_with_worker(
            &["llama".to_string(), "mistral".to_string()],
            &["safetensors".to_string()]
        ));

        // Incompatible architecture
        assert!(!metadata.is_compatible_with_worker(
            &["mistral".to_string()],
            &["safetensors".to_string()]
        ));

        // Incompatible format
        assert!(!metadata.is_compatible_with_worker(
            &["llama".to_string()],
            &["gguf".to_string()]
        ));
    }
}
