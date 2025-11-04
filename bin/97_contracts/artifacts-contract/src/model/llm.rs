// TEAM-405: LLM model configuration
//! LLM-specific model configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::{ModelType, config::{ModelConfig, InferenceParams}};

/// LLM model configuration (HuggingFace)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlmConfig {
    // ========== CRITICAL for inference ==========
    /// Model architecture (e.g., "LlamaForCausalLM")
    pub architecture: String,
    
    /// Model family (e.g., "llama", "gpt2", "mistral")
    pub model_family: String,
    
    /// Tokenizer configuration
    pub tokenizer: TokenizerConfig,
    
    /// Context length (max tokens)
    pub context_length: u32,
    
    /// Quantization type (e.g., "Q4_K_M", "FP16")
    #[serde(default)]
    pub quantization: Option<String>,
    
    // ========== Inference settings ==========
    /// Default temperature
    #[serde(default)]
    pub default_temperature: Option<f32>,
    
    /// Default top_p
    #[serde(default)]
    pub default_top_p: Option<f32>,
    
    /// Supported languages
    #[serde(default)]
    pub languages: Vec<String>,
    
    // ========== Metadata ==========
    /// Base model (if fine-tuned)
    #[serde(default)]
    pub base_model: Option<String>,
    
    /// License
    #[serde(default)]
    pub license: Option<String>,
    
    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,
    
    // ========== HuggingFace specific ==========
    /// Files in repo
    #[serde(default)]
    pub files: Vec<String>,
    
    /// SHA hash
    #[serde(default)]
    pub sha: Option<String>,
    
    /// Is gated
    #[serde(default)]
    pub gated: bool,
    
    /// Download/like stats
    #[serde(default)]
    pub downloads: u64,
    
    /// Like count
    #[serde(default)]
    pub likes: u64,
    
    /// Last modified timestamp
    #[serde(default)]
    pub last_modified: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Created timestamp
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenizerConfig {
    /// Beginning of sequence token
    #[serde(default)]
    pub bos_token: Option<String>,
    
    /// End of sequence token
    #[serde(default)]
    pub eos_token: Option<String>,
    
    /// Padding token
    #[serde(default)]
    pub pad_token: Option<String>,
    
    /// Unknown token
    #[serde(default)]
    pub unk_token: Option<String>,
    
    /// Chat template (Jinja2 format)
    #[serde(default)]
    pub chat_template: Option<String>,
}

impl LlmConfig {
    /// Create LlmConfig from HuggingFace API response
    pub fn from_huggingface(hf_data: &serde_json::Value) -> Self {
        // Extract architecture
        let architecture = hf_data["config"]["architectures"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Extract model family
        let model_family = hf_data["config"]["model_type"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        
        // Extract tokenizer config
        let tokenizer = TokenizerConfig {
            bos_token: hf_data["config"]["tokenizer_config"]["bos_token"]
                .as_str()
                .map(|s| s.to_string()),
            eos_token: hf_data["config"]["tokenizer_config"]["eos_token"]
                .as_str()
                .map(|s| s.to_string()),
            pad_token: hf_data["config"]["tokenizer_config"]["pad_token"]
                .as_str()
                .map(|s| s.to_string()),
            unk_token: hf_data["config"]["tokenizer_config"]["unk_token"]
                .as_str()
                .map(|s| s.to_string()),
            chat_template: hf_data["config"]["tokenizer_config"]["chat_template"]
                .as_str()
                .map(|s| s.to_string()),
        };
        
        // Extract context length (try multiple fields)
        let context_length = hf_data["config"]["max_position_embeddings"]
            .as_u64()
            .or_else(|| hf_data["config"]["n_positions"].as_u64())
            .or_else(|| hf_data["config"]["max_sequence_length"].as_u64())
            .unwrap_or(4096) as u32;
        
        // Detect quantization from tags or files
        let quantization = hf_data["tags"]
            .as_array()
            .and_then(|tags| {
                tags.iter()
                    .filter_map(|v| v.as_str())
                    .find(|tag| {
                        tag.contains("gguf") || 
                        tag.contains("q4") || 
                        tag.contains("q8") ||
                        tag.contains("fp16")
                    })
                    .map(|s| s.to_string())
            });
        
        // Extract files
        let files = hf_data["siblings"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v["rfilename"].as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        
        // Parse timestamps
        let last_modified = hf_data["lastModified"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        
        let created_at = hf_data["createdAt"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&chrono::Utc));
        
        Self {
            architecture,
            model_family,
            tokenizer,
            context_length,
            quantization,
            default_temperature: None,
            default_top_p: None,
            languages: hf_data["cardData"]["language"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            base_model: hf_data["cardData"]["base_model"].as_str().map(|s| s.to_string()),
            license: hf_data["cardData"]["license"].as_str().map(|s| s.to_string()),
            tags: hf_data["tags"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default(),
            files,
            sha: hf_data["sha"].as_str().map(|s| s.to_string()),
            gated: hf_data["gated"].as_bool().unwrap_or(false),
            downloads: hf_data["downloads"].as_u64().unwrap_or(0),
            likes: hf_data["likes"].as_u64().unwrap_or(0),
            last_modified,
            created_at,
        }
    }
}

impl ModelConfig for LlmConfig {
    fn model_type(&self) -> ModelType {
        ModelType::Llm
    }
    
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::json!({}))
    }
    
    fn is_compatible_with(&self, worker_type: &str) -> bool {
        match worker_type {
            "llama-cpp" | "llama.cpp" => {
                // Check if GGUF format
                self.quantization.is_some() || 
                self.files.iter().any(|f| f.ends_with(".gguf"))
            }
            "transformers" | "huggingface" => {
                // Check if has safetensors
                self.files.iter().any(|f| f.ends_with(".safetensors"))
            }
            "openvino" => {
                // Check if has OpenVINO files
                self.files.iter().any(|f| f.ends_with(".xml") || f.ends_with(".bin"))
            }
            _ => false
        }
    }
    
    fn inference_params(&self) -> InferenceParams {
        let mut additional = HashMap::new();
        
        if let Some(temp) = self.default_temperature {
            additional.insert("temperature".to_string(), serde_json::json!(temp));
        }
        if let Some(top_p) = self.default_top_p {
            additional.insert("top_p".to_string(), serde_json::json!(top_p));
        }
        
        InferenceParams {
            context_length: Some(self.context_length),
            batch_size: None,
            additional,
        }
    }
}
