// TEAM-405: Model configuration trait
//! Trait for model-type-specific configuration

use std::collections::HashMap;
use super::ModelType;

/// Trait for model-type-specific configuration
pub trait ModelConfig: Send + Sync {
    /// Get model type
    fn model_type(&self) -> ModelType;
    
    /// Serialize to JSON
    fn to_json(&self) -> serde_json::Value;
    
    /// Check if compatible with worker type
    fn is_compatible_with(&self, worker_type: &str) -> bool;
    
    /// Get inference parameters
    fn inference_params(&self) -> InferenceParams;
}

/// Inference parameters (generic across model types)
#[derive(Debug, Clone)]
pub struct InferenceParams {
    /// Context length (for LLMs)
    pub context_length: Option<u32>,
    
    /// Batch size
    pub batch_size: Option<u32>,
    
    /// Additional type-specific parameters
    pub additional: HashMap<String, serde_json::Value>,
}
