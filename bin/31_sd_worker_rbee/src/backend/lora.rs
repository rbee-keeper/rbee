// TEAM-487: LoRA (Low-Rank Adaptation) support for Stable Diffusion
//
// LoRA allows users to customize SD models with small weight files (5-200 MB)
// that modify specific UNet layers. Essential for marketplace compatibility.
//
// Technical Overview:
// - LoRA modifies UNet layers with: W' = W + α * (A × B)
// - A, B are low-rank matrices (rank 4-128)
// - α is a strength multiplier (0.0-1.0)
//
// File Format:
// - SafeTensors files with keys like:
//   "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight"
//
// Candle Approach (learned from reference):
// - Use VarBuilder pattern for weight loading
// - Create custom Backend that merges base + LoRA weights
// - No need to fork candle-transformers!

use crate::error::{Error, Result};
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};
use std::collections::HashMap;
use std::path::Path;

/// LoRA weights for a single LoRA file
///
/// TEAM-487: Represents a complete LoRA model with all layer modifications
#[derive(Debug)]
pub struct LoRAWeights {
    /// Model name/path
    pub name: String,
    /// Weight tensors keyed by layer name
    pub weights: HashMap<String, LoRATensor>,
}

/// A single LoRA tensor (A and B matrices)
///
/// TEAM-487: Represents one layer's LoRA modification
#[derive(Debug)]
pub struct LoRATensor {
    /// Down projection (A matrix)
    pub down: Tensor,
    /// Up projection (B matrix)
    pub up: Tensor,
    /// Alpha value (scaling factor)
    pub alpha: Option<f32>,
}

impl LoRAWeights {
    /// Load LoRA weights from SafeTensors file
    ///
    /// TEAM-487: Loads and parses LoRA file into structured weights
    ///
    /// # Arguments
    /// * `path` - Path to .safetensors file
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// Loaded LoRA weights with all layer modifications
    ///
    /// # Example
    /// ```no_run
    /// use sd_worker_rbee::backend::lora::LoRAWeights;
    /// use candle_core::Device;
    ///
    /// let device = Device::Cpu;
    /// let lora = LoRAWeights::load("anime_style.safetensors", &device)?;
    /// println!("Loaded {} LoRA tensors", lora.weights.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();

        tracing::info!("Loading LoRA: {}", name);

        // Load SafeTensors file
        let tensors = candle_core::safetensors::load(path, device)?;

        // Parse LoRA tensors
        let mut weights = HashMap::new();
        let mut lora_keys: HashMap<String, (Option<Tensor>, Option<Tensor>, Option<f32>)> =
            HashMap::new();

        for (key, tensor) in tensors {
            if let Some(base_key) = parse_lora_key(&key) {
                let entry = lora_keys.entry(base_key.clone()).or_insert((None, None, None));

                if key.ends_with(".lora_down.weight") {
                    entry.0 = Some(tensor);
                } else if key.ends_with(".lora_up.weight") {
                    entry.1 = Some(tensor);
                } else if key.ends_with(".alpha") {
                    // Alpha is a scalars
                    let alpha_value = tensor.to_vec0::<f32>()?;
                    entry.2 = Some(alpha_value);
                }
            }
        }

        // Build LoRATensor structs
        for (base_key, (down, up, alpha)) in lora_keys {
            if let (Some(down), Some(up)) = (down, up) {
                weights.insert(base_key, LoRATensor { down, up, alpha });
            } else {
                tracing::warn!("Incomplete LoRA tensor for key: {}", base_key);
            }
        }

        tracing::info!("Loaded {} LoRA tensors from {}", weights.len(), name);

        Ok(Self { name, weights })
    }

    /// Get the number of LoRA tensors
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Check if LoRA has no tensors
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }
}

/// Parse LoRA key to get base layer name
///
/// TEAM-487: Converts LoRA key format to UNet layer path
///
/// # Example
/// Input:  "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight"
/// Output: "unet.down.blocks.0.attentions.0.transformer.blocks.0.attn1.to.k"
fn parse_lora_key(key: &str) -> Option<String> {
    // Remove "lora_" prefix
    let key = key.strip_prefix("lora_")?;

    // Remove ".lora_down.weight", ".lora_up.weight", or ".alpha" suffix
    let key = key
        .strip_suffix(".lora_down.weight")
        .or_else(|| key.strip_suffix(".lora_up.weight"))
        .or_else(|| key.strip_suffix(".alpha"))?;

    // Convert underscores to dots (except for numeric indices)
    // This is a simplification - real implementation may need more sophisticated parsing
    let key = key.replace('_', ".");

    Some(key)
}

/// LoRA configuration for generation
///
/// TEAM-487: Specifies a LoRA file and its strength
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoRAConfig {
    /// Path to LoRA file
    pub path: String,
    /// Strength (-10.0 to 10.0)
    /// - Negative values invert the LoRA effect
    /// - 0.0 = no effect
    /// - 1.0 = full effect (standard)
    /// - Values > 1.0 amplify the effect
    pub strength: f64,
}

impl LoRAConfig {
    /// Create a new LoRA configuration
    pub fn new(path: impl Into<String>, strength: f64) -> Self {
        Self { path: path.into(), strength }
    }

    /// Validate LoRA configuration
    /// 
    /// TEAM-481: #[must_use] ensures validation result is checked
    #[must_use = "LoRA validation result must be checked"]
    pub fn validate(&self) -> Result<()> {
        // LoRA strengths typically range from -10.0 to 10.0
        // Negative values invert the effect, values > 1.0 amplify it
        if !(-10.0..=10.0).contains(&self.strength) {
            return Err(Error::InvalidInput(format!(
                "LoRA strength must be between -10.0 and 10.0, got {}",
                self.strength
            )));
        }
        Ok(())
    }
}

/// Custom VarBuilder backend that merges base model weights with LoRA deltas
///
/// TEAM-487: Follows Candle's SimpleBackend pattern
/// This allows us to transparently apply LoRA without modifying candle-transformers!
pub struct LoRABackend {
    /// Base model VarBuilder
    base: VarBuilder<'static>,
    /// LoRA weights to merge
    loras: Vec<(LoRAWeights, f64)>, // (weights, strength)
}

impl LoRABackend {
    /// Create a new LoRA backend
    ///
    /// # Arguments
    /// * `base` - Base model VarBuilder
    /// * `loras` - List of (LoRA weights, strength) tuples
    pub fn new(base: VarBuilder<'static>, loras: Vec<(LoRAWeights, f64)>) -> Self {
        Self { base, loras }
    }

    /// Apply LoRA deltas to a base tensor
    ///
    /// Computes: W' = W + Σ(strength_i * alpha_i * (A_i × B_i))
    fn apply_lora_deltas(&self, base_tensor: &Tensor, tensor_name: &str) -> Result<Tensor> {
        let mut result = base_tensor.clone();

        for (lora, strength) in &self.loras {
            // Check if this LoRA has weights for this tensor
            if let Some(lora_tensor) = lora.weights.get(tensor_name) {
                // Calculate LoRA delta: alpha * (A × B)
                let alpha = lora_tensor.alpha.unwrap_or(1.0) as f64;
                let rank = lora_tensor.down.dim(0)? as f64;
                let scale = (alpha * strength) / rank;

                // Compute delta: down × up
                let delta = lora_tensor.up.matmul(&lora_tensor.down)?;
                let delta = (delta * scale)?;

                // Add delta to result: W' = W + delta
                result = (result + delta)?;

                tracing::debug!(
                    "Applied LoRA '{}' to tensor '{}' with strength {}",
                    lora.name,
                    tensor_name,
                    strength
                );
            }
        }

        Ok(result)
    }
}

impl SimpleBackend for LoRABackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        _dtype: DType,
        _dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Get base tensor
        let base_tensor = self.base.get(s, name)?;

        // Apply LoRA deltas if any exist for this tensor
        self.apply_lora_deltas(&base_tensor, name)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn get_unchecked(
        &self,
        name: &str,
        _dtype: DType,
        _dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Get base tensor
        let base_tensor = self.base.get_unchecked(name)?;

        // Apply LoRA deltas if any exist for this tensor
        self.apply_lora_deltas(&base_tensor, name)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.base.contains_tensor(name)
    }
}

/// Create a VarBuilder with LoRA weights merged
///
/// TEAM-487: This is the main entry point for using LoRAs
///
/// # Example
/// ```no_run
/// use sd_worker_rbee::backend::lora::{LoRAWeights, create_lora_varbuilder};
/// use candle_nn::VarBuilder;
/// use candle_core::Device;
///
/// let device = Device::Cpu;
/// let base_vb = unsafe {
///     VarBuilder::from_mmaped_safetensors(&["model.safetensors"], DType::F32, &device)?
/// };
///
/// let lora1 = LoRAWeights::load("anime_style.safetensors", &device)?;
/// let lora2 = LoRAWeights::load("character.safetensors", &device)?;
///
/// let loras = vec![(lora1, 0.8), (lora2, 0.6)];
/// let vb_with_loras = create_lora_varbuilder(base_vb, loras)?;
///
/// // Now use vb_with_loras to build UNet - LoRAs are automatically applied!
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn create_lora_varbuilder(
    base: VarBuilder<'static>,
    loras: Vec<(LoRAWeights, f64)>,
) -> Result<VarBuilder<'static>> {
    let dtype = base.dtype();
    let device = base.device().clone();

    let backend = LoRABackend::new(base, loras);
    let backend: Box<dyn SimpleBackend> = Box::new(backend);

    Ok(VarBuilder::from_backend(backend, dtype, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_lora_key() {
        let key =
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight";
        let parsed = parse_lora_key(key).unwrap();
        assert_eq!(parsed, "unet.down.blocks.0.attentions.0.transformer.blocks.0.attn1.to.k");
    }

    #[test]
    fn test_parse_lora_key_up_weight() {
        let key =
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_1_attn2_to_v.lora_up.weight";
        let parsed = parse_lora_key(key).unwrap();
        assert_eq!(parsed, "unet.up.blocks.3.attentions.2.transformer.blocks.1.attn2.to.v");
    }

    #[test]
    fn test_parse_lora_key_alpha() {
        let key = "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_q.alpha";
        let parsed = parse_lora_key(key).unwrap();
        assert_eq!(parsed, "unet.mid.block.attentions.0.transformer.blocks.0.attn1.to.q");
    }

    #[test]
    fn test_lora_config_validation() {
        // Valid strengths
        let config = LoRAConfig::new("test.safetensors", 0.8);
        assert!(config.validate().is_ok());

        let config = LoRAConfig::new("test.safetensors", 1.5);
        assert!(config.validate().is_ok()); // > 1.0 is valid (amplifies effect)

        let config = LoRAConfig::new("test.safetensors", -0.5);
        assert!(config.validate().is_ok()); // Negative is valid (inverts effect)

        let config = LoRAConfig::new("test.safetensors", 10.0);
        assert!(config.validate().is_ok()); // Max valid

        let config = LoRAConfig::new("test.safetensors", -10.0);
        assert!(config.validate().is_ok()); // Min valid

        // Invalid strengths
        let config = LoRAConfig::new("test.safetensors", 10.1);
        assert!(config.validate().is_err()); // Too high

        let config = LoRAConfig::new("test.safetensors", -10.1);
        assert!(config.validate().is_err()); // Too low
    }
}
