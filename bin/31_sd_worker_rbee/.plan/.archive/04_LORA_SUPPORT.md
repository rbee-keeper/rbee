# LoRA Support Implementation

**Priority:** 🟠 HIGH - SHOULD HAVE  
**Estimated Effort:** 5-7 days  
**Status:** ❌ NOT IMPLEMENTED  
**Assignee:** TBD

---

## Problem

**Current State:** No LoRA support exists

**Impact:**
- CivitAI has **100,000+ LoRA models**
- LoRA is the #1 way users customize SD models
- Without LoRA, worker is severely limited
- Marketplace compatibility is minimal

**LoRA = Low-Rank Adaptation:**
- Small weight files (5-200 MB) that modify base models
- Add styles, characters, concepts without retraining
- Can stack multiple LoRAs (e.g., anime style + specific character)
- Essential for professional workflows

---

## What Is LoRA?

**Technical Overview:**
- LoRA modifies specific layers of the UNet
- Adds small weight matrices (rank 4-128)
- Applied as: `W' = W + α * (A × B)`
  - `W` = Original weight
  - `A`, `B` = LoRA matrices (low-rank)
  - `α` = Strength multiplier (0.0-1.0)

**File Format:**
- SafeTensors files with specific key patterns
- Keys like: `lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight`
- Organized by UNet layer structure

---

## Implementation Plan

### Step 1: LoRA Weight Loading

**File:** `src/backend/lora.rs` (NEW FILE)

```rust
use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;
use std::path::Path;

/// LoRA weights for a single LoRA file
#[derive(Debug)]
pub struct LoRAWeights {
    /// Model name/path
    pub name: String,
    /// Weight tensors keyed by layer name
    pub weights: HashMap<String, LoRATensor>,
}

/// A single LoRA tensor (A and B matrices)
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
    /// # Arguments
    /// * `path` - Path to .safetensors file
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    /// Loaded LoRA weights
    pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let path = path.as_ref();
        let name = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        tracing::info!("Loading LoRA: {}", name);
        
        // Load SafeTensors file
        let tensors = unsafe {
            candle_core::safetensors::load(path, device)?
        };
        
        // Parse LoRA tensors
        let mut weights = HashMap::new();
        let mut lora_keys: HashMap<String, (Option<Tensor>, Option<Tensor>, Option<f32>)> = HashMap::new();
        
        for (key, tensor) in tensors {
            if let Some(base_key) = parse_lora_key(&key) {
                let entry = lora_keys.entry(base_key.clone()).or_insert((None, None, None));
                
                if key.ends_with(".lora_down.weight") {
                    entry.0 = Some(tensor);
                } else if key.ends_with(".lora_up.weight") {
                    entry.1 = Some(tensor);
                } else if key.ends_with(".alpha") {
                    // Alpha is a scalar
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
}

/// Parse LoRA key to get base layer name
///
/// Example: "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight"
/// Returns: "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k"
fn parse_lora_key(key: &str) -> Option<String> {
    // Remove "lora_" prefix
    let key = key.strip_prefix("lora_")?;
    
    // Remove ".lora_down.weight", ".lora_up.weight", or ".alpha" suffix
    let key = key
        .strip_suffix(".lora_down.weight")
        .or_else(|| key.strip_suffix(".lora_up.weight"))
        .or_else(|| key.strip_suffix(".alpha"))?;
    
    // Convert underscores to dots (except for numeric indices)
    let key = key.replace('_', ".");
    
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_lora_key() {
        let key = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight";
        let parsed = parse_lora_key(key).unwrap();
        assert_eq!(parsed, "unet.down.blocks.0.attentions.0.transformer.blocks.0.attn1.to.k");
    }
}
```

---

### Step 2: LoRA Application to UNet

**File:** `src/backend/lora.rs` (continued)

```rust
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;

/// Apply multiple LoRAs to a UNet model
///
/// # Arguments
/// * `unet` - UNet model to modify
/// * `loras` - List of LoRA weights to apply
/// * `strengths` - Strength multiplier for each LoRA (0.0-1.0)
///
/// # Returns
/// Modified UNet with LoRAs applied
///
/// # Note
/// This modifies the UNet in-place by adding LoRA deltas to existing weights
pub fn apply_loras_to_unet(
    unet: &mut UNet2DConditionModel,
    loras: &[LoRAWeights],
    strengths: &[f64],
) -> Result<()> {
    if loras.len() != strengths.len() {
        anyhow::bail!(
            "LoRA count ({}) doesn't match strength count ({})",
            loras.len(),
            strengths.len()
        );
    }
    
    tracing::info!("Applying {} LoRAs to UNet", loras.len());
    
    for (lora, &strength) in loras.iter().zip(strengths.iter()) {
        tracing::info!("  Applying LoRA '{}' with strength {}", lora.name, strength);
        apply_single_lora(unet, lora, strength)?;
    }
    
    Ok(())
}

/// Apply a single LoRA to UNet
fn apply_single_lora(
    unet: &mut UNet2DConditionModel,
    lora: &LoRAWeights,
    strength: f64,
) -> Result<()> {
    // Iterate over all LoRA tensors
    for (layer_name, lora_tensor) in &lora.weights {
        // Calculate LoRA delta: α * (A × B)
        let alpha = lora_tensor.alpha.unwrap_or(1.0) as f64;
        let scale = (alpha * strength) / lora_tensor.down.dim(0)? as f64;
        
        // Compute delta: down × up
        let delta = lora_tensor.up.matmul(&lora_tensor.down)?;
        let delta = (delta * scale)?;
        
        // Apply delta to corresponding UNet weight
        // This requires accessing UNet internals - may need to modify UNet struct
        apply_delta_to_layer(unet, layer_name, &delta)?;
    }
    
    Ok(())
}

/// Apply delta to a specific UNet layer
///
/// NOTE: This is the tricky part - Candle's UNet doesn't expose weights directly
/// We may need to:
/// 1. Fork candle-transformers to add weight access
/// 2. Use unsafe to access private fields
/// 3. Rebuild UNet with modified VarBuilder
fn apply_delta_to_layer(
    unet: &mut UNet2DConditionModel,
    layer_name: &str,
    delta: &Tensor,
) -> Result<()> {
    // TODO: This requires modifying candle-transformers
    // For now, return error
    anyhow::bail!(
        "LoRA application requires candle-transformers modification. \
        Layer: {}, delta shape: {:?}",
        layer_name,
        delta.shape()
    )
}
```

---

### Step 3: LoRA Configuration

**File:** `src/backend/sampling.rs`

```rust
/// LoRA configuration for generation
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    /// Path to LoRA file
    pub path: String,
    /// Strength (0.0-1.0)
    pub strength: f64,
}

/// Add to SamplingConfig
pub struct SamplingConfig {
    // ... existing fields ...
    
    /// LoRAs to apply (optional)
    pub loras: Vec<LoRAConfig>,
}
```

---

### Step 4: Integration with Generation

**File:** `src/backend/model_loader.rs`

```rust
use crate::backend::lora::{LoRAWeights, apply_loras_to_unet};

/// Load model components with optional LoRAs
pub fn load_model_components_with_loras(
    model_path: &str,
    version: SDVersion,
    device: &Device,
    use_f16: bool,
    lora_configs: &[LoRAConfig],
) -> Result<ModelComponents> {
    // 1. Load base model
    let mut components = load_model_components(model_path, version, device, use_f16)?;
    
    // 2. Load and apply LoRAs
    if !lora_configs.is_empty() {
        let loras: Vec<LoRAWeights> = lora_configs
            .iter()
            .map(|config| LoRAWeights::load(&config.path, device))
            .collect::<Result<Vec<_>>>()?;
        
        let strengths: Vec<f64> = lora_configs
            .iter()
            .map(|config| config.strength)
            .collect();
        
        apply_loras_to_unet(&mut components.unet, &loras, &strengths)?;
    }
    
    Ok(components)
}
```

---

## Major Challenge: Candle UNet Modification

**Problem:** Candle's `UNet2DConditionModel` doesn't expose weights for modification.

**Solutions:**

### Option 1: Fork candle-transformers (RECOMMENDED)
```rust
// In our fork of candle-transformers
impl UNet2DConditionModel {
    /// Get mutable access to a layer's weights
    pub fn get_layer_weights_mut(&mut self, layer_name: &str) -> Option<&mut Tensor> {
        // Expose internal weights
    }
    
    /// Apply delta to a layer
    pub fn apply_delta(&mut self, layer_name: &str, delta: &Tensor) -> Result<()> {
        if let Some(weight) = self.get_layer_weights_mut(layer_name) {
            *weight = (weight.clone() + delta)?;
            Ok(())
        } else {
            Err(anyhow!("Layer not found: {}", layer_name))
        }
    }
}
```

### Option 2: Rebuild UNet with Modified VarBuilder
```rust
// Create a new VarBuilder that includes LoRA deltas
let modified_vb = create_varbuilder_with_loras(base_vb, loras, strengths)?;
let unet = UNet2DConditionModel::new(&unet_config, modified_vb)?;
```

### Option 3: Use Unsafe (NOT RECOMMENDED)
```rust
// Access private fields using unsafe
// This is fragile and breaks with Candle updates
```

---

## Testing Plan

### Unit Tests

**File:** `src/backend/lora.rs`

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_lora_loading() {
        // Test loading a LoRA file
        // Verify correct number of tensors
        // Verify tensor shapes
    }
    
    #[test]
    fn test_lora_key_parsing() {
        // Test parsing various LoRA key formats
    }
    
    #[test]
    fn test_multiple_loras() {
        // Test loading multiple LoRAs
        // Verify they don't conflict
    }
}
```

### Integration Tests

**File:** `tests/lora_integration.rs`

```rust
#[tokio::test]
#[ignore]
async fn test_lora_application() {
    // Load base model
    // Load LoRA
    // Generate with LoRA
    // Verify output is different from base
}

#[tokio::test]
#[ignore]
async fn test_multiple_loras_stacking() {
    // Load base model
    // Load 2-3 LoRAs
    // Generate with all LoRAs
    // Verify combined effect
}

#[tokio::test]
#[ignore]
async fn test_lora_strength_variation() {
    // Test strength 0.0, 0.5, 1.0
    // Verify output changes proportionally
}
```

---

## Acceptance Criteria

- [ ] LoRA SafeTensors files can be loaded
- [ ] LoRA weights are correctly parsed
- [ ] LoRA deltas can be applied to UNet
- [ ] Multiple LoRAs can be stacked
- [ ] Strength parameter works (0.0-1.0)
- [ ] Generation with LoRAs produces expected results
- [ ] Memory usage stays reasonable
- [ ] Performance impact is acceptable (< 20% slowdown)
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Documentation complete

---

## Example Usage

```rust
// Load model with LoRAs
let lora_configs = vec![
    LoRAConfig {
        path: "~/.cache/rbee/loras/anime_style.safetensors".to_string(),
        strength: 0.8,
    },
    LoRAConfig {
        path: "~/.cache/rbee/loras/specific_character.safetensors".to_string(),
        strength: 0.6,
    },
];

let models = load_model_components_with_loras(
    model_path,
    SDVersion::V1_5,
    &device,
    false,
    &lora_configs,
)?;

// Generate with LoRAs applied
let config = SamplingConfig {
    prompt: "a portrait of a character".to_string(),
    loras: lora_configs,
    // ... other config ...
};

let image = generate_image(&config, &models, |_, _| {})?;
```

---

## Marketplace Impact

**Before LoRA Support:**
- ✅ Checkpoint models only
- ❌ NO LoRA (100K+ models unavailable)

**After LoRA Support:**
- ✅ Checkpoint models
- ✅ LoRA models (100K+ models now available!)
- 🎯 Marketplace compatibility dramatically improved

---

## References

- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models"
- **Automatic1111:** `extensions-builtin/Lora/` (reference implementation)
- **ComfyUI:** `comfy/sd.py` (LoRA loading)
- **Diffusers:** `diffusers.loaders.LoraLoaderMixin`

---

## Estimated Timeline

- **Day 1-2:** Implement LoRA loading and parsing
- **Day 3-4:** Fork candle-transformers, add weight access
- **Day 5:** Implement LoRA application
- **Day 6:** Integration and testing
- **Day 7:** Bug fixes and documentation

**Total:** 5-7 days
