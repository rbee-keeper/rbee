# MoE (Mixture of Experts) Abstraction Design

## Problem Statement

We need to support MoE models like Mixtral while maintaining our clean architecture. MoE models have:
- Multiple expert networks
- Routing mechanisms
- Different computational patterns than standard models

## Design Options

### Option 1: Separate Mixtral Model (RECOMMENDED)
**Pros:**
- Clean separation of concerns
- MoE-specific code isolated
- Easy to add more MoE models later
- Follows existing pattern (Llama, Mistral, Phi, etc.)

**Cons:**
- Slightly more code duplication

**Implementation:**
```
models/
├── mixtral/           # Mixtral MoE safetensors
│   ├── components.rs
│   ├── loader.rs
│   └── mod.rs
└── mixtral_quantized/ # Mixtral MoE GGUF
    ├── components.rs
    ├── loader.rs
    └── mod.rs
```

### Option 2: Enhance Mistral to Support MoE
**Pros:**
- Less code

**Cons:**
- Mixes two different architectures in one module
- Harder to maintain
- Violates single responsibility principle

## Recommended Approach: Option 1

Create **separate Mixtral modules** that:
1. Use candle's `mixtral` model (not `mistral`)
2. Follow the same pattern as other models
3. Use helper functions for DRY code
4. Document MoE-specific considerations

## Implementation Plan

### 1. Mixtral Safetensors
```rust
// components.rs
pub struct MixtralModel {
    model: candle_transformers::models::mixtral::Model,
    eos_token_id: u32,
    vocab_size: usize,
    num_experts: usize,        // MoE-specific
    experts_per_tok: usize,    // MoE-specific
    device: Device,
    capabilities: ModelCapabilities,
}
```

### 2. Mixtral GGUF (Quantized)
- Mixtral GGUF files currently use Llama quantized loader
- We already handle this via `QuantizedLlama` (Mistral GGUF does the same)
- **Decision:** Keep using `QuantizedLlama` for Mixtral GGUF for now
- **Future:** If Mixtral-specific GGUF format emerges, create `mixtral_quantized`

### 3. Architecture Detection
```rust
match architecture.as_str() {
    "mixtral" => {
        let model = mixtral::MixtralModel::load(path, device)?;
        Ok(Model::Mixtral(model))
    }
    // ...
}
```

### 4. MoE-Specific Capabilities
```rust
pub struct ModelCapabilities {
    // ... existing fields ...
    pub is_moe: bool,                    // NEW: Is this a MoE model?
    pub num_experts: Option<usize>,      // NEW: Number of experts
    pub experts_per_token: Option<usize>, // NEW: Experts activated per token
}

impl ModelCapabilities {
    pub fn moe(
        architecture: &'static str,
        max_position_embeddings: usize,
        num_experts: usize,
        experts_per_token: usize,
    ) -> Self {
        Self {
            uses_position: true,
            supports_cache_reset: false, // Mixtral doesn't have clear_kv_cache
            architecture,
            max_position_embeddings,
            is_quantized: false,
            is_moe: true,
            num_experts: Some(num_experts),
            experts_per_token: Some(experts_per_token),
        }
    }
}
```

## Future MoE Models

This pattern will work for:
- **DeepSeek-MoE** (when available)
- **Qwen-MoE** (when available)
- **Future MoE architectures**

Each will:
1. Have their own module (e.g., `deepseek_moe/`)
2. Use `ModelCapabilities::moe(...)`
3. Document expert count and routing strategy

## Migration Path

### Phase 1: Implement Mixtral (NOW)
- Create `mixtral/` module
- Use existing `QuantizedLlama` for Mixtral GGUF
- Add MoE fields to `ModelCapabilities`

### Phase 2: Future MoE Models
- Create `{model}_moe/` modules as needed
- Reuse MoE capabilities pattern
- Consider MoE-specific helper functions if patterns emerge

## Key Decisions

1. ✅ **Separate modules for MoE models** - Clean, maintainable
2. ✅ **Extend ModelCapabilities** - Track MoE metadata
3. ✅ **Reuse QuantizedLlama for GGUF** - Works today, change if needed
4. ✅ **Document expert configuration** - Important for users

## Notes

- Mixtral doesn't have `clear_kv_cache()` - set `supports_cache_reset: false`
- Config fields are `pub(crate)` - extract from JSON like DeepSeek
- MoE models are computationally different but API-compatible via `ModelTrait`

---

**Status:** ✅ COMPLETE - Mixtral MoE Implemented  
**Implemented by:** TEAM-483  
**Date:** 2025-11-12

## Implementation Summary

Mixtral MoE has been successfully added to the codebase following the design pattern:

### Files Created
- `src/backend/models/mixtral/components.rs` - MixtralModel struct with MoE metadata
- `src/backend/models/mixtral/loader.rs` - Safetensors loader with JSON config extraction
- `src/backend/models/mixtral/mod.rs` - Module exports and ModelTrait implementation

### Integration Points
- ✅ Added to `Model` enum in `models/mod.rs`
- ✅ Added to delegation macro for trait dispatch
- ✅ Added to `load_model()` function with "mixtral" architecture detection
- ✅ Added `MIXTRAL` constant to `arch` module
- ✅ Added to sealed trait implementations
- ✅ Handles private config fields via JSON extraction (like DeepSeek)

### Key Features
- Supports Mixtral-8x7B and other Mixtral MoE variants
- Tracks MoE-specific metadata (num_experts, experts_per_tok)
- Sets `supports_cache_reset: false` (Mixtral doesn't have clear_kv_cache)
- Uses candle-transformers' `mixtral` module
- EOS token ID: 2 (same as Mistral)

### Architecture Detection
Models with `"architectures": ["MixtralForCausalLM"]` in config.json will be detected as "mixtral"
