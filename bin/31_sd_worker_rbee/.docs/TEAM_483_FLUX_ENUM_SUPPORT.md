# TEAM-483: FLUX Support - Step 1 Complete

**Date:** 2025-11-12  
**Status:** ✅ ENUM SUPPORT COMPLETE  
**Next:** Implement FLUX model loader

---

## What Was Implemented

### ✅ Added FLUX to SDVersion Enum

**File:** `src/backend/models/mod.rs`

Added two new FLUX variants to the `SDVersion` enum:
- `FluxDev` - FLUX.1-dev (50 steps, guidance-distilled, best quality)
- `FluxSchnell` - FLUX.1-schnell (4 steps, fast, good quality)

### ✅ Updated All Match Arms

**1. Repository URLs:**
```rust
Self::FluxDev => "black-forest-labs/FLUX.1-dev",
Self::FluxSchnell => "black-forest-labs/FLUX.1-schnell",
```

**2. Default Image Sizes:**
```rust
Self::FluxDev | Self::FluxSchnell => (1024, 1024),
```

**3. Default Steps:**
```rust
Self::FluxSchnell => 4,  // Fast (4 steps)
Self::FluxDev => 50,     // Quality (50 steps)
```

**4. Default Guidance Scale:**
```rust
Self::FluxSchnell => 0.0,  // No guidance
Self::FluxDev => 3.5,      // Lower than SD (7.5)
```

**5. Helper Methods:**
```rust
pub fn is_flux(&self) -> bool {
    matches!(self, Self::FluxDev | Self::FluxSchnell)
}
```

**6. String Parsing:**
```rust
"flux-dev" | "flux.1-dev" | "flux1-dev" => Ok(Self::FluxDev),
"flux-schnell" | "flux.1-schnell" | "flux1-schnell" => Ok(Self::FluxSchnell),
```

**7. Config Methods:**
- `clip_config()` - Panics for FLUX (uses different encoders)
- `unet_config()` - Panics for FLUX (uses transformer, not UNet)
- `vae_config()` - Panics for FLUX (uses different VAE)
- `tokenizer_repo()` - Panics for FLUX (uses T5 + CLIP)

---

## Design Decisions

### Why Panic for Config Methods?

FLUX models use completely different architectures:
- **Text Encoding:** T5-XXL + CLIP (not just CLIP)
- **Denoising:** Transformer (not UNet)
- **VAE:** FLUX-specific autoencoder (not SD VAE)

The existing config methods are SD-specific, so they should panic if called on FLUX models. The FLUX loader will use different code paths entirely.

### String Parsing Variants

Supported all common naming conventions:
- `flux-dev` (kebab-case)
- `flux.1-dev` (official name)
- `flux1-dev` (simplified)

---

## What's Next

### Step 2: Create FLUX Model Loader

**File:** `src/backend/models/flux_loader.rs` (NEW)

Need to implement:
1. `FluxComponents` struct
2. Load T5-XXL text encoder
3. Load CLIP text encoder
4. Load FLUX transformer (full or quantized)
5. Load FLUX VAE

**Reference:**
- `/reference/candle/candle-examples/examples/flux/main.rs`
- `/reference/candle/candle-transformers/src/models/flux/`

### Step 3: Create FLUX Generation Function

**File:** `src/backend/flux_generation.rs` (NEW)

Need to implement:
1. Text encoding with T5 + CLIP
2. Noise initialization
3. Sampling state creation
4. Denoising loop
5. VAE decode

### Step 4: Integration

**Files:**
- `src/backend/model_loader.rs` - Add `LoadedModel::Flux` variant
- `src/backend/generation_engine.rs` - Route FLUX requests

---

## Testing

### Unit Tests

```rust
#[test]
fn test_flux_version_parsing() {
    assert_eq!(SDVersion::from_str("flux-dev").unwrap(), SDVersion::FluxDev);
    assert_eq!(SDVersion::from_str("flux-schnell").unwrap(), SDVersion::FluxSchnell);
}

#[test]
fn test_flux_defaults() {
    assert_eq!(SDVersion::FluxDev.default_size(), (1024, 1024));
    assert_eq!(SDVersion::FluxDev.default_steps(), 50);
    assert_eq!(SDVersion::FluxDev.default_guidance_scale(), 3.5);
    
    assert_eq!(SDVersion::FluxSchnell.default_steps(), 4);
    assert_eq!(SDVersion::FluxSchnell.default_guidance_scale(), 0.0);
}

#[test]
fn test_is_flux() {
    assert!(SDVersion::FluxDev.is_flux());
    assert!(SDVersion::FluxSchnell.is_flux());
    assert!(!SDVersion::XL.is_flux());
}
```

---

## Candle Reference

**FLUX is fully implemented in Candle:**
- ✅ Model: `/reference/candle/candle-transformers/src/models/flux/model.rs`
- ✅ Quantized: `/reference/candle/candle-transformers/src/models/flux/quantized_model.rs`
- ✅ VAE: `/reference/candle/candle-transformers/src/models/flux/autoencoder.rs`
- ✅ Sampling: `/reference/candle/candle-transformers/src/models/flux/sampling.rs`
- ✅ Example: `/reference/candle/candle-examples/examples/flux/main.rs`

**Just need to integrate it!**

---

## Estimated Timeline

- ✅ **Day 1:** Enum support (COMPLETE)
- **Day 2:** FLUX model loader
- **Day 3:** FLUX generation function
- **Day 4:** Integration with existing code
- **Day 5:** Testing and bug fixes
- **Day 6:** Documentation

**Total:** 4-6 days remaining

---

## Summary

✅ FLUX enum support is complete and compiles successfully.  
✅ All match arms updated with proper FLUX handling.  
✅ Helper methods added (`is_flux()`).  
✅ String parsing supports all common FLUX naming conventions.  
✅ Config methods properly panic for FLUX (different architecture).

**Ready for Step 2: Implement FLUX model loader!**
