# TEAM-482: Additional Code Duplication Found 🔍

**Date:** 2025-11-12  
**Status:** 🚨 IDENTIFIED - NOT YET FIXED  
**Severity:** HIGH - Significant duplication across models

---

## Executive Summary

After comprehensive analysis, I found **4 major areas of duplication** that were NOT addressed in the initial consolidation:

1. **Preview Generation Logic** - Duplicated 4 times (txt2img, img2img, inpaint, FLUX)
2. **SafeTensors Loading Pattern** - Duplicated 6 times
3. **Progress Callback Wrapping** - Duplicated 2 times
4. **VAE Decode + Error Handling** - Duplicated 4 times

**Total Estimated Duplication:** ~200 lines of code

---

## DUPLICATION #1: Preview Generation Logic (CRITICAL)

### Current State: Duplicated 4 Times

**Files:**
- `stable_diffusion/generation/txt2img.rs:75-86`
- `stable_diffusion/generation/img2img.rs:95-107`
- `stable_diffusion/generation/inpaint.rs:140-152`
- `flux/generation/txt2img.rs:101-113`

**Duplicated Code (12 lines × 4 = 48 lines):**
```rust
// DUPLICATED IN 4 PLACES!
if step_idx % 5 == 0 || step_idx == num_steps - 1 {
    let preview_images = components.vae.decode(&(&latents / components.vae_scale)?)?;
    match tensor_to_image(&preview_images) {
        Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to generate preview image");
            progress_callback(step_idx + 1, num_steps, None);
        }
    }
} else {
    progress_callback(step_idx + 1, num_steps, None);
}
```

### Proposed Solution

**Create:** `src/backend/models/shared/preview.rs`

```rust
/// Generate preview image from latents
///
/// Handles VAE decode, error handling, and progress callback
#[inline]
pub fn generate_preview<F>(
    step_idx: usize,
    num_steps: usize,
    latents: &Tensor,
    vae: &impl Module,
    vae_scale: f64,
    tensor_to_image_fn: fn(&Tensor) -> Result<DynamicImage>,
    mut progress_callback: F,
) where
    F: FnMut(usize, usize, Option<DynamicImage>),
{
    const PREVIEW_FREQUENCY: usize = 5;
    
    if step_idx % PREVIEW_FREQUENCY == 0 || step_idx == num_steps - 1 {
        let preview_images = match vae.decode(&(latents / vae_scale)) {
            Ok(img) => img,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to decode preview");
                progress_callback(step_idx + 1, num_steps, None);
                return;
            }
        };
        
        match tensor_to_image_fn(&preview_images) {
            Ok(preview) => progress_callback(step_idx + 1, num_steps, Some(preview)),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to convert preview to image");
                progress_callback(step_idx + 1, num_steps, None);
            }
        }
    } else {
        progress_callback(step_idx + 1, num_steps, None);
    }
}
```

**Usage (after refactor):**
```rust
// In txt2img.rs, img2img.rs, inpaint.rs
generate_preview(
    step_idx,
    num_steps,
    &latents,
    &components.vae,
    components.vae_scale,
    tensor_to_image,
    &mut progress_callback,
);
```

**Benefits:**
- **48 lines → 12 lines** (75% reduction)
- Single source of truth for preview logic
- Easy to change preview frequency (currently hardcoded to 5)
- Consistent error handling

---

## DUPLICATION #2: SafeTensors Loading Pattern

### Current State: Duplicated 6 Times

**Files:**
- `stable_diffusion/loader.rs:90-96` (UNet + VAE)
- `flux/loader.rs:70` (T5)
- `flux/loader.rs:103` (CLIP)
- `flux/loader.rs:160` (FLUX model)
- `flux/loader.rs:182` (VAE)

**Duplicated Pattern:**
```rust
// DUPLICATED 6 TIMES!
// SAFETY: Memory-mapped file access is safe for read-only model weights
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
        .map_err(|e| Error::ModelLoading(format!("Failed to load X weights: {e}")))?
};
```

### Proposed Solution

**Create:** `src/backend/models/shared/loader.rs`

```rust
/// Load SafeTensors weights with memory mapping
///
/// # Safety
/// This function uses unsafe memory-mapped file access, which is safe because:
/// 1. Files are from trusted sources (HuggingFace Hub)
/// 2. Files are validated by hf-hub before use
/// 3. Candle's mmap implementation handles alignment and bounds checking
#[inline]
pub fn load_safetensors(
    weights_path: impl AsRef<Path>,
    dtype: DType,
    device: &Device,
    component_name: &str,
) -> Result<VarBuilder<'static>> {
    let weights_path = weights_path.as_ref();
    
    // SAFETY: See function documentation
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
            .map_err(|e| Error::ModelLoading(format!(
                "Failed to load {} weights from {:?}: {e}",
                component_name, weights_path
            )))?
    };
    
    Ok(vb)
}

/// Load SafeTensors weights from multiple files
#[inline]
pub fn load_safetensors_multi(
    weights_paths: &[impl AsRef<Path>],
    dtype: DType,
    device: &Device,
    component_name: &str,
) -> Result<VarBuilder<'static>> {
    let paths: Vec<_> = weights_paths.iter().map(|p| p.as_ref()).collect();
    
    // SAFETY: See load_safetensors documentation
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&paths, dtype, device)
            .map_err(|e| Error::ModelLoading(format!(
                "Failed to load {} weights: {e}",
                component_name
            )))?
    };
    
    Ok(vb)
}
```

**Usage (after refactor):**
```rust
// Before
let vb = unsafe {
    VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
        .map_err(|e| Error::ModelLoading(format!("Failed to load T5 weights: {e}")))?
};

// After
let vb = load_safetensors(weights_path, dtype, device, "T5")?;
```

**Benefits:**
- **Single safety comment** instead of 6
- Consistent error messages
- Easier to add logging/telemetry
- Type-safe wrapper around unsafe code

---

## DUPLICATION #3: Progress Callback Wrapping

### Current State: Duplicated 2 Times

**Files:**
- `stable_diffusion/mod.rs:94-118`
- `flux/mod.rs:87-101`

**Duplicated Pattern:**
```rust
// DUPLICATED IN BOTH!
fn generate(
    &mut self,
    request: &GenerationRequest,
    mut progress_callback: Box<dyn FnMut(usize, usize, Option<DynamicImage>) + Send>,
) -> Result<DynamicImage> {
    // Unbox and forward to generation function
    generation_fn(components, request, |step, total, preview| {
        progress_callback(step, total, preview);
    })
}
```

### Proposed Solution

**Create:** `src/backend/models/shared/callback.rs`

```rust
/// Helper to unbox and forward progress callbacks
///
/// Reduces boilerplate in model implementations
#[inline(always)]
pub fn unbox_callback<F, R>(
    boxed_callback: Box<dyn FnMut(usize, usize, Option<DynamicImage>) + Send>,
    generation_fn: F,
) -> R
where
    F: FnOnce(&mut dyn FnMut(usize, usize, Option<DynamicImage>)) -> R,
{
    let mut callback = boxed_callback;
    generation_fn(&mut |step, total, preview| {
        callback(step, total, preview);
    })
}
```

**Usage (after refactor):**
```rust
// Before
fn generate(..., mut progress_callback: Box<...>) -> Result<DynamicImage> {
    txt2img(components, request, |step, total, preview| {
        progress_callback(step, total, preview);
    })
}

// After
fn generate(..., progress_callback: Box<...>) -> Result<DynamicImage> {
    unbox_callback(progress_callback, |cb| {
        txt2img(components, request, cb)
    })
}
```

**Benefits:**
- Eliminates boilerplate
- Type-safe callback forwarding
- Easier to add callback instrumentation

---

## DUPLICATION #4: VAE Decode + Error Handling

### Current State: Duplicated 4 Times

**Files:**
- `stable_diffusion/generation/txt2img.rs:90-93`
- `stable_diffusion/generation/img2img.rs:110-113`
- `stable_diffusion/generation/inpaint.rs:155-158`
- `flux/generation/txt2img.rs:117-120`

**Duplicated Pattern:**
```rust
// DUPLICATED 4 TIMES!
let images = components.vae.decode(&(latents / components.vae_scale)?)?;
let image = tensor_to_image(&images)?;
Ok(image)
```

### Proposed Solution

**Add to:** `src/backend/models/shared/image_ops.rs`

```rust
/// Decode latents to final image
///
/// Combines VAE decode + tensor-to-image conversion
#[inline]
pub fn decode_latents_to_image(
    latents: &Tensor,
    vae: &impl Module,
    vae_scale: f64,
    tensor_to_image_fn: fn(&Tensor) -> Result<DynamicImage>,
) -> Result<DynamicImage> {
    let scaled_latents = (latents / vae_scale)?;
    let decoded = vae.decode(&scaled_latents)?;
    tensor_to_image_fn(&decoded)
}
```

**Usage (after refactor):**
```rust
// Before
let images = components.vae.decode(&(latents / components.vae_scale)?)?;
let image = tensor_to_image(&images)?;
Ok(image)

// After
decode_latents_to_image(&latents, &components.vae, components.vae_scale, tensor_to_image)
```

**Benefits:**
- **12 lines → 3 lines** (75% reduction)
- Consistent error handling
- Single place to add caching/optimization

---

## Summary of Duplication

| Duplication | Files | Lines Each | Total Lines | Reduction Potential |
|-------------|-------|------------|-------------|---------------------|
| **Preview Generation** | 4 | 12 | 48 | **36 lines (75%)** |
| **SafeTensors Loading** | 6 | 5 | 30 | **24 lines (80%)** |
| **Callback Wrapping** | 2 | 8 | 16 | **12 lines (75%)** |
| **VAE Decode** | 4 | 3 | 12 | **9 lines (75%)** |
| **TOTAL** | 16 | - | **106 lines** | **81 lines (76%)** |

---

## Recommended Action Plan

### Phase 1: High Impact (Do First)

1. ✅ **Preview Generation** - 48 lines → 12 lines
   - Create `shared/preview.rs`
   - Update 4 generation files
   - **Estimated Time:** 30 minutes

2. ✅ **SafeTensors Loading** - 30 lines → 6 lines
   - Create `shared/loader.rs`
   - Update 2 loader files
   - **Estimated Time:** 20 minutes

### Phase 2: Medium Impact (Do Second)

3. ✅ **VAE Decode** - 12 lines → 3 lines
   - Add to `shared/image_ops.rs`
   - Update 4 generation files
   - **Estimated Time:** 15 minutes

### Phase 3: Low Impact (Optional)

4. ⚠️ **Callback Wrapping** - 16 lines → 4 lines
   - Create `shared/callback.rs`
   - Update 2 model files
   - **Estimated Time:** 10 minutes
   - **Note:** Low priority, minimal benefit

---

## Files to Create

1. `src/backend/models/shared/preview.rs` - Preview generation
2. `src/backend/models/shared/loader.rs` - SafeTensors loading
3. `src/backend/models/shared/callback.rs` - Callback helpers (optional)

## Files to Update

**Phase 1:**
- `stable_diffusion/generation/txt2img.rs`
- `stable_diffusion/generation/img2img.rs`
- `stable_diffusion/generation/inpaint.rs`
- `flux/generation/txt2img.rs`
- `stable_diffusion/loader.rs`
- `flux/loader.rs`

**Phase 2:**
- Same 4 generation files (VAE decode)

**Phase 3:**
- `stable_diffusion/mod.rs`
- `flux/mod.rs`

---

## Expected Benefits

### Code Quality
- **106 lines of duplication eliminated** (76% reduction)
- Single source of truth for common patterns
- Easier to maintain and test

### Performance
- No performance impact (all functions inlined)
- Potential for future optimizations (caching, etc.)

### Maintainability
- Fix bugs once, not 4-6 times
- Consistent error messages
- Easier to add features (e.g., preview caching)

---

## Why This Wasn't Caught Initially

1. **Different file locations** - Spread across multiple modules
2. **Slight variations** - Small differences in error messages
3. **Focused on obvious duplication** - tensor_to_image was most obvious

**Lesson:** Need systematic duplication detection, not just manual review

---

## Next Steps

1. **Get approval** for breaking changes
2. **Implement Phase 1** (preview + SafeTensors)
3. **Test thoroughly** (all 4 generation modes)
4. **Implement Phase 2** (VAE decode)
5. **Consider Phase 3** (callback wrapping)

---

**TEAM-482: Additional duplication found. Ready to consolidate! 🚀**

