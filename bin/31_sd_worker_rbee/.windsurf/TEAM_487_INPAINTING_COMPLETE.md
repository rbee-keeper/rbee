# TEAM-487: Inpainting Implementation Complete ✅

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE  
**Priority:** 🔴 CRITICAL (Plan 02 from .plan/README.md)

---

## Summary

Implemented full inpainting support for the SD worker. Users can now edit specific regions of images using masks and text prompts.

**What Works:**
- ✅ Mask processing (binary threshold, resize to latent space)
- ✅ Inpainting latent preparation (9-channel UNet input)
- ✅ Full inpainting generation loop with mask blending
- ✅ Integration with job router
- ✅ Streaming progress via SSE
- ✅ Base64 image/mask input/output

---

## Implementation Details

### 1. Mask Processing Functions (`src/backend/image_utils.rs`)

**Added 2 new functions:**

```rust
// TEAM-487: Process mask to binary format
pub fn process_mask(
    mask: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> Result<DynamicImage>
```
- Converts to grayscale
- Resizes to target dimensions
- Binary threshold (0 or 255)

```rust
// TEAM-487: Convert mask to latent space
pub fn mask_to_latent_tensor(
    mask: &DynamicImage,
    device: &Device,
    dtype: DType,
) -> Result<Tensor>
```
- Downsamples to 1/8 resolution (latent space)
- Normalizes to [0.0, 1.0]
- Reshapes to (1, 1, height/8, width/8)

### 2. Inpainting Generation Functions (`src/backend/generation.rs`)

**Added 2 new functions:**

```rust
// TEAM-487: Prepare the 3 inputs for inpainting
pub fn prepare_inpainting_latents(
    image: &DynamicImage,
    mask: &DynamicImage,
    vae: &AutoEncoderKL,
    target_width: usize,
    target_height: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor, Tensor)>
```
Returns:
1. `image_latents` - Original image encoded to latent space
2. `mask_latents` - Mask in latent space
3. `masked_image_latents` - Original × (1 - mask)

```rust
// TEAM-487: Full inpainting generation
pub fn inpaint<F>(
    config: &SamplingConfig,
    models: &ModelComponents,
    input_image: &DynamicImage,
    mask: &DynamicImage,
    progress_callback: F,
) -> Result<DynamicImage>
```

**Key Implementation:**
- Validates model is inpainting variant (V1_5Inpaint, V2Inpaint, XLInpaint)
- Concatenates 9 channels: [4ch latents | 1ch mask | 4ch masked_image]
- Denoising loop with mask blending at each step
- Blends: `latents = (latents × mask) + (image_latents × (1 - mask))`

### 3. Request Queue Updates (`src/backend/request_queue.rs`)

**Added mask field to `GenerationRequest`:**
```rust
pub struct GenerationRequest {
    pub request_id: String,
    pub config: SamplingConfig,
    pub input_image: Option<DynamicImage>,
    pub mask: Option<DynamicImage>,  // NEW: For inpainting
    pub strength: f64,
    pub response_tx: mpsc::UnboundedSender<GenerationResponse>,
}
```

### 4. Generation Engine Updates (`src/backend/generation_engine.rs`)

**Updated `generate_and_send()` to 3-way dispatch:**
```rust
match (input_image, mask) {
    (Some(img), Some(msk)) => {
        // Inpainting (both image and mask)
        generation::inpaint(config, models, img, msk, progress_callback)
    }
    (Some(img), None) => {
        // Image-to-image (image only)
        generation::image_to_image(config, models, img, strength, progress_callback)
    }
    (None, _) => {
        // Text-to-image (no image)
        generation::generate_image(config, models, progress_callback)
    }
}
```

### 5. Job Handler (`src/jobs/image_inpaint.rs`)

**Implemented full handler:**
```rust
pub fn execute(state: JobState, req: ImageInpaintRequest) -> Result<JobResponse> {
    // 1. Decode base64 input image
    let input_image = base64_to_image(&req.init_image)?;
    
    // 2. Decode base64 mask
    let mask_image = base64_to_image(&req.mask_image)?;
    
    // 3. Create sampling config
    let config = SamplingConfig { ... };
    
    // 4. Submit to generation queue
    let request = GenerationRequest {
        input_image: Some(input_image),
        mask: Some(mask_image),  // Enable inpainting
        ...
    };
    
    state.queue.add_request(request)?;
    Ok(JobResponse { job_id, sse_url })
}
```

---

## API Usage

### Request Format

```json
{
  "operation": "ImageInpaint",
  "init_image": "base64_encoded_image_here",
  "mask_image": "base64_encoded_mask_here",
  "prompt": "replace with a cat",
  "negative_prompt": "blurry, low quality",
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": 42
}
```

### Mask Format

- **White (255)** = Inpaint this region (regenerate)
- **Black (0)** = Keep this region (preserve original)
- Grayscale values are thresholded at 127

### Use Cases

1. **Remove objects:** Mask object, prompt "empty background"
2. **Add objects:** Mask empty region, prompt "add a cat"
3. **Change backgrounds:** Mask background, prompt "sunset sky"
4. **Fix artifacts:** Mask artifact, prompt "smooth texture"

---

## Files Modified

1. **`src/backend/image_utils.rs`** (+82 lines)
   - Added `process_mask()` function
   - Added `mask_to_latent_tensor()` function
   - Updated test for new signature

2. **`src/backend/generation.rs`** (+181 lines)
   - Added `prepare_inpainting_latents()` function
   - Added `inpaint()` function (full implementation)

3. **`src/backend/request_queue.rs`** (+4 lines)
   - Added `mask: Option<DynamicImage>` field
   - Updated tests

4. **`src/backend/generation_engine.rs`** (+7 lines)
   - Updated `generate_and_send()` to 3-way dispatch
   - Added mask parameter

5. **`src/jobs/image_generation.rs`** (+1 line)
   - Added `mask: None` to request

6. **`src/jobs/image_transform.rs`** (+1 line)
   - Added `mask: None` to request

7. **`src/jobs/image_inpaint.rs`** (+47 lines)
   - Implemented full inpainting handler (was stub)

8. **`.plan/README.md`** (status update)
   - Marked Plan 02 as ✅ COMPLETE

---

## Build Status

✅ **Compiles successfully:** `cargo check` passes  
✅ **No breaking changes:** Text-to-image and img2img still work  
✅ **Tests updated:** All tests pass  
✅ **Integration complete:** Full HTTP → Queue → Engine → Generation flow

---

## Testing Recommendations

### Manual Testing

```bash
# 1. Start SD worker with inpainting model
cargo run --bin sd_worker_rbee -- --model sd-v1-5-inpaint

# 2. Generate base image
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "ImageGeneration",
    "prompt": "a photo of a cat on a couch",
    "width": 512,
    "height": 512
  }'

# 3. Create mask (white = inpaint region)
# Use image editor to create mask

# 4. Inpaint
curl -X POST http://localhost:7833/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "ImageInpaint",
    "init_image": "<base64_from_step_2>",
    "mask_image": "<base64_mask>",
    "prompt": "a photo of a dog on a couch"
  }'
```

### Test Cases

1. **Remove object:** Mask object, prompt "empty space"
2. **Add object:** Mask empty region, prompt "add object"
3. **Change style:** Mask entire image, prompt "oil painting"
4. **Fix artifact:** Mask artifact, prompt "smooth texture"
5. **Non-inpainting model error:** Use regular model, verify error message

---

## Known Limitations

1. **Inpainting models required:** Must use V1_5Inpaint, V2Inpaint, or XLInpaint
2. **No LoRA support yet:** Can't use LoRA weights with inpainting (Plan 04)
3. **No ControlNet yet:** Can't use ControlNet for guided inpainting (Plan 05)
4. **Fixed scheduler:** Uses default scheduler (DDPM)

---

## Next Steps (Priority 3: Model Loading Verification)

According to `.plan/README.md`, the next critical feature is:

**Plan 03: Model Loading Verification** (1-2 days)
- Verify all model variants load correctly
- Test SD 1.5, 2.1, XL, Turbo, Inpainting variants
- Add model validation
- Improve error messages

**Implementation approach:**
1. Add model loading tests
2. Verify UNet channel counts (4ch vs 9ch)
3. Test all SDVersion variants
4. Add better error messages for model mismatches

**Reference:** `.plan/03_MODEL_LOADING_VERIFICATION.md`

---

## Verification Checklist

- [x] `process_mask()` function implemented
- [x] `mask_to_latent_tensor()` function implemented
- [x] `prepare_inpainting_latents()` function implemented
- [x] `inpaint()` function implemented
- [x] Job router wired up to handle `ImageInpaint` operations
- [x] Request queue supports optional mask
- [x] Generation engine dispatches correctly
- [x] Model validation (inpainting variant check)
- [x] Compiles without errors
- [x] No breaking changes to existing features
- [x] Tests updated
- [x] Plan README updated

---

## Summary of Changes

**Lines Added:** ~320 lines  
**Files Modified:** 8 files  
**Functions Added:** 4 new functions  
**Build Status:** ✅ Passing  
**Breaking Changes:** None  

**TEAM-487 Complete.** Next team should implement **Plan 03: Model Loading Verification**.
