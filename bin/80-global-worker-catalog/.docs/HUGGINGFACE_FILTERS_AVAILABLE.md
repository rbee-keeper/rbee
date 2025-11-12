# HuggingFace Model Hub Filters Available for SD Worker

**Date:** November 12, 2025  
**Source:** HuggingFace Hub API Documentation

---

## Current SD Worker Configuration

```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
}
```

---

## All Available HuggingFace Filters

### 1. **`author`** (string)
Filter by model uploader/organization
- **Examples:** `"stabilityai"`, `"runwayml"`, `"black-forest-labs"`, `"CompVis"`
- **Use Case:** Find all Stability AI or Black Forest Labs models

### 2. **`library`** (string or string[])
Filter by foundational library
- **Available:** `"diffusers"`, `"pytorch"`, `"tensorflow"`, `"jax"`, `"onnx"`
- **Current:** `["diffusers"]` ✅
- **Use Case:** Only show models compatible with diffusers library

### 3. **`task`** / **`pipeline_tag`** (string or string[])
Filter by model task/pipeline type
- **Available for Images:**
  - `"text-to-image"` ✅ (current)
  - `"image-to-image"`
  - `"image-to-text"`
  - `"unconditional-image-generation"`
  - `"image-classification"`
  - `"image-segmentation"`
- **Current:** `["text-to-image"]` ✅
- **Use Case:** Filter by specific generation task

### 4. **`tags`** (string or string[])
Filter by custom tags
- **Available for SD Worker:**
  - `"stable-diffusion"` - All SD models
  - `"stable-diffusion-xl"` - SDXL models only
  - `"flux"` - FLUX models
  - `"lora"` - LoRA models
  - `"controlnet"` - ControlNet models
  - `"inpainting"` - Inpainting models
  - `"anime"` - Anime-style models
  - `"photorealistic"` - Photorealistic models
  - `"safetensors"` - Models with safetensors format
- **Use Case:** Fine-grained filtering by model characteristics

### 5. **`model_name`** (string)
Filter by partial model name
- **Examples:** `"stable-diffusion"`, `"flux"`, `"sdxl"`
- **Use Case:** Search for specific model families

### 6. **`language`** (string or string[])
Filter by language (less relevant for image models)
- **Examples:** `"en"`, `"multilingual"`
- **Use Case:** Filter text encoders by language support

### 7. **`trained_dataset`** (string or string[])
Filter by training dataset
- **Examples:** `"laion"`, `"imagenet"`
- **Use Case:** Find models trained on specific datasets

---

## Recommended Configurations for SD Worker

### Option 1: **Broad Discovery** (Current)
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
}
```
**Finds:** All text-to-image models compatible with diffusers (~96,000+ models)

### Option 2: **Stable Diffusion Only**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['stable-diffusion'],
}
```
**Finds:** Only Stable Diffusion family models

### Option 3: **SDXL + FLUX Only**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['stable-diffusion-xl', 'flux'],
}
```
**Finds:** Only SDXL and FLUX models (most modern)

### Option 4: **Safetensors Only** (Recommended for Security)
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['safetensors'],
}
```
**Finds:** Only models with safetensors format (safer than pickle)

### Option 5: **Multi-Task Support**
```typescript
huggingface: {
  tasks: [
    'text-to-image',
    'image-to-image',
    'unconditional-image-generation',
  ],
  libraries: ['diffusers'],
}
```
**Finds:** All image generation tasks (txt2img, img2img, unconditional)

### Option 6: **Specific Authors**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  author: ['stabilityai', 'runwayml', 'black-forest-labs'],
}
```
**Finds:** Only official models from major providers

---

## Filter Combinations

### Example 1: **Production-Ready Models**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['safetensors', 'stable-diffusion-xl'],
  author: ['stabilityai'],
}
```
**Result:** Official SDXL models with safetensors format

### Example 2: **Modern Models Only**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['flux', 'stable-diffusion-xl'],
}
```
**Result:** FLUX and SDXL models (no SD 1.x/2.x)

### Example 3: **Anime/Art Models**
```typescript
huggingface: {
  tasks: ['text-to-image'],
  libraries: ['diffusers'],
  tags: ['anime', 'stable-diffusion-xl'],
}
```
**Result:** Anime-style SDXL models

---

## API Usage Example

```python
from huggingface_hub import HfApi, ModelFilter

api = HfApi()

# Current SD Worker filter
filter = ModelFilter(
    task="text-to-image",
    library="diffusers",
)

models = api.list_models(filter=filter)
for model in models:
    print(model.modelId)
```

---

## Recommendations for SD Worker Catalog

### Keep Current Configuration ✅
The current broad filter is good for discovery:
```typescript
tasks: ['text-to-image'],
libraries: ['diffusers'],
```

### Optional: Add Safetensors Tag
For security-conscious users:
```typescript
tasks: ['text-to-image'],
libraries: ['diffusers'],
tags: ['safetensors'],  // Only safe models
```

### Optional: Add Multi-Task Support
If we add img2img support later:
```typescript
tasks: [
  'text-to-image',
  'image-to-image',
],
libraries: ['diffusers'],
```

---

## Summary

**Available Filters:** 7 types (author, library, task, tags, model_name, language, trained_dataset)

**Most Useful for SD Worker:**
1. ✅ `tasks` - Filter by generation type
2. ✅ `libraries` - Ensure diffusers compatibility
3. ✅ `tags` - Fine-grained filtering (safetensors, sdxl, flux, anime, etc.)
4. ⚠️ `author` - Filter by official providers (optional)

**Current Config:** ✅ **GOOD** - Broad discovery, finds all compatible models

**Recommended Addition:** Consider adding `tags: ['safetensors']` for security
