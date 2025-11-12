# TEAM-484: Marketplace Compatibility Matrix for Workers

**Date:** 2025-11-12  
**Status:** ✅ COMPLETE  
**Source of Truth:** `frontend/packages/marketplace-core/src/adapters/gwc/types.ts`

## Problem

Workers need to declare which marketplace vendors (HuggingFace, CivitAI) and model types they support, so the UI can filter models appropriately.

**Requirements:**
1. Cover images (1:1 ratio preferred)
2. README URLs (raw markdown)
3. Marketplace compatibility matrix (which vendors + model types)

## Solution

### New Types Added

**1. MarketplaceVendor**
```typescript
export type MarketplaceVendor = 'huggingface' | 'civitai'
```

**2. MarketplaceCompatibility**
```typescript
export interface MarketplaceCompatibility {
  /** Which marketplace vendors this worker supports */
  vendors: MarketplaceVendor[]
  
  /** HuggingFace task filters (if vendor includes 'huggingface') */
  huggingface?: {
    /** Supported HF tasks (e.g., 'text-generation', 'text-to-image') */
    tasks?: string[]
    /** Supported HF libraries (e.g., 'transformers', 'diffusers') */
    libraries?: string[]
  }
  
  /** CivitAI model type filters (if vendor includes 'civitai') */
  civitai?: {
    /** Supported CivitAI model types (e.g., 'Checkpoint', 'LORA') */
    modelTypes?: string[]
    /** Supported CivitAI base models (e.g., 'SD 1.5', 'SDXL 1.0') */
    baseModels?: string[]
  }
}
```

**3. Enhanced GWCWorker**
```typescript
export interface GWCWorker {
  // ... existing fields ...
  
  /** Cover image URL (preferably 1:1 ratio) */
  coverImage?: string
  
  /** README URL (raw markdown) */
  readmeUrl?: string
  
  /** Which marketplace vendors and model types this worker supports */
  marketplaceCompatibility: MarketplaceCompatibility
}
```

## Worker Capabilities Analysis

### LLM Worker (`llm-worker-rbee`)

**Capabilities:**
- ✅ SafeTensors format (current)
- ⏳ GGUF format (aspirational - TEAM-409)
- ✅ Text generation
- ✅ Streaming support
- ✅ Max context: 32K tokens

**Marketplace Compatibility:**
```typescript
{
  vendors: ['huggingface'],
  huggingface: {
    tasks: ['text-generation', 'text2text-generation', 'conversational'],
    libraries: ['transformers'],
  },
}
```

**Why no CivitAI?**
- CivitAI focuses on image generation models (Stable Diffusion, LoRA, etc.)
- LLM worker is text-only
- No overlap in model types

**Cover Image:** `https://backend.rbee.dev/images/llm-worker-rbee.png`  
**README:** `https://raw.githubusercontent.com/rbee-keeper/rbee/refs/heads/development/bin/30_llm_worker_rbee/README.md`

---

### SD Worker (`sd-worker-rbee`)

**Capabilities:**
- ✅ SafeTensors format
- ✅ Text-to-image generation
- ✅ Image-to-image transformation
- ✅ Inpainting support
- ✅ Streaming progress via SSE
- ✅ Multiple SD versions (1.5, 2.1, XL, Turbo, SD3)

**Marketplace Compatibility:**
```typescript
{
  vendors: ['huggingface', 'civitai'],
  huggingface: {
    tasks: ['text-to-image'],
    libraries: ['diffusers'],
  },
  civitai: {
    modelTypes: ['Checkpoint', 'LORA', 'Controlnet'],
    baseModels: ['SD 1.5', 'SD 2.1', 'SDXL 1.0', 'SDXL Turbo', 'SD 3'],
  },
}
```

**Why both HuggingFace and CivitAI?**
- HuggingFace has official Stable Diffusion models (diffusers library)
- CivitAI has community-trained checkpoints, LoRAs, and Controlnets
- Both use SafeTensors format
- Both are compatible with Stable Diffusion architecture

**Cover Image:** `https://backend.rbee.dev/images/sd-worker-rbee.png`  
**README:** `https://github.com/rbee-keeper/rbee/blob/development/bin/31_sd_worker_rbee/README.md`

## Compatibility Matrix

| Worker | HuggingFace | CivitAI | Reason |
|--------|-------------|---------|--------|
| **LLM Worker** | ✅ | ❌ | Text generation only (HF has LLMs, CivitAI has image models) |
| **SD Worker** | ✅ | ✅ | Image generation (both HF and CivitAI have SD models) |

## Files Modified

### 1. `/frontend/packages/marketplace-core/src/adapters/gwc/types.ts`
**Added:**
- `MarketplaceVendor` type
- `MarketplaceCompatibility` interface
- `coverImage?: string` to `GWCWorker`
- `readmeUrl?: string` to `GWCWorker`
- `marketplaceCompatibility: MarketplaceCompatibility` to `GWCWorker`

### 2. `/frontend/packages/marketplace-core/src/index.ts`
**Exported:**
- `MarketplaceCompatibility`
- `MarketplaceVendor`

### 3. `/bin/80-global-worker-catalog/src/data.ts`
**Updated both workers:**
- Added `coverImage` URLs
- Added `readmeUrl` URLs
- Added `marketplaceCompatibility` objects

## Usage Example

```typescript
import type { GWCWorker } from '@rbee/marketplace-core'

// Filter models for a specific worker
function getCompatibleModels(worker: GWCWorker) {
  const { marketplaceCompatibility } = worker
  
  // Check if worker supports HuggingFace
  if (marketplaceCompatibility.vendors.includes('huggingface')) {
    const hfTasks = marketplaceCompatibility.huggingface?.tasks || []
    // Fetch models from HF with these tasks
  }
  
  // Check if worker supports CivitAI
  if (marketplaceCompatibility.vendors.includes('civitai')) {
    const modelTypes = marketplaceCompatibility.civitai?.modelTypes || []
    const baseModels = marketplaceCompatibility.civitai?.baseModels || []
    // Fetch models from CivitAI with these filters
  }
}
```

## Filter Mappings

### HuggingFace Task Filters

**LLM Worker supports:**
- `text-generation` - Standard LLM inference
- `text2text-generation` - Translation, summarization
- `conversational` - Chat models

**SD Worker supports:**
- `text-to-image` - Stable Diffusion models

**Reference:** `/frontend/packages/marketplace-core/src/adapters/huggingface/types.ts`

### CivitAI Model Type Filters

**SD Worker supports:**
- `Checkpoint` - Full SD models (e.g., SD 1.5, SDXL)
- `LORA` - Low-Rank Adaptation models
- `Controlnet` - Conditional control models

**SD Worker base models:**
- `SD 1.5` - Stable Diffusion 1.5
- `SD 2.1` - Stable Diffusion 2.1
- `SDXL 1.0` - Stable Diffusion XL
- `SDXL Turbo` - Fast SDXL variant
- `SD 3` - Stable Diffusion 3

**Reference:** `/frontend/packages/marketplace-core/src/adapters/civitai/types.ts`

## Verification

```bash
# Build marketplace-core
cd /home/vince/Projects/rbee/frontend/packages/marketplace-core
pnpm build
# ✅ Success

# Type check GWC
cd /home/vince/Projects/rbee/bin/80-global-worker-catalog
pnpm type-check
# ✅ Success
```

## Next Steps

1. ✅ UI can now filter models based on worker compatibility
2. ✅ Display cover images in worker catalog
3. ✅ Fetch and render README markdown
4. ✅ Filter HuggingFace models by task
5. ✅ Filter CivitAI models by type and base model

## Key Insights

**Why this matters:**
- **LLM Worker** only shows text generation models (no image models from CivitAI)
- **SD Worker** shows both HF diffusers and CivitAI checkpoints
- **Type safety** ensures UI can't request incompatible models
- **Extensible** for future workers (audio, video, etc.)

**Design Decision:**
> Workers declare what they support, not what they don't support. This makes adding new marketplace vendors easy (just add to the vendor list).

---

**Created by:** TEAM-484  
**Source of Truth:** `marketplace-core/src/adapters/gwc/types.ts`  
**Result:** Type-safe marketplace compatibility matrix for all workers
