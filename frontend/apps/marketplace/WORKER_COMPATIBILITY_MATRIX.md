# Worker Compatibility Matrix - CRITICAL FILTERING REQUIREMENT

**Date:** 2025-11-12  
**Status:** 🚨 MUST IMPLEMENT  
**Purpose:** Define model compatibility rules to prevent showing unusable models

---

## The Problem

**CURRENT STATE (BAD):**
- HuggingFace page shows ALL models (100K+ models)
- CivitAI page shows ALL models (50K+ models)
- Users see models they CANNOT use with our workers
- Terrible UX: "Why can't I download this model?"

**REQUIRED STATE (GOOD):**
- Only show models compatible with installed workers
- Filter by format, task, and architecture
- Clear compatibility indicators
- Never show unusable models

---

## Available Workers (MVP)

### 1. LLM Worker (`llm-worker-rbee`)

**Variants:**
- `llm-worker-rbee-cpu` (x86_64, aarch64)
- `llm-worker-rbee-cuda` (x86_64 + NVIDIA)
- `llm-worker-rbee-metal` (aarch64 + Apple Silicon)
- `llm-worker-rbee-rocm` (x86_64 + AMD)

**Supported Formats:**
- ✅ `safetensors` (IMPLEMENTED)
- ⏳ `gguf` (ASPIRATIONAL - TEAM-409)

**Supported Tasks (HuggingFace):**
- ✅ `text-generation` (Llama, Mistral, etc.)
- ❌ `text2text-generation` (NOT SUPPORTED)
- ❌ `fill-mask` (NOT SUPPORTED)
- ❌ `token-classification` (NOT SUPPORTED)
- ❌ ALL other tasks (NOT SUPPORTED)

**Supported Architectures:**
- ✅ Llama (all variants)
- ❌ GPT-2, GPT-J, BLOOM, etc. (NOT SUPPORTED)

**Max Context Length:** 32,768 tokens

---

### 2. SD Worker (`sd-worker-rbee`)

**Variants:**
- `sd-worker-rbee-cpu` (x86_64, aarch64)
- `sd-worker-rbee-cuda` (x86_64 + NVIDIA)
- `sd-worker-rbee-metal` (aarch64 + Apple Silicon)
- `sd-worker-rbee-rocm` (x86_64 + AMD)

**Supported Formats:**
- ✅ `safetensors` (IMPLEMENTED)

**Supported Tasks (HuggingFace):**
- ✅ `text-to-image` (Stable Diffusion)
- ✅ `image-to-image` (SD variants)
- ❌ `image-classification` (NOT SUPPORTED)
- ❌ `object-detection` (NOT SUPPORTED)
- ❌ ALL other tasks (NOT SUPPORTED)

**Supported Models (CivitAI):**
- ✅ `Checkpoint` (SD 1.5, SD 2.1, SDXL, SD3)
- ✅ `LORA` (LoRA adapters)
- ✅ `Controlnet` (ControlNet models)
- ❌ `TextualInversion` (NOT SUPPORTED)
- ❌ `Hypernetwork` (NOT SUPPORTED)
- ❌ `AestheticGradient` (NOT SUPPORTED)
- ❌ `Poses` (NOT SUPPORTED)

**Supported Base Models (CivitAI):**
- ✅ SD 1.4, SD 1.5, SD 2.0, SD 2.1 (all variants)
- ✅ SDXL 0.9, SDXL 1.0, SDXL Turbo, SDXL Distilled
- ✅ SD 3, SD 3.5
- ⏳ Flux.1 D, Flux.1 S (ASPIRATIONAL)
- ❌ Pony, Illustrious (NOT SUPPORTED)

---

## Filtering Rules

### HuggingFace Models

**MUST filter by:**
1. **Pipeline Tag** (task type)
   - LLM Worker: ONLY `text-generation`
   - SD Worker: ONLY `text-to-image`
   
2. **Library** (framework)
   - ONLY `transformers` OR `diffusers`
   - Exclude: pytorch, tensorflow, jax, etc.

3. **Model Architecture** (inferred from model card)
   - LLM Worker: ONLY Llama-based models
   - SD Worker: ONLY Stable Diffusion models

4. **Format** (implicit)
   - ONLY models with SafeTensors files
   - Exclude: GGUF (until TEAM-409 completes)

**Example Filter:**
```typescript
// HuggingFace LLM models
{
  pipeline_tag: 'text-generation',
  library: 'transformers',
  // Additional client-side filter: architecture === 'llama'
}

// HuggingFace SD models
{
  pipeline_tag: 'text-to-image',
  library: 'diffusers',
}
```

---

### CivitAI Models

**MUST filter by:**
1. **Model Type**
   - SD Worker: ONLY `Checkpoint`, `LORA`, `Controlnet`
   - Exclude: TextualInversion, Hypernetwork, AestheticGradient, Poses

2. **Base Model**
   - SD Worker: ONLY SD 1.x, SD 2.x, SDXL, SD 3
   - Exclude: Pony, Illustrious, Flux (until supported)

3. **Format** (implicit)
   - ONLY SafeTensors format
   - Exclude: PickleTensor (security risk)

**Example Filter:**
```typescript
// CivitAI SD models
{
  types: ['Checkpoint', 'LORA', 'Controlnet'],
  baseModels: [
    'SD 1.4', 'SD 1.5', 'SD 2.0', 'SD 2.1',
    'SDXL 0.9', 'SDXL 1.0', 'SDXL Turbo',
    'SD 3', 'SD 3.5'
  ],
}
```

---

## Implementation Plan

### Phase 1: Hard-Coded Filters (IMMEDIATE)

**HuggingFace:**
```typescript
// /apps/marketplace/app/models/huggingface/page.tsx
const filters: HuggingFaceListModelsParams = {
  pipeline_tag: 'text-generation', // ONLY text-gen for LLM worker
  library: 'transformers',          // ONLY transformers
  sort: 'downloads',
  limit: 50,
}
```

**CivitAI:**
```typescript
// /apps/marketplace/app/models/civitai/page.tsx
const filters: CivitAIListModelsParams = {
  types: ['Checkpoint', 'LORA', 'Controlnet'], // ONLY SD-compatible types
  baseModels: [
    'SD 1.4', 'SD 1.5', 'SD 2.0', 'SD 2.1',
    'SDXL 0.9', 'SDXL 1.0', 'SDXL Turbo',
    'SD 3', 'SD 3.5'
  ],
  sort: 'Most Downloaded',
  limit: 50,
}
```

---

### Phase 2: Worker Registry Integration (FUTURE)

**Goal:** Dynamically filter based on installed workers

```typescript
// Fetch installed workers from API
const installedWorkers = await fetch('/api/workers/installed')

// Build compatibility matrix
const compatibleFilters = buildFiltersFromWorkers(installedWorkers)

// Apply to vendor pages
<ModelPageContainer filters={compatibleFilters} />
```

**Worker Registry API:**
```typescript
interface InstalledWorker {
  id: string // 'llm-worker-rbee-cuda'
  supportedFormats: string[] // ['safetensors', 'gguf']
  supportedTasks: string[] // ['text-generation']
  supportedArchitectures: string[] // ['llama']
}
```

---

### Phase 3: UI Indicators (FUTURE)

**Compatibility badges:**
- ✅ "Compatible with LLM Worker"
- ✅ "Compatible with SD Worker"
- ⚠️ "Requires GGUF support (coming soon)"
- ❌ "Not compatible with installed workers"

**Filter UI:**
- Checkbox: "Show only compatible models" (default: ON)
- Dropdown: "Compatible with worker: [LLM Worker | SD Worker | All]"

---

## Critical Rules

### NEVER Show These Models

**HuggingFace:**
- ❌ GPT-2, GPT-J, BLOOM (wrong architecture)
- ❌ BERT, RoBERTa (wrong task)
- ❌ Audio models (ASR, TTS)
- ❌ Vision models (classification, detection)
- ❌ PyTorch-only models (no SafeTensors)

**CivitAI:**
- ❌ Pony, Illustrious (not supported)
- ❌ Flux.1 (not supported yet)
- ❌ TextualInversion, Hypernetwork (not supported)
- ❌ PickleTensor format (security risk)

---

## Testing Checklist

### HuggingFace Page
- [ ] Only shows `text-generation` models
- [ ] Only shows `transformers` library
- [ ] Only shows Llama-based models (client-side filter)
- [ ] No GPT-2, BERT, or other architectures
- [ ] No audio/vision models

### CivitAI Page
- [ ] Only shows Checkpoint, LORA, Controlnet
- [ ] Only shows SD 1.x, 2.x, SDXL, SD 3 base models
- [ ] No Pony, Illustrious, Flux models
- [ ] No TextualInversion, Hypernetwork

### User Experience
- [ ] All shown models are downloadable
- [ ] All shown models work with installed workers
- [ ] No "why can't I use this?" confusion
- [ ] Clear compatibility messaging

---

## Summary

**Current State:** 🚨 BROKEN - Shows 150K+ models, most unusable  
**Required State:** ✅ FIXED - Shows only compatible models (~5K models)

**Immediate Action:**
1. Add hard-coded filters to HuggingFace page (pipeline_tag, library)
2. Add hard-coded filters to CivitAI page (types, baseModels)
3. Test that only compatible models appear
4. Document compatibility rules in UI

**Future Work:**
- Dynamic filtering based on installed workers
- Compatibility badges on model cards
- "Show incompatible models" toggle (advanced users)

---

**RULE ZERO:** Never show models users cannot use. Compatibility filtering is MANDATORY, not optional.
