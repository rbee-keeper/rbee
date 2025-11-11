# TEAM-476: HuggingFace & CivitAI Filter Parity

**Date:** 2025-11-11  
**Status:** ✅ VERIFIED  
**Purpose:** Document complete filter and sorting parity for both vendors

## API Documentation Sources

- **HuggingFace:** https://huggingface.co/docs/hub/en/api
- **CivitAI:** https://developer.civitai.com/docs/api/public-rest

## Filter Comparison

### Common Filters (Both Vendors)

| Feature | HuggingFace | CivitAI | Notes |
|---------|-------------|---------|-------|
| **Search/Query** | `search` | `query` | Text search |
| **Author/Username** | `author` | `username` | Filter by creator |
| **Limit** | `limit` | `limit` | Results per page |
| **Page** | ❌ (Link header) | `page` | Page number |
| **Cursor** | ❌ | `cursor` | Cursor pagination |
| **Sort** | `sort` | `sort` | Sort order |
| **Tags** | `filter` | `tag` | Tag filtering |

### HuggingFace-Specific Filters

```typescript
interface HuggingFaceListModelsParams {
  // Basic
  search?: string                    // Text search
  author?: string                    // Filter by author/org
  limit?: number                     // Results limit
  
  // Sorting
  sort?: 'trending' | 'downloads' | 'likes' | 'updated' | 'created'
  direction?: -1 | 1                 // -1 = desc, 1 = asc
  
  // Filtering
  filter?: string | string[]         // Tag-based filtering
  pipeline_tag?: HuggingFaceTask     // Specific task (e.g., 'text-generation')
  library?: HuggingFaceLibrary       // Library (e.g., 'transformers', 'diffusers')
  language?: string                  // Language code (e.g., 'en', 'fr')
  dataset?: string                   // Trained on dataset
  license?: HuggingFaceLicense       // License type
  
  // Data fetching
  full?: boolean                     // Fetch complete model data
  config?: boolean                   // Fetch repo config
}
```

**HuggingFace Sort Options:**
- `trending` - Trending models
- `downloads` - Most downloaded
- `likes` - Most liked
- `updated` - Recently updated
- `created` - Recently created

**HuggingFace Tasks (pipeline_tag):**
- Text: `text-generation`, `text2text-generation`, `fill-mask`, `token-classification`, `question-answering`, `summarization`, `translation`, `text-classification`, `conversational`, `feature-extraction`, `sentence-similarity`, `zero-shot-classification`
- Image: `image-classification`, `image-segmentation`, `object-detection`, `image-to-text`, `text-to-image`
- Audio: `audio-classification`, `automatic-speech-recognition`, `text-to-speech`
- Other: `other`

**HuggingFace Libraries:**
- `transformers`, `pytorch`, `tensorflow`, `jax`, `rust`, `onnx`, `safetensors`, `diffusers`, `timm`, `spacy`, `sentence-transformers`, `sklearn`, `fastai`, `stable-baselines3`, `ml-agents`, `adapter-transformers`, `espnet`, `asteroid`, `pyannote-audio`, `fairseq`, `speechbrain`, `nemo`, `keras`, `fasttext`, `stanza`, `flair`, `allennlp`, `paddlenlp`, `span-marker`

### CivitAI-Specific Filters

```typescript
interface CivitAIListModelsParams {
  // Basic
  query?: string                     // Text search
  username?: string                  // Filter by creator
  tag?: string                       // Single tag filter
  limit?: number                     // Results limit
  page?: number                      // Page number
  cursor?: string                    // Cursor for pagination
  
  // Sorting
  sort?: 'Highest Rated' | 'Most Downloaded' | 'Newest'
  period?: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
  
  // Model Types
  types?: CivitAIModelType[]         // Array of model types
  
  // NSFW Filtering
  nsfwLevel?: CivitAINSFWLevel[]     // Bit flags: 1,2,4,8,16
  browsingLevel?: number             // NSFW browsing level
  supercedes_nsfw?: boolean          // Override NSFW
  
  // Base Models
  baseModels?: CivitAIBaseModel[]    // SD versions, SDXL, Flux, etc.
  
  // Licensing
  allowCommercialUse?: CivitAICommercialUse[]  // 'None', 'Image', 'Rent', 'Sell'
  allowDerivatives?: boolean         // Allow derivatives
  allowDifferentLicense?: boolean    // Allow different license
  allowNoCredit?: boolean            // Allow no credit
  
  // Other
  rating?: number                    // Rating filter (1-5)
  favorites?: boolean                // Show favorites
  hidden?: boolean                   // Show hidden
  primaryFileOnly?: boolean          // Primary file only
}
```

**CivitAI Sort Options:**
- `Highest Rated` - Top rated models
- `Most Downloaded` - Most downloads
- `Newest` - Recently added

**CivitAI Time Periods:**
- `AllTime` - All time
- `Year` - Past year
- `Month` - Past month
- `Week` - Past week
- `Day` - Past day

**CivitAI Model Types:**
- `Checkpoint` - Full model checkpoints
- `TextualInversion` - Textual inversion embeddings
- `Hypernetwork` - Hypernetworks
- `AestheticGradient` - Aesthetic gradients
- `LORA` - LoRA adapters
- `Controlnet` - ControlNet models
- `Poses` - Pose models

**CivitAI NSFW Levels (Bit Flags):**
- `1` - None
- `2` - Soft
- `4` - Mature
- `8` - X
- `16` - XXX

**CivitAI Base Models:**
- SD: `SD 1.4`, `SD 1.5`, `SD 2.0`, `SD 2.0 768`, `SD 2.1`, `SD 2.1 768`, `SD 2.1 Unclip`
- SDXL: `SDXL 0.9`, `SDXL 1.0`, `SDXL 1.0 LCM`, `SDXL Distilled`, `SDXL Turbo`
- SD 3: `SD 3`, `SD 3.5`
- Other: `Pony`, `Illustrious`, `Flux.1 D`, `Flux.1 S`, `Other`

**CivitAI Commercial Use:**
- `None` - No commercial use
- `Image` - Image generation only
- `Rent` - Can rent
- `Sell` - Can sell

## Pagination Comparison

### HuggingFace Pagination

```typescript
// Uses Link header (GitHub-style)
const response = await fetch('https://huggingface.co/api/models?limit=100')
const linkHeader = response.headers.get('Link')
// Link: <https://huggingface.co/api/models?limit=100&page=2>; rel="next"

// No total count available
// No cursor support
```

**Implementation:**
- Link header parsing
- No `total` count
- `hasNext` from Link header
- Page-based (implicit)

### CivitAI Pagination

```typescript
// Returns metadata with pagination info
{
  items: [...],
  metadata: {
    totalItems: 1234,
    currentPage: 1,
    pageSize: 100,
    totalPages: 13,
    nextPage: "https://civitai.com/api/v1/models?page=2",
    nextCursor: "abc123"
  }
}
```

**Implementation:**
- Metadata object
- `total` count available
- `hasNext` from metadata
- Both page and cursor support

## Usage Examples

### HuggingFace - Text Generation Models

```typescript
import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'

const params: HuggingFaceListModelsParams = {
  pipeline_tag: 'text-generation',
  library: 'transformers',
  sort: 'downloads',
  direction: -1,
  language: 'en',
  limit: 50,
}

const response = await huggingfaceAdapter.fetchModels(params)
```

### CivitAI - SDXL Checkpoints

```typescript
import type { CivitAIListModelsParams } from '@rbee/marketplace-core'

const params: CivitAIListModelsParams = {
  types: ['Checkpoint', 'LORA'],
  baseModels: ['SDXL 1.0', 'SDXL Turbo'],
  sort: 'Most Downloaded',
  period: 'Month',
  nsfwLevel: [1, 2, 4], // None, Soft, Mature
  allowCommercialUse: ['Image', 'Sell'],
  limit: 50,
}

const response = await civitaiAdapter.fetchModels(params)
```

## Type Safety

Both adapters provide full TypeScript type safety:

```typescript
// HuggingFace - Type-safe filters
<ModelListContainer<HuggingFaceListModelsParams>
  vendor="huggingface"
  filters={{
    sort: 'downloads', // ✅ Type-safe
    pipeline_tag: 'text-generation', // ✅ Type-safe
    library: 'transformers', // ✅ Type-safe
  }}
/>

// CivitAI - Type-safe filters
<ModelListContainer<CivitAIListModelsParams>
  vendor="civitai"
  filters={{
    sort: 'Most Downloaded', // ✅ Type-safe
    types: ['Checkpoint'], // ✅ Type-safe
    baseModels: ['SDXL 1.0'], // ✅ Type-safe
  }}
/>
```

## Summary

✅ **Complete Parity** - All vendor-specific filters supported  
✅ **Type Safe** - Full TypeScript types for all options  
✅ **No Heuristics** - Direct pass-through to vendor APIs  
✅ **Future Proof** - Easy to add new filters as vendors add them  
✅ **Well Documented** - Clear mapping between vendor APIs  

---

**TEAM-476 RULE ZERO:** Vendor-specific filters with full parity. No abstraction leaks. Type-safe. Future-proof.
