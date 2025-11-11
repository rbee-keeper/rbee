# TEAM-476: HuggingFace Adapter Implementation

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Fetch from HuggingFace Hub API and normalize to MarketplaceModel

## What Was Implemented

### 1. Fetch Function: `fetchHuggingFaceModels()`

**Features:**
- ✅ Builds query string from parameters
- ✅ Handles all filter types (search, author, tags, etc.)
- ✅ Supports pipeline_tag, library, language, dataset, license filters
- ✅ Parses Link header for pagination
- ✅ Returns normalized `MarketplaceModel[]`
- ✅ Error handling with console logging

**API Endpoint:**
```
GET https://huggingface.co/api/models
```

**Query Parameters Supported:**
- `search` - Filter by substring
- `author` - Filter by author/organization
- `filter` - Filter by tags (string or array)
- `sort` - Sort by trending, downloads, likes, updated, created
- `direction` - Sort direction (-1 = desc, 1 = asc)
- `limit` - Max models to fetch
- `full` - Fetch full model data
- `config` - Fetch repo config
- `pipeline_tag` - Filter by task type
- `library` - Filter by library
- `language` - Filter by language
- `dataset` - Filter by training dataset
- `license` - Filter by license

### 2. Convert Function: `convertHFModel()`

**Normalization Logic:**

| Field | Source | Logic |
|-------|--------|-------|
| `id` | `model.id` | Direct mapping |
| `name` | `model.id.split('/')[1]` | Extract from ID |
| `author` | `model.author` or `model.id.split('/')[0]` | Fallback to ID |
| `downloads` | `model.downloads` | Direct mapping |
| `likes` | `model.likes` | Direct mapping |
| `tags` | `model.tags` | Direct mapping |
| `type` | `model.pipeline_tag` or `model.library_name` | Fallback chain |
| `nsfw` | `model.gated` | Conservative: gated = NSFW |
| `sizeBytes` | `model.safetensors.total` | From metadata |
| `createdAt` | `model.createdAt` | Parse to Date |
| `updatedAt` | `model.lastModified` | Parse to Date |
| `url` | `https://huggingface.co/${id}` | Construct URL |
| `license` | `model.cardData.license` | From card data |
| `metadata` | Various | HF-specific data |

### 3. Single Model Fetch: `fetchHuggingFaceModel()`

**Features:**
- ✅ Fetch single model by ID
- ✅ Same normalization as list
- ✅ Error handling

**API Endpoint:**
```
GET https://huggingface.co/api/models/{modelId}
```

## API Research Findings

### Pagination
- **Method:** Link header (GitHub-style)
- **Format:** `Link: <url>; rel="next"`
- **No total count** - HuggingFace doesn't provide it
- **No page numbers** - Uses cursor-based pagination

### NSFW Detection
- **No explicit NSFW flag** in HuggingFace API
- **Solution:** Use `gated` flag as proxy
- **Logic:** `gated === true || gated === 'manual' || gated === 'auto'`
- **Conservative approach** - Some gated models may not be NSFW

### Missing Data
- **No preview images** in list API (would need separate call)
- **No description** in list API (would need separate call)
- **No total count** for pagination

## Usage Examples

### Basic Fetch
```typescript
import { fetchHuggingFaceModels } from '@rbee/marketplace-core/adapters/huggingface'

const response = await fetchHuggingFaceModels({
  limit: 100,
  sort: 'downloads',
  direction: -1,
})

console.log(response.items) // MarketplaceModel[]
console.log(response.meta.hasNext) // boolean
```

### Filter by Task
```typescript
const response = await fetchHuggingFaceModels({
  pipeline_tag: 'text-generation',
  limit: 50,
})
```

### Filter by Author
```typescript
const response = await fetchHuggingFaceModels({
  author: 'meta-llama',
  limit: 50,
})
```

### Multiple Filters
```typescript
const response = await fetchHuggingFaceModels({
  filter: ['text-generation', 'transformers'],
  library: 'transformers',
  sort: 'trending',
  limit: 100,
})
```

### Single Model
```typescript
import { fetchHuggingFaceModel } from '@rbee/marketplace-core/adapters/huggingface'

const model = await fetchHuggingFaceModel('meta-llama/Llama-2-7b-hf')
console.log(model.name) // "Llama-2-7b-hf"
```

## Files Created

```
marketplace-core/src/adapters/huggingface/
├── index.ts          ✅ COMPLETE (183 lines)
├── types.ts          ✅ COMPLETE (from previous work)
└── README.md         ✅ COMPLETE (documentation)
```

## Contract Compliance

✅ **Returns `MarketplaceModel`** - Normalized format  
✅ **Returns `PaginatedResponse<T>`** - Standard pagination  
✅ **No raw HuggingFace types exposed** - Only contract types  
✅ **Error handling** - Console logging + throw  
✅ **Type safety** - Full TypeScript support  

## Next Steps

1. ✅ HuggingFace adapter complete
2. ⏭️ Implement CivitAI adapter (same pattern)
3. ⏭️ Add retry logic for failed requests
4. ⏭️ Add caching layer
5. ⏭️ Use adapters in Next.js marketplace app

## Testing

### Manual Test
```typescript
// Test in browser console or Node.js
import { fetchHuggingFaceModels } from '@rbee/marketplace-core/adapters/huggingface'

const response = await fetchHuggingFaceModels({ limit: 10 })
console.log('Fetched:', response.items.length, 'models')
console.log('First model:', response.items[0])
console.log('Has next:', response.meta.hasNext)
```

### Expected Output
```
[HuggingFace API] Fetching: https://huggingface.co/api/models?limit=10
[HuggingFace API] Fetched 10 models
Fetched: 10 models
First model: {
  id: "meta-llama/Llama-2-7b-hf",
  name: "Llama-2-7b-hf",
  author: "meta-llama",
  downloads: 1234567,
  likes: 5678,
  type: "text-generation",
  ...
}
Has next: true
```

## API Documentation

- **Official Docs:** https://huggingface.co/docs/hub/en/api
- **Endpoint:** `https://huggingface.co/api/models`
- **Rate Limits:** ~1000 requests/hour (estimated)
- **Auth:** Optional (not required for public models)

---

**TEAM-476 RULE ZERO:** This adapter implements the contract. Next.js imports `MarketplaceModel`, never raw `HuggingFaceModel`.
