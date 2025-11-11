# HuggingFace Adapter

**TEAM-476:** Fetch from HuggingFace Hub API and normalize to `MarketplaceModel`

## API Documentation

- **Docs:** https://huggingface.co/docs/hub/en/api
- **Endpoint:** `https://huggingface.co/api/models`
- **Pagination:** Link header (GitHub-style)
- **Auth:** Optional (not required for public models)

## Functions

### `fetchHuggingFaceModels(params)`

Fetch models from HuggingFace Hub API with filtering and sorting.

**Parameters:**
```typescript
interface HuggingFaceListModelsParams {
  search?: string              // Filter by substring
  author?: string              // Filter by author/org
  filter?: string | string[]   // Filter by tags
  sort?: 'trending' | 'downloads' | 'likes' | 'updated' | 'created'
  direction?: -1 | 1           // -1 = descending, 1 = ascending
  limit?: number               // Max models to fetch
  full?: boolean               // Fetch full model data
  config?: boolean             // Fetch repo config
  pipeline_tag?: string        // Filter by task (e.g., "text-generation")
  library?: string             // Filter by library (e.g., "transformers")
  language?: string            // Filter by language
  dataset?: string             // Filter by training dataset
  license?: string             // Filter by license
}
```

**Returns:**
```typescript
Promise<PaginatedResponse<MarketplaceModel>>
```

**Example:**
```typescript
import { fetchHuggingFaceModels } from '@rbee/marketplace-core/adapters/huggingface'

// Fetch trending text-generation models
const response = await fetchHuggingFaceModels({
  pipeline_tag: 'text-generation',
  sort: 'trending',
  limit: 50,
  full: true,
})

console.log(response.items) // MarketplaceModel[]
console.log(response.meta.hasNext) // boolean
```

### `fetchHuggingFaceModel(modelId)`

Fetch a single model by ID.

**Parameters:**
- `modelId: string` - Model ID (e.g., "meta-llama/Llama-2-7b-hf")

**Returns:**
```typescript
Promise<MarketplaceModel>
```

**Example:**
```typescript
const model = await fetchHuggingFaceModel('meta-llama/Llama-2-7b-hf')
console.log(model.name) // "Llama-2-7b-hf"
console.log(model.author) // "meta-llama"
```

### `convertHFModel(model)`

Convert raw HuggingFace model to normalized `MarketplaceModel`.

**Parameters:**
- `model: HuggingFaceModel` - Raw model from API

**Returns:**
```typescript
MarketplaceModel
```

## Field Mapping

| MarketplaceModel | HuggingFaceModel | Notes |
|------------------|------------------|-------|
| `id` | `id` | Full model ID (e.g., "meta-llama/Llama-2-7b-hf") |
| `name` | `id.split('/')[1]` | Extracted from ID |
| `author` | `author` or `id.split('/')[0]` | Organization/user |
| `description` | N/A | Not available in list API |
| `downloads` | `downloads` | Download count |
| `likes` | `likes` | Like count |
| `tags` | `tags` | Array of tags |
| `type` | `pipeline_tag` or `library_name` | Task type or library |
| `nsfw` | `gated` | Conservative: gated = NSFW |
| `imageUrl` | N/A | Not available in list API |
| `sizeBytes` | `safetensors.total` | Total model size |
| `createdAt` | `createdAt` | Creation date |
| `updatedAt` | `lastModified` | Last modified date |
| `url` | `https://huggingface.co/${id}` | Model page URL |
| `license` | `cardData.license` | License type |
| `metadata` | Various | HF-specific data |

## Pagination

HuggingFace uses **Link header pagination** (GitHub-style):

```
Link: <https://huggingface.co/api/models?limit=100&cursor=abc>; rel="next"
```

The adapter parses this header to determine `hasNext`.

**Note:** HuggingFace does NOT provide:
- Total count
- Page numbers
- Previous page links (in most cases)

## Filtering Examples

### By Task (Pipeline Tag)
```typescript
await fetchHuggingFaceModels({
  pipeline_tag: 'text-generation',
  limit: 100,
})
```

### By Library
```typescript
await fetchHuggingFaceModels({
  library: 'transformers',
  limit: 100,
})
```

### By License
```typescript
await fetchHuggingFaceModels({
  license: 'apache-2.0',
  limit: 100,
})
```

### By Author
```typescript
await fetchHuggingFaceModels({
  author: 'meta-llama',
  limit: 100,
})
```

### Multiple Filters
```typescript
await fetchHuggingFaceModels({
  filter: ['text-generation', 'transformers', 'en'],
  sort: 'downloads',
  direction: -1,
  limit: 50,
})
```

## NSFW Detection

HuggingFace doesn't have explicit NSFW flags. The adapter uses **gated models** as a proxy:

```typescript
const nsfw = model.gated === true || model.gated === 'manual' || model.gated === 'auto'
```

**Gated models** require:
- Manual approval
- License agreement
- Authentication

This is a **conservative approach** - some gated models may not be NSFW, but it's safer to flag them.

## Metadata

The adapter stores HuggingFace-specific data in `metadata`:

```typescript
{
  sha: string              // Git commit SHA
  private: boolean         // Private model flag
  gated: boolean | string  // Gated access flag
  disabled: boolean        // Disabled flag
  library_name: string     // Library (e.g., "transformers")
  pipeline_tag: string     // Task type
  trendingScore: number    // Trending score
  transformersInfo: object // Transformers-specific info
}
```

## Error Handling

```typescript
try {
  const response = await fetchHuggingFaceModels({ limit: 100 })
} catch (error) {
  console.error('HuggingFace API error:', error)
  // Handle error (network, API error, etc.)
}
```

## Rate Limits

HuggingFace has **generous rate limits** for public API:
- No authentication required for public models
- ~1000 requests/hour (estimated)
- Use `limit` parameter to reduce requests

## Next Steps

1. ✅ Fetch implementation complete
2. ⏭️ Add retry logic for failed requests
3. ⏭️ Add caching layer
4. ⏭️ Add Link header parsing for next page URL
5. ⏭️ Add filtering for GGUF-compatible models

---

**TEAM-476 RULE ZERO:** This adapter returns `MarketplaceModel`. Next.js only imports the contract, never raw HuggingFace types.
