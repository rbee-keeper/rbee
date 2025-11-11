# TEAM-476: Marketplace Adapters Complete

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Both HuggingFace and CivitAI adapters with LIST + DETAILS APIs

## ✅ Implementation Complete

### Architecture

```
Next.js Marketplace App
  ↓
  ├─ fetchHuggingFaceModels() → MarketplaceModel[]  (LIST)
  ├─ fetchHuggingFaceModel(id) → MarketplaceModel   (DETAILS)
  ├─ fetchCivitAIModels() → MarketplaceModel[]      (LIST)
  └─ fetchCivitAIModel(id) → MarketplaceModel       (DETAILS)
```

## HuggingFace Adapter ✅

### LIST API: `fetchHuggingFaceModels(params)`

**Endpoint:** `GET https://huggingface.co/api/models`

**Features:**
- ✅ All query parameters supported (search, author, filter, sort, etc.)
- ✅ Pipeline tag, library, language, dataset, license filters
- ✅ Link header pagination parsing
- ✅ Returns normalized `MarketplaceModel[]`

**Pagination:**
- Uses Link header (GitHub-style)
- No total count available
- Cursor-based pagination

### DETAILS API: `fetchHuggingFaceModel(modelId)`

**Endpoint:** `GET https://huggingface.co/api/models/{modelId}`

**Features:**
- ✅ Fetch single model by ID
- ✅ Same normalization as list
- ✅ Full model details

### Usage (Table View)

```typescript
import { fetchHuggingFaceModels } from '@rbee/marketplace-core/adapters/huggingface'

// LIST: For table view
const response = await fetchHuggingFaceModels({
  pipeline_tag: 'text-generation',
  sort: 'downloads',
  limit: 100,
})

// Render in table
<Table>
  {response.items.map(model => (
    <Row key={model.id}>
      <Cell>{model.name}</Cell>
      <Cell>{model.author}</Cell>
      <Cell>{model.downloads}</Cell>
    </Row>
  ))}
</Table>
```

## CivitAI Adapter ✅

### LIST API: `fetchCivitAIModels(params, apiKey?)`

**Endpoint:** `GET https://civitai.com/api/v1/models`

**Features:**
- ✅ All query parameters supported (types, sort, period, NSFW, etc.)
- ✅ API key support for NSFW content
- ✅ Cursor-based pagination
- ✅ Returns normalized `MarketplaceModel[]`

**Pagination:**
- Cursor-based (metadata.nextCursor)
- Total count available (metadata.totalItems)
- Page numbers supported

**NSFW Handling:**
- Optional API key parameter
- Environment variable support (`CIVITAI_API_KEY`)
- NSFW levels as bit flags (1,2,4,8,16)

### DETAILS API: `fetchCivitAIModel(modelId, apiKey?)`

**Endpoint:** `GET https://civitai.com/api/v1/models/{modelId}`

**Features:**
- ✅ Fetch single model by ID
- ✅ Same normalization as list
- ✅ Full model details with versions

### Usage (Card View)

```typescript
import { fetchCivitAIModels } from '@rbee/marketplace-core/adapters/civitai'

// LIST: For card grid
const response = await fetchCivitAIModels({
  types: ['Checkpoint', 'LORA'],
  sort: 'Most Downloaded',
  limit: 50,
  nsfwLevel: [1, 2, 4], // None, Soft, Mature
}, 'your-api-key-here')

// Render in card grid
<Grid>
  {response.items.map(model => (
    <ModelCard
      key={model.id}
      model={model}
      imageUrl={model.imageUrl}
      nsfw={model.nsfw}
    />
  ))}
</Grid>
```

## Unified Contract ✅

Both adapters return the same `MarketplaceModel` format:

```typescript
interface MarketplaceModel {
  id: string
  name: string
  description?: string
  author: string
  downloads: number
  likes: number
  tags: string[]
  type: string
  nsfw: boolean
  imageUrl?: string
  sizeBytes?: number
  createdAt: Date
  updatedAt: Date
  url: string
  license?: string
  metadata?: Record<string, unknown>
}
```

## View Recommendations

### HuggingFace → Table View

**Why:**
- No preview images in list API
- Focus on technical details (downloads, likes, type)
- Better for browsing by metrics

**Columns:**
- Model Name
- Author
- Type (pipeline_tag)
- Downloads
- Likes
- Actions

### CivitAI → Card View

**Why:**
- Rich preview images available
- Visual-first content (art, characters, etc.)
- Better for browsing by appearance

**Card Content:**
- Large preview image
- Model name
- Author
- Downloads/Likes
- NSFW badge
- Tags

## API Comparison

| Feature | HuggingFace | CivitAI |
|---------|-------------|---------|
| **LIST API** | ✅ | ✅ |
| **DETAILS API** | ✅ | ✅ |
| **Pagination** | Link header | Cursor + page |
| **Total count** | ❌ | ✅ |
| **Preview images** | ❌ | ✅ |
| **NSFW filter** | Gated flag | NSFW levels |
| **API key** | Optional | Required for NSFW |
| **Rate limits** | ~1000/hour | Unknown |

## Next Steps

1. ✅ HuggingFace adapter complete (LIST + DETAILS)
2. ✅ CivitAI adapter complete (LIST + DETAILS)
3. ⏭️ Use adapters in Next.js marketplace app
4. ⏭️ Create `/models/huggingface` page (table view)
5. ⏭️ Create `/models/civitai` page (card view)
6. ⏭️ Add client-side filtering
7. ⏭️ Add pagination controls

## Testing

### HuggingFace

```typescript
// List models
const hf = await fetchHuggingFaceModels({ limit: 10 })
console.log('HF models:', hf.items.length)

// Get model details
const model = await fetchHuggingFaceModel('meta-llama/Llama-2-7b-hf')
console.log('Model:', model.name)
```

### CivitAI

```typescript
// List models
const cai = await fetchCivitAIModels({ limit: 10 })
console.log('CivitAI models:', cai.items.length)

// Get model details
const model = await fetchCivitAIModel(3036)
console.log('Model:', model.name)
```

---

**TEAM-476 RULE ZERO:** Both adapters implement the same contract. Next.js imports `MarketplaceModel`, never raw vendor types. HuggingFace = table view, CivitAI = card view.
