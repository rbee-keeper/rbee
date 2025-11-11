# TEAM-476: Marketplace Packages Architecture

**Date:** 2025-11-11  
**Status:** ✅ Types Complete, ⏭️ Adapters Next  
**Purpose:** Create 3 packages for marketplace with proper separation of concerns

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Next.js Marketplace App (port 7823)             │
│                                                              │
│  Imports: MarketplaceModel from @rbee/marketplace-contracts │
│  Uses: civitai-adapter, huggingface-adapter                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
┌──────▼─────────┐             ┌──────▼─────────┐
│ civitai-adapter│             │ huggingface-   │
│                │             │ adapter        │
│ Fetches from:  │             │ Fetches from:  │
│ civitai.com    │             │ huggingface.co │
│                │             │                │
│ Converts:      │             │ Converts:      │
│ CivitAIModel → │             │ HFModel →      │
│ MarketplaceModel             │ MarketplaceModel
└────────┬───────┘             └────────┬───────┘
         │                              │
         │   Both depend on contracts   │
         └──────────────┬───────────────┘
                        │
            ┌───────────▼────────────┐
            │ marketplace-contracts  │
            │                        │
            │ Types:                 │
            │ - MarketplaceModel     │
            │ - CivitAIModel         │
            │ - HuggingFaceModel     │
            │ - PaginatedResponse    │
            └────────────────────────┘
```

## Package Structure

### 1. marketplace-contracts ✅ COMPLETE

**Purpose:** Type definitions (contract between adapters and Next.js)

**Location:** `/frontend/packages/marketplace-contracts`

**Exports:**
- `MarketplaceModel` - Common interface Next.js expects
- `CivitAIModel` - Full CivitAI API types
- `HuggingFaceModel` - Full HuggingFace API types
- `PaginatedResponse<T>` - Pagination wrapper
- All enums and helper types

**Dependencies:** None (pure types)

**Files Created:**
```
marketplace-contracts/
├── package.json
├── tsconfig.json
├── README.md
└── src/
    ├── index.ts          (exports all types)
    ├── common.ts         (MarketplaceModel, PaginatedResponse)
    ├── civitai.ts        (CivitAI API types)
    └── huggingface.ts    (HuggingFace API types)
```

### 2. civitai-adapter ⏭️ TODO

**Purpose:** Fetch from CivitAI API and convert to MarketplaceModel

**Location:** `/frontend/packages/civitai-adapter`

**Exports:**
```typescript
export async function fetchCivitAIModels(
  params: CivitAIListModelsParams
): Promise<PaginatedResponse<MarketplaceModel>>

export function convertCivitAIModel(
  model: CivitAIModel
): MarketplaceModel
```

**Dependencies:**
- `@rbee/marketplace-contracts` (types)

**Responsibilities:**
- Fetch from `https://civitai.com/api/v1/models`
- Handle pagination
- Convert `CivitAIModel` → `MarketplaceModel`
- Handle errors
- Optional: Add API key support for NSFW content

### 3. huggingface-adapter ⏭️ TODO

**Purpose:** Fetch from HuggingFace API and convert to MarketplaceModel

**Location:** `/frontend/packages/huggingface-adapter`

**Exports:**
```typescript
export async function fetchHuggingFaceModels(
  params: HuggingFaceListModelsParams
): Promise<PaginatedResponse<MarketplaceModel>>

export function convertHFModel(
  model: HuggingFaceModel
): MarketplaceModel
```

**Dependencies:**
- `@rbee/marketplace-contracts` (types)

**Responsibilities:**
- Fetch from `https://huggingface.co/api/models`
- Handle pagination (Link header)
- Convert `HuggingFaceModel` → `MarketplaceModel`
- Handle errors
- Filter for GGUF-compatible models

## Type Contract

### MarketplaceModel (What Next.js Expects)

```typescript
interface MarketplaceModel {
  id: string              // Unique identifier
  name: string            // Model name
  description?: string    // Short description
  author: string          // Creator/organization
  downloads: number       // Download count
  likes: number           // Like/favorite count
  tags: string[]          // Tags/categories
  type: string            // Model type
  nsfw: boolean           // NSFW flag
  imageUrl?: string       // Primary image
  sizeBytes?: number      // Model size
  createdAt: Date         // Created date
  updatedAt: Date         // Updated date
  url: string             // External URL
  license?: string        // License
  metadata?: Record<string, unknown>  // Adapter-specific data
}
```

## API Documentation

### CivitAI API v1
- **Docs:** https://github.com/civitai/civitai/wiki/REST-API-Reference
- **Endpoint:** `https://civitai.com/api/v1/models`
- **Auth:** Optional (required for NSFW content)
- **Pagination:** Cursor-based
- **Rate Limits:** Unknown

**Key Fields:**
- `id` (number)
- `name` (string)
- `type` (Checkpoint, LORA, etc.)
- `nsfw` (boolean)
- `stats.downloadCount`, `stats.favoriteCount`
- `creator.username`
- `modelVersions[0].images[0].url`

### HuggingFace Hub API
- **Docs:** https://huggingface.co/docs/hub/en/api
- **Endpoint:** `https://huggingface.co/api/models`
- **Auth:** Optional
- **Pagination:** Link header
- **Rate Limits:** Generous

**Key Fields:**
- `id` (string, e.g., "meta-llama/Llama-2-7b-hf")
- `author` (string)
- `downloads` (number)
- `likes` (number)
- `tags` (string[])
- `pipeline_tag` (task type)
- `library_name` (transformers, etc.)

## Usage Example

### In Next.js Marketplace App

```typescript
import type { MarketplaceModel } from '@rbee/marketplace-contracts'
import { fetchCivitAIModels } from '@rbee/civitai-adapter'
import { fetchHuggingFaceModels } from '@rbee/huggingface-adapter'

// Server Component
export default async function ModelsPage() {
  // Both return the same MarketplaceModel[] format
  const civitaiModels = await fetchCivitAIModels({ 
    limit: 100, 
    types: ['Checkpoint', 'LORA'] 
  })
  
  const hfModels = await fetchHuggingFaceModels({ 
    limit: 100, 
    filter: 'text-generation' 
  })

  return (
    <div>
      <ModelGrid models={civitaiModels.items} />
      <ModelGrid models={hfModels.items} />
    </div>
  )
}

// Client Component
function ModelGrid({ models }: { models: MarketplaceModel[] }) {
  return (
    <div className="grid grid-cols-3 gap-6">
      {models.map(model => (
        <ModelCard key={model.id} model={model} />
      ))}
    </div>
  )
}
```

## Benefits

✅ **Separation of Concerns** - Each package has one job  
✅ **Type Safety** - Compile-time checks across packages  
✅ **Testability** - Mock adapters easily  
✅ **Maintainability** - Change adapter without touching Next.js  
✅ **Extensibility** - Add new adapters (Ollama, Replicate, etc.)  
✅ **Reusability** - Use adapters in Keeper UI too  

## Next Steps

1. ✅ Create marketplace-contracts package (DONE)
2. ⏭️ Create civitai-adapter package
   - Implement `fetchCivitAIModels()`
   - Implement `convertCivitAIModel()`
   - Add tests
3. ⏭️ Create huggingface-adapter package
   - Implement `fetchHuggingFaceModels()`
   - Implement `convertHFModel()`
   - Add tests
4. ⏭️ Use adapters in Next.js marketplace app
   - Create `/models/civitai` page
   - Create `/models/huggingface` page
   - Implement client-side filtering

## File Structure

```
frontend/packages/
├── marketplace-contracts/     ✅ COMPLETE
│   ├── package.json
│   ├── tsconfig.json
│   ├── README.md
│   └── src/
│       ├── index.ts
│       ├── common.ts
│       ├── civitai.ts
│       └── huggingface.ts
│
├── civitai-adapter/          ⏭️ TODO
│   ├── package.json
│   ├── tsconfig.json
│   ├── README.md
│   └── src/
│       ├── index.ts
│       ├── fetch.ts
│       └── convert.ts
│
└── huggingface-adapter/      ⏭️ TODO
    ├── package.json
    ├── tsconfig.json
    ├── README.md
    └── src/
        ├── index.ts
        ├── fetch.ts
        └── convert.ts
```

---

**TEAM-476 RULE ZERO:** Contracts define the interface. Adapters implement the interface. Next.js only imports contracts, never raw API types.
