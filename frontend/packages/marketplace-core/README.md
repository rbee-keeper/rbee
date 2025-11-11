# @rbee/marketplace-contracts

**TEAM-476:** Type contracts for marketplace adapters and Next.js app

## Purpose

This package defines the **contract** between:
1. **Marketplace adapters** (civitai-adapter, huggingface-adapter)
2. **Next.js marketplace app** (frontend/apps/marketplace)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Marketplace App                   │
│                  (expects MarketplaceModel)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐             ┌───────▼────────┐
│ civitai-adapter│             │huggingface-    │
│                │             │adapter         │
│ (converts      │             │(converts       │
│  CivitAIModel  │             │ HFModel        │
│  to Marketplace│             │ to Marketplace │
│  Model)        │             │ Model)         │
└───────┬────────┘             └───────┬────────┘
        │                               │
        │    Both import contracts      │
        └───────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │ marketplace-contracts │
            │                       │
            │ - MarketplaceModel    │
            │ - CivitAIModel        │
            │ - HuggingFaceModel    │
            └───────────────────────┘
```

## Types Provided

### Common Types (Contract)

**`MarketplaceModel`** - What Next.js expects:
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

**`PaginatedResponse<T>`** - Standard pagination:
```typescript
interface PaginatedResponse<T> {
  items: T[]
  meta: PaginationMeta
}
```

### CivitAI Types

Full type definitions for CivitAI API v1:
- `CivitAIModel` - Full model response
- `CivitAIListModelsParams` - Query parameters
- `CivitAIModelType`, `CivitAISort`, etc. - Enums

### HuggingFace Types

Full type definitions for HuggingFace Hub API:
- `HuggingFaceModel` - Full model response
- `HuggingFaceListModelsParams` - Query parameters
- `HuggingFaceTask`, `HuggingFaceSort`, etc. - Enums

## Usage

### In civitai-adapter

```typescript
import type {
  MarketplaceModel,
  CivitAIModel,
  CivitAIListModelsParams,
} from '@rbee/marketplace-contracts'

export function convertCivitAIModel(model: CivitAIModel): MarketplaceModel {
  return {
    id: String(model.id),
    name: model.name,
    author: model.creator.username,
    // ... convert all fields
  }
}
```

### In huggingface-adapter

```typescript
import type {
  MarketplaceModel,
  HuggingFaceModel,
  HuggingFaceListModelsParams,
} from '@rbee/marketplace-contracts'

export function convertHFModel(model: HuggingFaceModel): MarketplaceModel {
  return {
    id: model.id,
    name: model.id.split('/')[1] || model.id,
    author: model.author || model.id.split('/')[0],
    // ... convert all fields
  }
}
```

### In Next.js marketplace app

```typescript
import type { MarketplaceModel } from '@rbee/marketplace-contracts'
import { fetchCivitAIModels } from '@rbee/civitai-adapter'
import { fetchHuggingFaceModels } from '@rbee/huggingface-adapter'

// Both return MarketplaceModel[]
const civitaiModels = await fetchCivitAIModels({ limit: 100 })
const hfModels = await fetchHuggingFaceModels({ limit: 100 })

// Same interface, can be used interchangeably
function ModelCard({ model }: { model: MarketplaceModel }) {
  return <div>{model.name} by {model.author}</div>
}
```

## Benefits

✅ **Type Safety** - Compile-time checks across packages  
✅ **Single Source of Truth** - Contract defined once  
✅ **Adapter Independence** - Each adapter converts to common format  
✅ **Easy Testing** - Mock `MarketplaceModel` for tests  
✅ **Future-Proof** - Add new adapters without changing Next.js  

## API Documentation

### CivitAI
- Docs: https://github.com/civitai/civitai/wiki/REST-API-Reference
- Endpoint: `https://civitai.com/api/v1/models`

### HuggingFace
- Docs: https://huggingface.co/docs/hub/en/api
- Endpoint: `https://huggingface.co/api/models`

## Next Steps

1. ✅ Types defined (this package)
2. ⏭️ Create `civitai-adapter` package
3. ⏭️ Create `huggingface-adapter` package
4. ⏭️ Use adapters in Next.js marketplace app

---

**TEAM-476 RULE ZERO:** This package is the contract. Adapters MUST return `MarketplaceModel`. Next.js ONLY imports from this package, never directly from adapters.
