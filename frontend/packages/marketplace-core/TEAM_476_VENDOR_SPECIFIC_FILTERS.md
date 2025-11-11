# TEAM-476: Vendor-Specific Filters - NO HEURISTICS!

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Support vendor-specific filters and sorting WITHOUT mapping heuristics

## Problem Solved

**BEFORE (BAD - Heuristics):**
```typescript
// Adapter tries to map "generic" params to vendor params
if (params.sort === 'downloads') civitaiParams.sort = 'Most Downloaded'
else if (params.sort === 'likes') civitaiParams.sort = 'Most Reactions'
// What if CivitAI adds new sort options? More if/else!
```

**AFTER (GOOD - Direct):**
```typescript
// Adapter accepts vendor-specific params directly!
async fetchModels(params: CivitAIListModelsParams) {
  return fetchCivitAIModels(params) // No mapping!
}
```

## Architecture

### 1. Generic Adapter Interface with Type Parameter

**File:** `/src/adapters/adapter.ts`

```typescript
export interface MarketplaceAdapter<TFilters = unknown> {
  readonly name: string
  fetchModels(params?: TFilters): Promise<PaginatedResponse<MarketplaceModel>>
  fetchModel(id: string | number): Promise<MarketplaceModel>
}
```

**Key:** `TFilters` = Vendor-specific filter type!

### 2. Vendor Adapters with Specific Types

**CivitAI:**
```typescript
export const civitaiAdapter: MarketplaceAdapter<CivitAIListModelsParams> = {
  name: 'civitai',
  async fetchModels(params: CivitAIListModelsParams = {}) {
    return fetchCivitAIModels(params) // Direct pass-through!
  },
}
```

**CivitAI Filters (Vendor-Specific):**
```typescript
interface CivitAIListModelsParams {
  limit?: number
  page?: number
  query?: string
  tag?: string
  username?: string
  types?: CivitAIModelType[]
  sort?: CivitAISort // 'Most Downloaded' | 'Most Reactions' | 'Highest Rated' | ...
  period?: CivitAITimePeriod // 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
  rating?: number
  nsfwLevel?: number[]
  baseModels?: string[]
  allowCommercialUse?: string[]
  // ... all CivitAI-specific options!
}
```

**HuggingFace:**
```typescript
export const huggingfaceAdapter: MarketplaceAdapter<HuggingFaceListModelsParams> = {
  name: 'huggingface',
  async fetchModels(params: HuggingFaceListModelsParams = {}) {
    return fetchHuggingFaceModels(params) // Direct pass-through!
  },
}
```

**HuggingFace Filters (Vendor-Specific):**
```typescript
interface HuggingFaceListModelsParams {
  limit?: number
  search?: string
  author?: string
  filter?: string
  sort?: HuggingFaceSort // 'downloads' | 'likes' | 'trending' | 'updated' | 'created'
  direction?: 1 | -1
  pipeline_tag?: string
  library?: string
  language?: string
  dataset?: string
  license?: string
  // ... all HuggingFace-specific options!
}
```

### 3. Container with Generic Filters

**File:** `/apps/marketplace/src/components/ModelListContainer.tsx`

```typescript
export interface ModelListContainerProps<TFilters = unknown> {
  vendor: VendorName
  filters?: TFilters // Vendor-specific filters!
  children: (props: ModelListRenderProps) => ReactNode
}

export function ModelListContainer<TFilters>({ vendor, filters }: Props<TFilters>) {
  const adapter = getAdapter(vendor)
  const response = await adapter.fetchModels(filters) // Type-safe!
}
```

## Usage Examples

### CivitAI Page (Vendor-Specific Filters)

```typescript
import type { CivitAIListModelsParams } from '@rbee/marketplace-core'

export default function CivitAIModelsPage() {
  return (
    <ModelListContainer<CivitAIListModelsParams>
      vendor="civitai"
      filters={{
        types: ['Checkpoint', 'LORA'],
        sort: 'Most Downloaded', // CivitAI-specific!
        period: 'Month', // CivitAI-specific!
        nsfwLevel: [1, 2, 4], // CivitAI-specific!
        baseModels: ['SD 1.5', 'SDXL 1.0'], // CivitAI-specific!
      }}
    >
      {({ models }) => <CardGrid models={models} />}
    </ModelListContainer>
  )
}
```

### HuggingFace Page (Vendor-Specific Filters)

```typescript
import type { HuggingFaceListModelsParams } from '@rbee/marketplace-core'

export default function HuggingFaceModelsPage() {
  return (
    <ModelListContainer<HuggingFaceListModelsParams>
      vendor="huggingface"
      filters={{
        sort: 'downloads', // HuggingFace-specific!
        direction: -1, // HuggingFace-specific!
        pipeline_tag: 'text-generation', // HuggingFace-specific!
        library: 'transformers', // HuggingFace-specific!
        language: 'en', // HuggingFace-specific!
      }}
    >
      {({ models }) => <Table models={models} />}
    </ModelListContainer>
  )
}
```

## Benefits

✅ **No Heuristics** - No mapping logic, no if/else  
✅ **Type Safe** - TypeScript enforces vendor-specific filters  
✅ **Full Vendor Features** - Access ALL vendor options  
✅ **No Abstraction Leaks** - Each vendor uses its own types  
✅ **Easy to Extend** - Add new vendor = define its filter type  
✅ **Future Proof** - Vendor adds new filter? Just use it!  

## Comparison

### Generic Filters (OLD - BAD)

```typescript
// Generic params (limited, requires mapping)
interface MarketplaceFilterParams {
  sort?: 'downloads' | 'likes' | 'trending' // Limited!
  direction?: 'asc' | 'desc'
}

// Adapter must map
if (params.sort === 'downloads') civitaiParams.sort = 'Most Downloaded'
// What about CivitAI's 'Highest Rated', 'Most Reactions', etc.?
```

**Problems:**
- ❌ Can't access all vendor features
- ❌ Requires mapping heuristics
- ❌ Breaks when vendor adds new options
- ❌ Abstraction leak (generic params don't match vendor)

### Vendor-Specific Filters (NEW - GOOD)

```typescript
// CivitAI params (full access)
interface CivitAIListModelsParams {
  sort?: 'Most Downloaded' | 'Most Reactions' | 'Highest Rated' | ... // All options!
  period?: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
  nsfwLevel?: number[]
  baseModels?: string[]
}

// Adapter passes through
async fetchModels(params: CivitAIListModelsParams) {
  return fetchCivitAIModels(params) // Direct!
}
```

**Benefits:**
- ✅ Full access to all vendor features
- ✅ No mapping needed
- ✅ Type-safe
- ✅ Future-proof

## Adding New Vendors

**Step 1:** Define vendor-specific filter type

```typescript
// /src/adapters/ollama/types.ts
export interface OllamaListModelsParams {
  limit?: number
  search?: string
  sort?: 'name' | 'modified' | 'size' // Ollama-specific!
  tags?: string[]
}
```

**Step 2:** Create adapter with specific type

```typescript
// /src/adapters/ollama/index.ts
export const ollamaAdapter: MarketplaceAdapter<OllamaListModelsParams> = {
  name: 'ollama',
  async fetchModels(params: OllamaListModelsParams = {}) {
    return fetchOllamaModels(params) // Direct!
  },
}
```

**Step 3:** Register in registry

```typescript
// /src/adapters/registry.ts
export const adapters = {
  civitai: civitaiAdapter,
  huggingface: huggingfaceAdapter,
  ollama: ollamaAdapter, // ← That's it!
}
```

**Step 4:** Use with vendor-specific filters

```typescript
<ModelListContainer<OllamaListModelsParams>
  vendor="ollama"
  filters={{
    sort: 'modified', // Ollama-specific!
    tags: ['llama', 'chat'],
  }}
>
  {({ models }) => <List models={models} />}
</ModelListContainer>
```

## Files Modified

**Created:**
- `/src/adapters/adapter.ts` - Generic interface with `TFilters` parameter

**Modified:**
- `/src/adapters/civitai/index.ts` - Use `MarketplaceAdapter<CivitAIListModelsParams>`
- `/src/adapters/huggingface/index.ts` - Use `MarketplaceAdapter<HuggingFaceListModelsParams>`
- `/src/index.ts` - Export `BaseFilterParams` instead of `MarketplaceFilterParams`
- `/apps/marketplace/src/components/ModelListContainer.tsx` - Generic `TFilters` parameter

## Exports

```typescript
// From @rbee/marketplace-core
export type { MarketplaceAdapter, BaseFilterParams } from './adapters/adapter'
export type { VendorName } from './adapters/registry'
export { getAdapter, adapters } from './adapters/registry'

// Vendor-specific types
export type { CivitAIListModelsParams, CivitAISort, ... } from './adapters/civitai/types'
export type { HuggingFaceListModelsParams, HuggingFaceSort, ... } from './adapters/huggingface/types'
```

---

**TEAM-476 RULE ZERO:** Vendor-specific filters. No heuristics. No mapping. Type-safe. Future-proof. Each vendor uses its own types!
