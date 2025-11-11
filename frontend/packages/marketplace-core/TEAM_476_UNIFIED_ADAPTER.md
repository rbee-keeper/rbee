# TEAM-476: Unified Adapter Interface - RULE ZERO FIX

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Create ONE unified API for all vendors - no vendor-specific code in containers

## Problem Solved

**BEFORE (BAD):**
```typescript
// Container needs to know about EVERY vendor
if (source === 'civitai') {
  const { fetchCivitAIModels } = await import('@rbee/marketplace-core/adapters/civitai')
  response = await fetchCivitAIModels({ limit, page })
} else if (source === 'huggingface') {
  const { fetchHuggingFaceModels } = await import('@rbee/marketplace-core/adapters/huggingface')
  response = await fetchHuggingFaceModels({ limit, sort, direction })
}
// Adding Ollama? Replicate? More if/else!
```

**AFTER (GOOD):**
```typescript
// Container doesn't care about vendors!
const adapter = getAdapter(vendor)
const response = await adapter.fetchModels({ limit, page })
// Adding Ollama? Just register it in registry.ts!
```

## Architecture

### 1. Unified Adapter Interface

**File:** `/src/adapters/adapter.ts`

```typescript
export interface MarketplaceAdapter {
  readonly name: string
  fetchModels(params?: MarketplaceFilterParams): Promise<PaginatedResponse<MarketplaceModel>>
  fetchModel(id: string | number): Promise<MarketplaceModel>
}
```

**Generic Filter Params:**
```typescript
export interface MarketplaceFilterParams {
  query?: string
  limit?: number
  page?: number
  cursor?: string
  sort?: 'downloads' | 'likes' | 'trending' | 'updated' | 'created'
  direction?: 'asc' | 'desc'
  tags?: string[]
  author?: string
  nsfw?: boolean
}
```

### 2. Adapter Registry

**File:** `/src/adapters/registry.ts`

```typescript
export const adapters = {
  civitai: civitaiAdapter,
  huggingface: huggingfaceAdapter,
  // Future: ollama, replicate, etc.
} as const

export function getAdapter(vendor: VendorName): MarketplaceAdapter {
  return adapters[vendor]
}
```

### 3. Vendor Adapters

Each adapter implements the interface:

**CivitAI:**
```typescript
export const civitaiAdapter: MarketplaceAdapter = {
  name: 'civitai',
  async fetchModels(params) {
    // Map generic params → CivitAI params
    const civitaiParams = { limit: params.limit, page: params.page, ... }
    return fetchCivitAIModels(civitaiParams)
  },
  async fetchModel(id) {
    return fetchCivitAIModel(Number(id))
  },
}
```

**HuggingFace:**
```typescript
export const huggingfaceAdapter: MarketplaceAdapter = {
  name: 'huggingface',
  async fetchModels(params) {
    // Map generic params → HuggingFace params
    const hfParams = { limit: params.limit, search: params.query, ... }
    return fetchHuggingFaceModels(hfParams)
  },
  async fetchModel(id) {
    return fetchHuggingFaceModel(String(id))
  },
}
```

## Usage

### Container (No Vendor Logic!)

```typescript
export function ModelListContainer({ vendor, filters }: Props) {
  const fetchModels = async () => {
    // ONE line - works for ALL vendors!
    const adapter = getAdapter(vendor)
    const response = await adapter.fetchModels({ ...filters, limit, page })
    setModels(response.items)
  }
}
```

### Pages

```typescript
// CivitAI page
<ModelListContainer vendor="civitai" filters={{ sort: 'downloads' }}>
  {({ models }) => <CardGrid models={models} />}
</ModelListContainer>

// HuggingFace page
<ModelListContainer vendor="huggingface" filters={{ sort: 'likes' }}>
  {({ models }) => <Table models={models} />}
</ModelListContainer>

// Future: Ollama page (no container changes!)
<ModelListContainer vendor="ollama" filters={{ sort: 'trending' }}>
  {({ models }) => <List models={models} />}
</ModelListContainer>
```

## Adding New Vendors

**Step 1:** Create adapter implementing `MarketplaceAdapter`

```typescript
// /src/adapters/ollama/index.ts
export const ollamaAdapter: MarketplaceAdapter = {
  name: 'ollama',
  async fetchModels(params) {
    // Map params → Ollama API
    return fetchOllamaModels(...)
  },
  async fetchModel(id) {
    return fetchOllamaModel(id)
  },
}
```

**Step 2:** Register in registry

```typescript
// /src/adapters/registry.ts
export const adapters = {
  civitai: civitaiAdapter,
  huggingface: huggingfaceAdapter,
  ollama: ollamaAdapter, // ← That's it!
} as const
```

**Step 3:** Use it!

```typescript
<ModelListContainer vendor="ollama" />
```

## Benefits

✅ **No Vendor Logic in Containers** - Container doesn't know about vendors  
✅ **Type Safe** - `VendorName` is type-checked  
✅ **Easy to Extend** - Add vendor = 2 lines in registry  
✅ **Unified Interface** - All vendors use same params  
✅ **Adapter Pattern** - Each vendor maps generic → specific params  
✅ **Future Proof** - Add Ollama, Replicate, etc. without touching containers  

## Files Created/Modified

**Created:**
- `/src/adapters/adapter.ts` - Unified interface
- `/src/adapters/registry.ts` - Adapter registry

**Modified:**
- `/src/adapters/civitai/index.ts` - Added `civitaiAdapter` export
- `/src/adapters/huggingface/index.ts` - Added `huggingfaceAdapter` export
- `/src/index.ts` - Export unified interface
- `/apps/marketplace/src/components/ModelListContainer.tsx` - Use unified interface

## Exports

```typescript
// From @rbee/marketplace-core
export type { MarketplaceAdapter, MarketplaceFilterParams } from './adapters/adapter'
export type { VendorName } from './adapters/registry'
export { getAdapter, adapters } from './adapters/registry'
```

---

**TEAM-476 RULE ZERO:** ONE unified interface. Containers don't know about vendors. Adding vendors = registry update only. No if/else vendor logic!
