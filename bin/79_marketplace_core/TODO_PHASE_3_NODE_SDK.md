# TODO: Phase 3 - Update Node.js SDK

**TEAM-429: Import shared filter types from WASM bindings**

## Status: ✅ COMPLETE

See `PHASE_3_NODE_SDK_COMPLETE.md` for implementation details.

## What Needs to Be Done

### 1. Update `marketplace-node/src/civitai.ts`

**Current signature:**
```typescript
export async function fetchCivitAIModels(options: {
  query?: string
  limit?: number
  page?: number
  types?: string[]
  sort?: 'Highest Rated' | 'Most Downloaded' | 'Newest'
  nsfw?: boolean
  period?: 'AllTime' | 'Year' | 'Month' | 'Week' | 'Day'
  baseModel?: string
}): Promise<CivitAIModel[]>
```

**New signature:**
```typescript
import { CivitaiFilters } from '../wasm/marketplace_sdk'

export async function fetchCivitAIModels(
  filters: CivitaiFilters
): Promise<CivitAIModel[]>
```

### 2. Update Implementation

Replace manual parameter building with filter struct fields:

```typescript
const params = new URLSearchParams({
  limit: String(filters.limit),
  page: filters.page ? String(filters.page) : undefined,
  sort: filters.sort.as_str(), // Use the as_str() method
})

// Model types
if (filters.model_type !== CivitaiModelType.All) {
  params.append('types', filters.model_type.as_str())
} else {
  params.append('types', 'Checkpoint')
  params.append('types', 'LORA')
}

// Time period
if (filters.time_period !== TimePeriod.AllTime) {
  params.append('period', filters.time_period.as_str())
}

// Base model
if (filters.base_model !== BaseModel.All) {
  params.append('baseModel', filters.base_model.as_str())
}

// NSFW filtering
const nsfwLevels = filters.nsfw.max_level.allowed_levels()
for (const level of nsfwLevels) {
  params.append('nsfwLevel', String(level.as_number()))
}
```

### 3. Update `marketplace-node/src/index.ts`

**Current:**
```typescript
export async function getCompatibleCivitaiModels(options = {}) {
  const { limit = 100, page, types = ['Checkpoint', 'LORA'], period, baseModel } = options
  
  try {
    const civitaiModels = await fetchCivitAIModels({
      limit,
      page,
      types,
      sort: 'Most Downloaded',
      period,
      baseModel,
    })
    // ...
  }
}
```

**New:**
```typescript
import { CivitaiFilters } from './wasm/marketplace_sdk'

export async function getCompatibleCivitaiModels(
  filters?: Partial<CivitaiFilters>
): Promise<Model[]> {
  const defaultFilters = CivitaiFilters.default()
  const mergedFilters = { ...defaultFilters, ...filters }
  
  try {
    const civitaiModels = await fetchCivitAIModels(mergedFilters)
    return civitaiModels.map(convertCivitAIModel)
  } catch (error) {
    console.error('[marketplace-node] Failed to fetch CivitAI models:', error)
    return []
  }
}
```

### 4. Rebuild WASM Bindings

After updating Rust SDK, rebuild WASM to get TypeScript types:

```bash
cd bin/79_marketplace_core/marketplace-sdk
wasm-pack build --target nodejs --out-dir ../marketplace-node/wasm
```

This will generate:
- `marketplace_sdk.d.ts` - TypeScript definitions
- `marketplace_sdk.js` - JavaScript bindings
- `marketplace_sdk_bg.wasm` - WASM binary

### 5. Update Tests

Update any tests that call `fetchCivitAIModels` or `getCompatibleCivitaiModels`:

```typescript
// Old
const models = await fetchCivitAIModels({
  limit: 10,
  types: ['Checkpoint'],
  sort: 'Most Downloaded',
})

// New
import { CivitaiFilters, CivitaiModelType, CivitaiSort } from './wasm/marketplace_sdk'

const filters = {
  ...CivitaiFilters.default(),
  limit: 10,
  model_type: CivitaiModelType.Checkpoint,
  sort: CivitaiSort.MostDownloaded,
}
const models = await fetchCivitAIModels(filters)
```

## Files to Modify

- [ ] `bin/79_marketplace_core/marketplace-node/src/civitai.ts`
- [ ] `bin/79_marketplace_core/marketplace-node/src/index.ts`
- [ ] `bin/79_marketplace_core/marketplace-node/test-civitai.ts` (if exists)
- [ ] `bin/79_marketplace_core/marketplace-node/test-civitai.js` (if exists)

## Verification

```bash
cd bin/79_marketplace_core/marketplace-node
npm run build
npm test
```

## Benefits

✅ Type-safe filters from Rust
✅ No more duplicate type definitions
✅ Automatic TypeScript types from WASM
✅ Consistent API across Rust and Node.js

---

**Next:** Phase 4 - Update Frontend
