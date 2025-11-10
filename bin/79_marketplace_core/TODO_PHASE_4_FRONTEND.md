# TODO: Phase 4 - Update Frontend

**TEAM-XXX: Replace frontend filter types with contract imports**

## Status: NOT STARTED

## What Needs to Be Done

### 1. Delete Duplicate Filter Definitions

**File:** `frontend/apps/marketplace/app/models/civitai/filters.ts`

**DELETE these type definitions (lines 5-15):**
```typescript
export type TimePeriod = 'AllTime' | 'Month' | 'Week' | 'Day'
export type ModelType = 'All' | 'Checkpoint' | 'LORA'
export type BaseModel = 'All' | 'SDXL 1.0' | 'SD 1.5' | 'SD 2.1'
export type SortBy = 'downloads' | 'likes' | 'newest'

export interface CivitaiFilters {
  timePeriod: TimePeriod
  modelType: ModelType
  baseModel: BaseModel
  sort: SortBy
}
```

### 2. Import from Contract

**Add to top of file:**
```typescript
import {
  CivitaiFilters,
  TimePeriod,
  CivitaiModelType,
  BaseModel,
  CivitaiSort,
  NsfwLevel,
  NsfwFilter,
} from '@rbee/artifacts-contract'
```

### 3. Update Filter Groups

**Replace hardcoded options with contract enums:**

```typescript
export const CIVITAI_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'timePeriod',
    label: 'Time Period',
    options: [
      { label: 'All Time', value: TimePeriod.AllTime },
      { label: 'Month', value: TimePeriod.Month },
      { label: 'Week', value: TimePeriod.Week },
      { label: 'Day', value: TimePeriod.Day },
    ],
  },
  {
    id: 'modelType',
    label: 'Model Type',
    options: [
      { label: 'All Types', value: CivitaiModelType.All },
      { label: 'Checkpoint', value: CivitaiModelType.Checkpoint },
      { label: 'LoRA', value: CivitaiModelType.Lora },
    ],
  },
  {
    id: 'baseModel',
    label: 'Base Model',
    options: [
      { label: 'All Models', value: BaseModel.All },
      { label: 'SDXL 1.0', value: BaseModel.SdxlV1 },
      { label: 'SD 1.5', value: BaseModel.SdV15 },
      { label: 'SD 2.1', value: BaseModel.SdV21 },
    ],
  },
]

export const CIVITAI_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: CivitaiSort.MostDownloaded },
    { label: 'Most Likes', value: CivitaiSort.HighestRated },
    { label: 'Newest', value: CivitaiSort.Newest },
  ],
}
```

### 4. Add NSFW Filter Group

**NEW filter group for NSFW levels:**

```typescript
export const NSFW_FILTER_GROUP: FilterGroup = {
  id: 'nsfw',
  label: 'Content Filter',
  options: [
    { label: 'PG (Safe for work)', value: NsfwLevel.None },
    { label: 'PG-13 (Suggestive)', value: NsfwLevel.Soft },
    { label: 'R (Mature)', value: NsfwLevel.Mature },
    { label: 'X (Explicit)', value: NsfwLevel.X },
    { label: 'XXX (Pornographic)', value: NsfwLevel.Xxx },
  ],
}
```

### 5. Update `buildFilterParams()`

**Replace manual mapping with contract types:**

```typescript
export function buildFilterParams(filters: CivitaiFilters) {
  const params = {
    limit: filters.limit,
    page: filters.page,
  }

  // Model types
  if (filters.model_type !== CivitaiModelType.All) {
    params.types = [filters.model_type.as_str()]
  } else {
    params.types = ['Checkpoint', 'LORA']
  }

  // Time period
  if (filters.time_period !== TimePeriod.AllTime) {
    params.period = filters.time_period.as_str()
  }

  // Base model
  if (filters.base_model !== BaseModel.All) {
    params.baseModel = filters.base_model.as_str()
  }

  // Sort
  params.sort = filters.sort.as_str()

  // NSFW filtering
  const nsfwLevels = filters.nsfw.max_level.allowed_levels()
  params.nsfwLevel = nsfwLevels.map(l => l.as_number())

  return params
}
```

### 6. Update Pre-generated Filters

**Use contract types in PREGENERATED_FILTERS:**

```typescript
export const PREGENERATED_FILTERS: GenericFilterConfig<CivitaiFilters>[] = [
  {
    filters: {
      time_period: TimePeriod.AllTime,
      model_type: CivitaiModelType.All,
      base_model: BaseModel.All,
      sort: CivitaiSort.MostDownloaded,
      nsfw: NsfwFilter.default(),
      page: null,
      limit: 100,
    },
    path: '',
  },
  {
    filters: {
      time_period: TimePeriod.Month,
      model_type: CivitaiModelType.All,
      base_model: BaseModel.All,
      sort: CivitaiSort.MostDownloaded,
      nsfw: NsfwFilter.default(),
      page: null,
      limit: 100,
    },
    path: 'filter/month',
  },
  // ... etc
]
```

## Files to Modify

- [ ] `frontend/apps/marketplace/app/models/civitai/filters.ts`
- [ ] `frontend/apps/marketplace/app/models/civitai/page.tsx`
- [ ] `frontend/apps/marketplace/app/models/civitai/[...filter]/page.tsx`
- [ ] `frontend/apps/marketplace/components/ModelsFilterBar.tsx`

## Package.json Update

Add dependency to `@rbee/artifacts-contract`:

```json
{
  "dependencies": {
    "@rbee/artifacts-contract": "workspace:*"
  }
}
```

## Verification

```bash
cd frontend/apps/marketplace
npm run build
npm run type-check
```

## Benefits

✅ No duplicate type definitions
✅ Type-safe filters from contract
✅ Automatic updates when contract changes
✅ Same types in Next.js and Tauri

---

**Next:** Phase 5 - Add Filters to Tauri GUI
