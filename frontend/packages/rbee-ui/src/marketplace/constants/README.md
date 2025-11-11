# Marketplace Filter Constants

**TEAM-467: SINGLE SOURCE OF TRUTH for all marketplace filter constants**

This directory contains the **shared filter constants** used by:
- ✅ **Next.js marketplace app** (`/frontend/apps/marketplace`)
- ✅ **Tauri Keeper app** (`/bin/00_rbee_keeper/ui`)

## Files

### `filter-constants.ts`
Raw filter values used in URLs and API calls.

**Exports:**
- `HF_SORTS`, `HF_SIZES`, `HF_LICENSES` - HuggingFace filter values
- `CIVITAI_NSFW_LEVELS`, `CIVITAI_TIME_PERIODS`, `CIVITAI_MODEL_TYPES`, `CIVITAI_BASE_MODELS`, `CIVITAI_SORTS` - CivitAI filter values
- Type exports for all constants

### `filter-groups.ts`
UI-ready `FilterGroup` objects for use with `UniversalFilterBar`.

**Exports:**
- `HUGGINGFACE_FILTER_GROUPS` - Array of filter groups for HF
- `HUGGINGFACE_SORT_GROUP` - Sort options for HF
- `CIVITAI_FILTER_GROUPS` - Array of filter groups for CivitAI
- `CIVITAI_SORT_GROUP` - Sort options for CivitAI
- `HuggingFaceFilters` - Type-safe filter state interface
- `CivitaiFilters` - Type-safe filter state interface

## Usage

### In Next.js App

```typescript
import {
  HUGGINGFACE_FILTER_GROUPS,
  HUGGINGFACE_SORT_GROUP,
  type HuggingFaceFilters,
} from '@rbee/ui/marketplace'
```

### In Tauri Keeper App

```typescript
import {
  CIVITAI_FILTER_GROUPS,
  CIVITAI_SORT_GROUP,
  type CivitaiFilters,
} from '@rbee/ui/marketplace'
```

### For Manifest Generation

```typescript
import {
  HF_SORTS,
  HF_SIZES,
  CIVITAI_TIME_PERIODS,
  CIVITAI_MODEL_TYPES,
} from '@rbee/ui/marketplace'
```

## CivitAI NSFW Levels

**Source of Truth:** WASM contract from Rust SDK
```typescript
// /bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts
export type NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
```

**Mapping:**

| URL Slug | API Enum | Numeric Values | Description |
|----------|----------|----------------|-------------|
| `pg` | `None` | `[1]` | PG only |
| `pg13` | `Soft` | `[1, 2]` | PG + PG-13 |
| `r` | `Mature` | `[1, 2, 4]` | Up to R-rated |
| `x` | `X` | `[1, 2, 4, 8]` | Up to X-rated |
| `all` | `XXX` | `[1, 2, 4, 8, 16]` | ALL levels (default) |

## Migration Notes

### Before (Duplicated) ❌
```
/frontend/apps/marketplace/config/filter-constants.ts  (Next.js)
/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx  (Tauri)
/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx  (Tauri)
```

### After (Centralized) ✅
```
/frontend/packages/rbee-ui/src/marketplace/constants/
  ├── filter-constants.ts  (Raw values)
  ├── filter-groups.ts     (UI-ready FilterGroups)
  └── index.ts             (Re-exports)
```

**Apps now import from:** `@rbee/ui/marketplace`

## Rule Zero Compliance

✅ **Breaking changes > backwards compatibility**
- Removed all duplicated filter definitions
- Apps now import from shared package
- No `_v2` or `_new` constants

✅ **Single source of truth**
- ONE place to edit filter values
- Changes propagate to both apps automatically
- Type-safe with shared interfaces

✅ **Delete deprecated code**
- Removed inline filter definitions from Tauri pages
- Next.js app re-exports from shared package for backwards compatibility
