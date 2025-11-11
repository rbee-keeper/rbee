# TEAM-467: Complete Filter Centralization

**Status:** ✅ COMPLETE

## Summary

Eliminated **ALL** filter constant duplication across the codebase by creating a **single shared package** used by both Next.js and Tauri apps.

## Problem

Filter constants were duplicated in **3 places:**
1. ❌ Next.js app: `/frontend/apps/marketplace/config/filter-constants.ts`
2. ❌ Tauri app: `/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`
3. ❌ Tauri app: `/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx`

**Risk:** Changes in one place wouldn't propagate to others, causing inconsistencies.

## Solution

Created **shared constants package** in `@rbee/ui/marketplace`:

```
/frontend/packages/rbee-ui/src/marketplace/constants/
├── filter-constants.ts    # Raw filter values (URL slugs, API values)
├── filter-groups.ts       # UI-ready FilterGroup objects
├── index.ts               # Re-exports
└── README.md              # Documentation
```

## Changes Made

### 1. Created Shared Constants Package ✅

**Location:** `/frontend/packages/rbee-ui/src/marketplace/constants/`

**Files:**
- `filter-constants.ts` - Raw constants (HF + CivitAI)
- `filter-groups.ts` - UI-ready FilterGroup definitions
- `index.ts` - Barrel export
- `README.md` - Usage documentation

### 2. Updated Next.js App ✅

**File:** `/frontend/apps/marketplace/config/filter-constants.ts`

**Before:** Defined all constants inline (76 lines)

**After:** Re-exports from shared package (27 lines)
```typescript
export {
  HF_SORTS,
  HF_SIZES,
  // ... all constants
} from '@rbee/ui/marketplace'
```

### 3. Updated Tauri Keeper App ✅

**Files Updated:**
- `/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`
- `/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx`

**Before:** Defined FilterGroup arrays inline (~50 lines each)

**After:** Imports from shared package
```typescript
import {
  CIVITAI_FILTER_GROUPS,
  CIVITAI_SORT_GROUP,
  type CivitaiFilters,
} from '@rbee/ui/marketplace'
```

### 4. Verified CivitAI API Compliance ✅

**Source of Truth:** WASM contract from Rust SDK
```typescript
// /bin/79_marketplace_core/marketplace-node/wasm/marketplace_sdk.d.ts
export type NsfwLevel = "None" | "Soft" | "Mature" | "X" | "XXX"
```

**Documented Mapping:**
- `'pg'` → `'None'` (API: `[1]`)
- `'pg13'` → `'Soft'` (API: `[1, 2]`)
- `'r'` → `'Mature'` (API: `[1, 2, 4]`)
- `'x'` → `'X'` (API: `[1, 2, 4, 8]`)
- `'all'` → `'XXX'` (API: `[1, 2, 4, 8, 16]`)

### 5. Created Shared Filter Parser ✅

**File:** `/frontend/apps/marketplace/config/filter-parser.ts`

Converts filter paths (e.g., `"filter/week/loras/sdxl"`) to API parameters using the shared constants.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ @rbee/ui/marketplace/constants                              │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ SINGLE SOURCE OF TRUTH                                      │
│                                                             │
│ filter-constants.ts:                                        │
│ - HF_SORTS, HF_SIZES, HF_LICENSES                         │
│ - CIVITAI_NSFW_LEVELS, CIVITAI_TIME_PERIODS, etc.        │
│                                                             │
│ filter-groups.ts:                                          │
│ - HUGGINGFACE_FILTER_GROUPS, HUGGINGFACE_SORT_GROUP      │
│ - CIVITAI_FILTER_GROUPS, CIVITAI_SORT_GROUP              │
│ - HuggingFaceFilters, CivitaiFilters (types)             │
└─────────────────────────────────────────────────────────────┘
                    ▲                    ▲
                    │                    │
        ┌───────────┴──────┐   ┌────────┴─────────────┐
        │                  │   │                      │
┌───────▼──────────────┐   │   │   ┌──────────────────▼─────┐
│ Next.js Marketplace  │   │   │   │ Tauri Keeper App       │
│ ━━━━━━━━━━━━━━━━━━━ │   │   │   │ ━━━━━━━━━━━━━━━━━━━━━ │
│                      │   │   │   │                        │
│ config/              │   │   │   │ pages/                 │
│ - filter-constants   │───┘   └───│ - MarketplaceCivitai   │
│ - filter-parser      │           │ - MarketplaceHuggingFace│
│ - filters            │           │                        │
│                      │           │ Both import from       │
│ Re-exports from      │           │ @rbee/ui/marketplace   │
│ @rbee/ui/marketplace │           │                        │
└──────────────────────┘           └────────────────────────┘
```

## Benefits

### Before ❌
- **3 separate definitions** of filter constants
- **Manual synchronization** required
- **High risk** of inconsistencies
- **Duplicated code** (~150 lines total)

### After ✅
- **1 shared package** for all filter constants
- **Automatic propagation** of changes
- **Type-safe** with shared interfaces
- **DRY principle** enforced
- **~80% less code** in apps

## Files Changed

### Created
- ✅ `/frontend/packages/rbee-ui/src/marketplace/constants/filter-constants.ts`
- ✅ `/frontend/packages/rbee-ui/src/marketplace/constants/filter-groups.ts`
- ✅ `/frontend/packages/rbee-ui/src/marketplace/constants/index.ts`
- ✅ `/frontend/packages/rbee-ui/src/marketplace/constants/README.md`

### Modified
- ✅ `/frontend/packages/rbee-ui/src/marketplace/index.ts` - Export constants
- ✅ `/frontend/apps/marketplace/config/filter-constants.ts` - Re-export from shared
- ✅ `/bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx` - Use shared constants
- ✅ `/bin/00_rbee_keeper/ui/src/pages/MarketplaceHuggingFace.tsx` - Use shared constants

### Documentation
- ✅ `/frontend/apps/marketplace/.docs/TEAM_467_FILTER_CENTRALIZATION.md`
- ✅ `/frontend/apps/marketplace/.docs/TEAM_467_FAIL_FAST_FIX.md`

## Rule Zero Compliance

✅ **Breaking changes > backwards compatibility**
- Removed all duplicated filter definitions
- Updated both apps to use shared package
- No `_v2` or wrapper functions

✅ **Single source of truth**
- ONE package for ALL filter constants
- Changes propagate automatically
- Type-safe with shared interfaces

✅ **Delete deprecated code**
- Removed inline filter definitions from Tauri pages
- Removed duplicated constants from Next.js app
- No legacy code left behind

## Verification

### Import Paths

**Next.js app:**
```typescript
import { HF_SORTS, CIVITAI_TIME_PERIODS } from '@rbee/ui/marketplace'
```

**Tauri Keeper app:**
```typescript
import {
  CIVITAI_FILTER_GROUPS,
  CIVITAI_SORT_GROUP,
  type CivitaiFilters,
} from '@rbee/ui/marketplace'
```

**Manifest generation:**
```typescript
import { getAllCivitAIFilters, getAllHFFilters } from '../config/filters'
// Which internally uses constants from @rbee/ui/marketplace
```

### Type Safety

All filter states are now type-safe:
```typescript
// Shared types from @rbee/ui/marketplace
type HuggingFaceFilters = {
  sort: 'downloads' | 'likes'
  size: 'all' | 'small' | 'medium' | 'large'
  license: 'all' | 'apache' | 'mit' | 'other'
}

type CivitaiFilters = {
  timePeriod: 'all' | 'week' | 'month' | 'day'
  modelType: 'all' | 'checkpoints' | 'loras'
  baseModel: 'all' | 'sdxl' | 'sd15' | 'sd21'
  sort: 'downloads' | 'likes' | 'newest'
}
```

## Related Work

- **TEAM-467 FAIL FAST Fix:** Fixed manifest generation to exit immediately on errors
- **TEAM-467 Filter Parser:** Created shared parser for filter paths → API params
- **TEAM-467 CivitAI API Compliance:** Verified NSFW levels against WASM contract

## Next Steps

None required - centralization is complete! ✅

**To add new filters:**
1. Update `/frontend/packages/rbee-ui/src/marketplace/constants/filter-constants.ts`
2. Update `/frontend/packages/rbee-ui/src/marketplace/constants/filter-groups.ts`
3. Changes automatically propagate to both apps

**No more duplication!** 🎉
