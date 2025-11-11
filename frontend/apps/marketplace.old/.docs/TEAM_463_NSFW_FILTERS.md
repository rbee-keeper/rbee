# TEAM-463: NSFW Filter Levels for CivitAI

**Date:** 2025-11-10  
**Author:** TEAM-463  
**Status:** ✅ COMPLETE

## Summary

Added 5 NSFW filter levels to CivitAI marketplace with dedicated SSG pages for each rating.

## Changes Made

### 1. Added NSFW Filter Group

**File:** `app/models/civitai/filters.ts`

Added new filter group with 5 NSFW levels:
- **PG (None)** - Safe for work
- **PG-13 (Soft)** - Suggestive content
- **R (Mature)** - Mature themes, partial nudity
- **X** - Explicit nudity
- **XXX** - Pornographic content

```typescript
{
  id: 'nsfwLevel',
  label: 'Content Rating',
  options: [
    { label: 'PG (Safe for work)', value: 'None' },
    { label: 'PG-13 (Suggestive)', value: 'Soft' },
    { label: 'R (Mature)', value: 'Mature' },
    { label: 'X (Explicit)', value: 'X' },
    { label: 'XXX (Pornographic)', value: 'XXX' },
  ],
}
```

### 2. Added Pregenerated Filter Paths

Created SSG pages for each NSFW level:

| Path | NSFW Level | Description |
|------|-----------|-------------|
| `/models/civitai` | PG (None) | Default - Safe for work |
| `/models/civitai/filter/pg` | PG (None) | Safe for work |
| `/models/civitai/filter/pg13` | PG-13 (Soft) | Suggestive content |
| `/models/civitai/filter/r` | R (Mature) | Mature content |
| `/models/civitai/filter/x` | X | Explicit content |
| `/models/civitai/filter/xxx` | XXX | Pornographic content |

Also added combinations:
- `/models/civitai/filter/r/checkpoints` - R-rated checkpoints
- `/models/civitai/filter/r/loras` - R-rated LORAs

### 3. Updated API Integration

**File:** `app/models/civitai/filters.ts`

```typescript
export function buildFilterParams(filters: CivitaiFilters): NodeCivitaiFilters {
  return {
    time_period: filters.timePeriod,
    model_type: filters.modelType,
    base_model: filters.baseModel,
    sort: convertSortToApi(filters.sort),
    nsfw: {
      max_level: filters.nsfwLevel || 'None',  // ← TEAM-463
      blur_mature: true,
    },
    page: null,
    limit: 100,
  }
}
```

### 4. Updated UI to Show NSFW Rating

**File:** `app/models/civitai/[...filter]/page.tsx`

Added NSFW rating badges with color coding:
- 🟢 Green - Safe for work (PG)
- 🟡 Yellow - Suggestive (PG-13)
- 🟠 Orange - Mature (R)
- 🔴 Red - Explicit (X, XXX)

```tsx
{currentFilter.nsfwLevel && (
  <div className="flex items-center gap-2">
    <div className={`size-2 rounded-full ${
      currentFilter.nsfwLevel === 'None' ? 'bg-green-500' :
      currentFilter.nsfwLevel === 'Soft' ? 'bg-yellow-500' :
      currentFilter.nsfwLevel === 'Mature' ? 'bg-orange-500' :
      'bg-red-500'
    }`} />
    <span>
      {currentFilter.nsfwLevel === 'None' ? 'Safe for work' :
       currentFilter.nsfwLevel === 'Soft' ? 'Suggestive content' :
       currentFilter.nsfwLevel === 'Mature' ? 'Mature content' :
       'Explicit content'}
    </span>
  </div>
)}
```

### 5. Updated Metadata for SEO

Added NSFW level to page titles and descriptions:

```typescript
// Before: "CivitAI Models | rbee Marketplace"
// After:  "R Checkpoint SDXL 1.0 - CivitAI Models | rbee Marketplace"
```

## How It Works

### Filter Flow

1. **User selects NSFW level** from filter dropdown
2. **Frontend converts** to API format:
   ```typescript
   nsfwLevel: 'Mature' → nsfw: { max_level: 'Mature', blur_mature: true }
   ```
3. **Backend filters** CivitAI API results based on NSFW level
4. **UI displays** appropriate rating badge and content warning

### SSG Pre-generation

All NSFW filter pages are pre-generated at build time:

```typescript
export async function generateStaticParams() {
  return PREGENERATED_FILTERS
    .filter((f) => f.path !== '')
    .map((f) => ({
      filter: f.path.split('/'),
    }))
}
```

This generates:
- `filter/pg` → `['filter', 'pg']`
- `filter/pg13` → `['filter', 'pg13']`
- `filter/r` → `['filter', 'r']`
- `filter/x` → `['filter', 'x']`
- `filter/xxx` → `['filter', 'xxx']`
- `filter/r/checkpoints` → `['filter', 'r', 'checkpoints']`
- etc.

## Total SSG Pages Added

**Before:** 10 pregenerated filter pages  
**After:** 17 pregenerated filter pages (+7)

New pages:
1. `/models/civitai/filter/pg` (PG)
2. `/models/civitai/filter/pg13` (PG-13)
3. `/models/civitai/filter/r` (R)
4. `/models/civitai/filter/x` (X)
5. `/models/civitai/filter/xxx` (XXX)
6. `/models/civitai/filter/r/checkpoints` (R + Checkpoints)
7. `/models/civitai/filter/r/loras` (R + LORAs)

## Files Modified

1. ✅ `frontend/apps/marketplace/app/models/civitai/filters.ts`
   - Added NSFW filter group
   - Added 7 new pregenerated filter paths
   - Updated `buildFilterParams()` to handle NSFW levels

2. ✅ `frontend/apps/marketplace/app/models/civitai/[...filter]/page.tsx`
   - Added NSFW level to metadata
   - Added NSFW rating badge to UI
   - Updated filter description to show rating

3. ✅ `frontend/apps/marketplace/app/models/civitai/page.tsx`
   - Updated default filter comment

4. ✅ `frontend/apps/marketplace/app/models/civitai/[slug]/page.tsx`
   - Fixed typo: `sizeKB` → `sizeKb`

## Verification

```bash
# TypeScript compilation
cd frontend/apps/marketplace
pnpm tsc --noEmit
# ✅ SUCCESS
```

## Usage Examples

### Filtering by NSFW Level

```typescript
// Get PG-rated models only
const filters = {
  timePeriod: 'AllTime',
  modelType: 'All',
  baseModel: 'All',
  sort: 'downloads',
  nsfwLevel: 'None',  // ← Safe for work
}

// Get R-rated checkpoints
const filters = {
  timePeriod: 'AllTime',
  modelType: 'Checkpoint',
  baseModel: 'All',
  sort: 'downloads',
  nsfwLevel: 'Mature',  // ← Mature content
}
```

### URL Structure

```
/models/civitai                    → PG (default)
/models/civitai/filter/pg          → PG (explicit)
/models/civitai/filter/pg13        → PG-13
/models/civitai/filter/r           → R
/models/civitai/filter/x           → X
/models/civitai/filter/xxx         → XXX
/models/civitai/filter/r/checkpoints → R + Checkpoints
```

## Next Steps (Optional)

1. **Add NSFW toggle in UI** - Quick toggle between PG and user's preferred rating
2. **User preferences** - Remember user's NSFW preference in localStorage
3. **More combinations** - Add NSFW filters for other popular combinations (e.g., `pg13/sdxl`, `r/loras/sdxl`)
4. **Image blurring** - Implement `blur_mature` functionality for preview images

## Rule Zero Compliance

✅ **No backwards compatibility shims** - Updated existing filter system directly  
✅ **No entropy** - Didn't create `nsfwFilter_v2` or similar  
✅ **Breaking changes** - Updated `CivitaiFilters` interface, compiler caught all issues  
✅ **Clean implementation** - Single source of truth for NSFW levels in contracts

---

**Result:** CivitAI marketplace now supports 5 NSFW filter levels with dedicated SSG pages for instant loading and SEO optimization.
