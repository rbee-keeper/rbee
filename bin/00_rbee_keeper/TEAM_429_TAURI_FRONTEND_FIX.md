# TEAM-429: Tauri Frontend Fix

**Date:** 2025-01-XX  
**Status:** ✅ FIXED

## Problem

The Tauri frontend was calling `marketplace_list_civitai_models` with the old signature:

```typescript
// ❌ OLD (broken)
await invoke('marketplace_list_civitai_models', {
  limit: 100,
})
```

**Error:**
```
Error: invalid args `filters` for command `marketplace_list_civitai_models`: 
command marketplace_list_civitai_models missing required key filters
```

## Root Cause

The backend Tauri command was updated in Phase 5 to accept a `CivitaiFilters` object:

```rust
#[tauri::command]
pub async fn marketplace_list_civitai_models(
    filters: artifacts_contract::CivitaiFilters,  // ← Expects filters object
) -> Result<Vec<marketplace_sdk::Model>, String>
```

But the frontend was never updated to pass the new parameter format.

## Solution

Updated the Tauri frontend to pass a proper `filters` object with snake_case field names:

```typescript
// ✅ NEW (correct)
await invoke<Model[]>('marketplace_list_civitai_models', {
  filters: {
    time_period: 'AllTime',
    model_type: 'All',
    base_model: 'All',
    sort: 'Most Downloaded',
    nsfw: {
      max_level: 'None',
      blur_mature: true,
    },
    page: null,
    limit: 100,
  },
})
```

## Files Modified

- ✅ `bin/00_rbee_keeper/ui/src/pages/MarketplaceCivitai.tsx`

## Changes Made

### Before
```typescript
const result = await invoke<Model[]>('marketplace_list_civitai_models', {
  limit: 100,
})
```

### After
```typescript
const result = await invoke<Model[]>('marketplace_list_civitai_models', {
  filters: {
    time_period: 'AllTime',
    model_type: 'All',
    base_model: 'All',
    sort: 'Most Downloaded',
    nsfw: {
      max_level: 'None',
      blur_mature: true,
    },
    page: null,
    limit: 100,
  },
})
```

## Key Points

1. **Parameter name:** Changed from individual `limit` parameter to `filters` object
2. **Field names:** Using snake_case (Rust convention) - `time_period`, `model_type`, `base_model`
3. **NSFW filter:** Added required `nsfw` object with `max_level` and `blur_mature`
4. **Sort value:** Using API format `'Most Downloaded'` instead of frontend format `'downloads'`

## Verification

```bash
cargo build --bin rbee-keeper
# ✅ Build succeeds

./rbee
# ✅ Marketplace page should now load models
```

## Future Enhancement (Optional)

The current implementation uses hardcoded default filters. For a better UX, you could:

1. **Make filters reactive:** Update the query when frontend filters change
2. **Add filter UI:** Create FilterBar component (as documented in Phase 5)
3. **Persist filters:** Save user preferences to localStorage

**Example (reactive filters):**
```typescript
const {
  data: rawModels = [],
  isLoading,
  error,
} = useQuery({
  queryKey: ['marketplace', 'civitai-models', filters],  // ← Add filters to key
  queryFn: async () => {
    const result = await invoke<Model[]>('marketplace_list_civitai_models', {
      filters: {
        time_period: filters.timePeriod,
        model_type: filters.modelType,
        base_model: filters.baseModel,
        sort: convertSortToApi(filters.sort),
        nsfw: {
          max_level: 'None',
          blur_mature: true,
        },
        page: null,
        limit: 100,
      },
    })
    return result
  },
  staleTime: 5 * 60 * 1000,
})
```

## Related Documents

- `TEAM_429_FINAL_SUMMARY.md` - Complete integration review
- `TEAM_429_INTEGRATION_REVIEW.md` - Phase-by-phase analysis
- `TODO_PHASE_5_TAURI_GUI.md` - Original Phase 5 requirements

---

**TEAM-429:** Fixed Tauri frontend to pass correct `filters` object to backend command. The marketplace page should now load models successfully.
