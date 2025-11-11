# TEAM-467: FAIL FAST Fix for Manifest Generation

**Related:** See [TEAM_467_FILTER_CENTRALIZATION.md](./TEAM_467_FILTER_CENTRALIZATION.md) for filter constant consolidation.

**Problem:** The `regenerate-manifests.sh` script was NOT failing fast. It continued running through all filters even when errors occurred, showing the same error repeatedly.

## Root Causes

### 1. SDK Swallowing Errors (marketplace-node)
**File:** `/bin/79_marketplace_core/marketplace-node/src/index.ts`

The `getCompatibleCivitaiModels()` function was catching errors and returning empty arrays instead of throwing:

```typescript
// ❌ BEFORE - Swallowed errors
try {
  const civitaiModels = await fetchCivitAIModels(mergedFilters)
  return civitaiModels.map(convertCivitAIModel)
} catch (error) {
  console.error('[marketplace-node] Failed to fetch CivitAI models:', error)
  return []  // <-- Returns empty array, script continues
}

// ✅ AFTER - Throws errors
const civitaiModels = await fetchCivitAIModels(mergedFilters)
return civitaiModels.map(convertCivitAIModel)
// Let errors propagate to caller
```

### 2. Invalid NSFW Level
**Files:** 
- `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`
- `/bin/79_marketplace_core/marketplace-node/src/civitai.ts`

The script was using `max_level: 'All'` which is not a valid `NsfwLevel`. Valid values are: `None`, `Soft`, `Mature`, `X`, `XXX`.

This caused `nsfwLevels.forEach()` to crash with `Cannot read properties of undefined`.

**Fix:**
```typescript
// ✅ Use 'XXX' for all NSFW levels
nsfw: {
  max_level: 'XXX',  // Includes all levels: [1, 2, 4, 8, 16]
  blur_mature: false,
}
```

### 3. Missing Validation
**File:** `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`

Added validation to check SDK return values:

```typescript
const models = await getCompatibleCivitaiModels(filters)

// FAIL FAST: SDK logs errors but doesn't throw - check return value
if (!models || !Array.isArray(models)) {
  console.error(`❌ FATAL: CivitAI SDK returned invalid data for ${filter}`)
  console.error(`💥 ABORTING: Check error logs above`)
  process.exit(1)
}
```

## Changes Made

### `/bin/79_marketplace_core/marketplace-node/src/index.ts`
- **Removed try-catch** from `getCompatibleCivitaiModels()` - let errors propagate
- Added comment explaining why we don't swallow errors

### `/bin/79_marketplace_core/marketplace-node/src/civitai.ts`
- **Added validation** for NSFW level before lookup
- **Better error message** with valid values and hints

### `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`
- **Fixed NSFW level**: Changed `'All'` → `'XXX'`
- **Added SDK response validation** in both `fetchCivitAIModelsViaSDK()` and `fetchHFModelsViaSDK()`
- **Exit immediately** if SDK returns invalid data

## Behavior Now

✅ **Script exits immediately** on first error
✅ **Clear error messages** showing what went wrong
✅ **No repeated errors** - fails fast before processing more filters

## Rebuild Required

After making changes to `marketplace-node`, you must rebuild:

```bash
cd /home/vince/Projects/rbee/bin/79_marketplace_core/marketplace-node
pnpm run build
```

This compiles TypeScript → JavaScript in `dist/` which the script uses.

## Testing

Run the script - it should now STOP on first error:

```bash
cd /home/vince/Projects/rbee/frontend/apps/marketplace
./scripts/regenerate-manifests.sh
```

If it still shows errors without stopping, check:
1. Did you rebuild marketplace-node?
2. Is the error coming from a different source?
3. Check the stack trace to identify where the error originates
