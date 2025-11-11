# TEAM-476: Fail-Fast Filter Validation

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE

## Issue

Filters were silently falling back to defaults when invalid values were provided. No error messages, making debugging impossible.

## Solution: Fail-Fast Validation

Added comprehensive logging at every step of the filtering process to catch issues immediately.

### What Was Added

#### 1. URL Parameter Validation
```typescript
// TEAM-476: FAIL FAST - Log invalid filter values
if (periodParam && !(CIVITAI_TIME_PERIODS as readonly string[]).includes(periodParam)) {
  console.error(`❌ INVALID FILTER: period="${periodParam}" not in`, CIVITAI_TIME_PERIODS)
}
if (typeParam && !(CIVITAI_MODEL_TYPES as readonly string[]).includes(typeParam)) {
  console.error(`❌ INVALID FILTER: type="${typeParam}" not in`, CIVITAI_MODEL_TYPES)
}
// ... and so on for all filter types
```

#### 2. Filter Application Logging
```typescript
console.log('[Filter] URL params:', { periodParam, typeParam, baseParam, sortParam, nsfwParam })
console.log('[Filter] Applied filter:', currentFilter)
console.log('[Filter] Initial models count:', initialModels.length)
```

#### 3. Step-by-Step Filtering Logs
```typescript
// Before each filter
console.log('[Filter] Starting with', result.length, 'models')

// After model type filter
console.log(`[Filter] Model type filter: "${currentFilter.modelType}" → ${beforeCount} to ${result.length} models`)

// If no models after filtering
if (result.length === 0) {
  console.error(`❌ NO MODELS after type filter! Looking for type="${currentFilter.modelType}"`)
  console.error('Available types:', [...new Set(initialModels.map(m => m.type))])
}

// After NSFW filter
console.log(`[Filter] NSFW filter: "None" (PG only) → ${beforeCount} to ${result.length} models`)
console.log(`[Filter] NSFW breakdown: ${nsfwCount} NSFW, ${safeCount} safe`)

// After sorting
console.log(`[Filter] After sorting by "${currentFilter.sort}": ${result.length} models`)

// Final result
console.log('[Filter] FINAL RESULT:', result.length, 'models')
```

## How to Debug Filters

### 1. Open Browser Console
Press F12 or right-click → Inspect → Console

### 2. Navigate to Filtered Page
Example: `http://localhost:7823/models/civitai?type=Checkpoint&nsfw=None`

### 3. Check Console Output

**Valid Filter:**
```
[Filter] URL params: { typeParam: 'Checkpoint', nsfwParam: 'None', ... }
[Filter] Applied filter: { modelType: 'Checkpoint', nsfwLevel: 'None', ... }
[Filter] Initial models count: 100
[Filter] Starting with 100 models
[Filter] Model type filter: "Checkpoint" → 100 to 70 models
[Filter] NSFW filter: "None" (PG only) → 70 to 45 models
[Filter] After sorting by "Most Downloads": 45 models
[Filter] FINAL RESULT: 45 models
```

**Invalid Filter:**
```
❌ INVALID FILTER: type="Checkpoints" not in ['All', 'Checkpoint', 'LORA']
[Filter] Applied filter: { modelType: 'All', ... }  ← Fell back to default!
```

**No Models After Filter:**
```
[Filter] Model type filter: "Checkpoint" → 100 to 0 models
❌ NO MODELS after type filter! Looking for type="Checkpoint"
Available types: ['LORA', 'TextualInversion', 'Hypernetwork']  ← Wrong data!
```

## Common Issues You'll Catch

### 1. Invalid Filter Values
- **Error:** `❌ INVALID FILTER: type="Checkpoints" not in ...`
- **Cause:** Typo in URL parameter (should be `Checkpoint` not `Checkpoints`)
- **Fix:** Use exact values from the allowed list

### 2. Wrong Field Being Filtered
- **Error:** `❌ NO MODELS after type filter!` + `Available types: ['Unknown']`
- **Cause:** Model `type` field not being populated correctly
- **Fix:** Check SSR data normalization in `page.tsx`

### 3. NSFW Data Missing
- **Log:** `NSFW breakdown: 0 NSFW, 0 safe`
- **Cause:** `nsfw` field not being passed from API
- **Fix:** Check `convertCivitAIModel()` in marketplace-node

### 4. Filter Falling Back to Default
- **Log:** URL has `?type=Checkpoint` but Applied filter shows `modelType: 'All'`
- **Cause:** Invalid parameter value or validation logic error
- **Fix:** Check console errors for validation failures

## Files Modified

1. **`/frontend/apps/marketplace/app/models/civitai/CivitAIFilterPage.tsx`**
   - Added URL parameter validation with error logging
   - Added step-by-step filter application logging
   - Added "no models" detection with available values logging

## Benefits

✅ **Immediate feedback** - See exactly what's wrong in console  
✅ **No silent failures** - Invalid filters log errors  
✅ **Debug visibility** - See every step of filtering process  
✅ **Data validation** - Catch missing fields immediately  
✅ **Faster debugging** - No more guessing what went wrong  

## Next Steps

1. Open browser console and test filters
2. Look for `❌` errors or unexpected fallbacks
3. Check if NSFW breakdown shows actual NSFW vs safe counts
4. Verify model types match what you expect

---

**FAIL FAST > Silent failures**  
**Console errors > Guessing what went wrong**
