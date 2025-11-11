# TEAM-471: CivitAI API Resilience Improvements

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Problem:** Manifest generation failing due to CivitAI API 502/504/524 errors

## Problem Analysis

### Root Causes Identified

1. **504 Gateway Timeout** - CivitAI servers overloaded during bulk requests
2. **502 Bad Gateway** - Transient API issues during high load
3. **524 Cloudflare Timeout** - Cloudflare timing out before CivitAI responds
4. **FAIL FAST Policy** - Script exited immediately on ANY error, leaving manifests empty
5. **No Retry Logic** - Single failure = complete script failure

### Investigation Process

1. Reviewed CivitAI API documentation (no documented rate limits)
2. Tested failing filter individually (`filter/x/week/sd21`) - **worked fine**
3. Confirmed errors were **transient** (API overload, not invalid requests)
4. Identified FAIL FAST as major issue (299 filters, 1 failure = 0 manifests)

## Solutions Implemented

### 1. ✅ Retry Logic with Exponential Backoff

**File:** `/bin/79_marketplace_core/marketplace-node/src/civitai/civitai.ts`

**Changes:**
- Added `fetchWithRetry()` function
- Retries up to 3 times on: **502, 504, 524, 429** status codes
- Exponential backoff: 1s → 2s → 4s (max 10s)
- Applied to both `fetchCivitAIModels()` and `fetchCivitAIModel()`

**Code:**
```typescript
async function fetchWithRetry(url: string, maxRetries = 3): Promise<Response> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(url)
      
      // Retry on 502 Bad Gateway, 504 Gateway Timeout, 524 Cloudflare Timeout, or 429 Too Many Requests
      if (response.status === 502 || response.status === 504 || response.status === 524 || response.status === 429) {
        const delay = Math.min(1000 * Math.pow(2, i), 10000) // Exponential backoff, max 10s
        console.log(`  ⏳ CivitAI ${response.status} - Retry ${i + 1}/${maxRetries} after ${delay}ms...`)
        await new Promise(resolve => setTimeout(resolve, delay))
        continue
      }
      
      return response
    } catch (error) {
      // Network error - retry with exponential backoff
      if (i === maxRetries - 1) throw error
      const delay = Math.min(1000 * Math.pow(2, i), 10000)
      console.log(`  ⏳ Network error - Retry ${i + 1}/${maxRetries} after ${delay}ms...`)
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }
  throw new Error(`Max retries (${maxRetries}) exceeded`)
}
```

### 2. ✅ Balanced Rate Limiting

**File:** `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`

**Changes:**
- **Before:** 3 concurrent, 100ms delay (too aggressive → 504 errors)
- **Avoided:** 1 concurrent, 500ms delay (too slow → 2.5 minutes for CivitAI alone)
- **Optimal:** 2 concurrent, 200ms delay (~30-40 seconds total)

**Code:**
```typescript
// TEAM-471: Balanced approach - 2 concurrent with 200ms delay (retry logic handles 504s)
const limiter = new RateLimiter(2, 200)
```

### 3. ✅ FAIL GRACEFULLY Instead of FAIL FAST

**File:** `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`

**Changes:**
- **Before:** `process.exit(1)` on ANY error → empty manifests
- **After:** Log error, track failure, continue with other filters
- **Result:** Partial manifests better than no manifests

**Code:**
```typescript
// TEAM-471: Track failures for summary
const failedCivitAIFilters: string[] = []

const civitaiPromises = CIVITAI_FILTERS.map((filter) =>
  limiter.run(async () => {
    try {
      const models = await fetchCivitAIModelsViaSDK(filter)
      console.log(`✅ CivitAI ${filter}: ${models.length} models`)
      // ... process models ...
      return { filter, count: models.length, success: true }
    } catch (error) {
      // FAIL GRACEFULLY - Log error but continue
      console.error(`⚠️  CivitAI ${filter} FAILED:`, error instanceof Error ? error.message : error)
      failedCivitAIFilters.push(filter)
      return { filter, count: 0, success: false }
    }
  }),
)
```

### 4. ✅ Failure Summary Report

**File:** `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`

**Changes:**
- Added failure tracking and summary at end
- Shows which filters failed (for debugging)
- Clarifies that manifests were generated with available data

**Code:**
```typescript
// TEAM-471: Report failures
if (failedCivitAIFilters.length > 0) {
  console.log(`\n⚠️  Failed CivitAI filters (${failedCivitAIFilters.length}):`)
  for (const f of failedCivitAIFilters) {
    console.log(`     - ${f}`)
  }
  console.log(`\n💡 These filters will be retried on next run. Manifests generated with available data.`)
}
```

### 5. ✅ Debug Script for Testing Individual Filters

**File:** `/frontend/apps/marketplace/scripts/test-single-filter.ts`

**Purpose:** Test individual filters to isolate issues

**Usage:**
```bash
pnpm tsx scripts/test-single-filter.ts "filter/x/week/sd21"
```

## Performance Impact

### Before (FAIL FAST)
- ❌ 1 API error = Script exits
- ❌ 0 manifests generated
- ❌ No visibility into which filter failed
- ❌ No retry on transient errors

### After (FAIL GRACEFULLY + Retry)
- ✅ API errors logged but script continues
- ✅ Manifests generated with available data
- ✅ Clear failure summary at end
- ✅ Automatic retry on 502/504/524/429 (up to 3 times)
- ✅ ~30-40 seconds for 299 CivitAI filters (balanced rate limiting)

### Estimated Time
- **299 CivitAI filters:** ~30-40 seconds (2 concurrent, 200ms delay)
- **HF filters:** ~10-15 seconds
- **Total:** ~45-60 seconds (vs 2.5 minutes with ultra-conservative approach)

## Error Handling Strategy

### Retryable Errors (Auto-Retry)
- **502 Bad Gateway** - CivitAI server error
- **504 Gateway Timeout** - CivitAI server overloaded
- **524 Cloudflare Timeout** - Cloudflare timeout
- **429 Too Many Requests** - Rate limiting
- **Network errors** - Connection issues

### Non-Retryable Errors (Log & Continue)
- **400 Bad Request** - Invalid filter (shouldn't happen)
- **404 Not Found** - Invalid endpoint (shouldn't happen)
- **Other 4xx/5xx** - Log and continue

## Testing

### Test Cases
1. ✅ Individual filter test: `filter/x/week/sd21` - **PASSED**
2. ✅ Full manifest generation with retry logic - **COMPLETE**
3. ✅ Graceful degradation on transient errors - **VERIFIED**

### Verification
```bash
# Test single filter
pnpm tsx scripts/test-single-filter.ts "filter/x/week/sd21"

# Run full generation
bash scripts/regenerate-manifests.sh
```

### Note on Console Output
The `[CivitAI] Fetching...` logs appear quickly because they're logged when promises are queued, not when they execute. The rate limiting is working correctly - you can verify this by watching the `✅ CivitAI filter: X models` success logs, which appear at the rate-limited pace (2 concurrent, 200ms delay between batches).

## Files Modified

1. `/bin/79_marketplace_core/marketplace-node/src/civitai/civitai.ts`
   - Added `fetchWithRetry()` with exponential backoff
   - Added 502, 524 to retry logic

2. `/frontend/apps/marketplace/scripts/generate-model-manifests.ts`
   - Changed rate limiter: 2 concurrent, 200ms delay
   - Removed FAIL FAST logic
   - Added graceful error handling
   - Added failure tracking and summary

3. `/frontend/apps/marketplace/scripts/test-single-filter.ts` (NEW)
   - Debug script for testing individual filters

## Lessons Learned

1. **FAIL FAST is not always best** - For bulk operations, graceful degradation is better
2. **Transient errors are common** - Always implement retry logic for external APIs
3. **Rate limiting is a balance** - Too aggressive = errors, too conservative = slow
4. **Visibility matters** - Clear error reporting helps debugging

## Next Steps (Optional)

1. Monitor failure rate in production
2. Consider adding CivitAI API token for better rate limits
3. Add metrics/telemetry for API performance
4. Consider caching successful responses

## Related Issues

- Original issue: Manifests directory empty due to 504 errors
- Root cause: FAIL FAST policy + no retry logic
- Solution: Retry logic + graceful degradation + balanced rate limiting
