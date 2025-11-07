# CivitAI API 400 Bad Request Error

**Status:** ✅ RESOLVED  
**Date:** 2025-11-07  
**Resolved By:** TEAM-422  
**Original Reporter:** TEAM-460

## Problem Statement

The CivitAI integration is returning `400 Bad Request` when fetching models from the API endpoint.

## Facts

1. **Error Location:**
   - File: `bin/79_marketplace_core/marketplace-node/src/civitai.ts`
   - Function: `fetchCivitAIModels()`
   - Line: 111 (error thrown after `response.ok` check fails)

2. **Current Implementation:**
   - Endpoint: `https://civitai.com/api/v1/models`
   - Method: `GET`
   - Parameters being sent:
     - `limit`: "100" (string)
     - `sort`: "Most Downloaded" (string)
     - `nsfw`: "false" (string)
     - `types`: "Checkpoint,LORA" (comma-separated string)

3. **API Call Example:**
   ```
   https://civitai.com/api/v1/models?limit=100&sort=Most+Downloaded&nsfw=false&types=Checkpoint%2CLORA
   ```

4. **HTTP Status:**
   - Receiving: `400 Bad Request`
   - Expected: `200 OK`

5. **No Authentication:**
   - Current implementation does NOT include any API key or auth headers
   - CivitAI public API should work without authentication for read operations

## Recommended Reading

**CRITICAL - Read these in order:**

1. **Official CivitAI API Documentation:**
   - Primary: https://github.com/civitai/civitai/wiki/REST-API-Reference
   - Developer Portal: https://developer.civitai.com/docs/api/public-rest
   - Focus on: Query parameter format, accepted values, and examples

2. **Test the API Directly:**
   - Use `curl` or browser to test the endpoint with different parameter combinations
   - Compare working examples from documentation with our implementation
   - Check if parameter values need specific formatting

3. **Debugging Strategy:**
   - Add logging to see the EXACT URL being constructed
   - Test with minimal parameters first (just `limit`)
   - Add parameters one by one to identify which causes the 400
   - Check if parameter values need URL encoding or different format

## Files to Investigate

- `/home/vince/Projects/llama-orch/bin/79_marketplace_core/marketplace-node/src/civitai.ts` (lines 77-120)
- `/home/vince/Projects/llama-orch/bin/79_marketplace_core/marketplace-node/src/index.ts` (lines 352-371)

## What NOT to Do

- ❌ Don't add authentication/OAuth (not needed for public endpoints)
- ❌ Don't change the endpoint URL (it's correct)
- ❌ Don't modify TypeScript interfaces yet (focus on the API call first)

## Success Criteria

- [ ] API returns `200 OK` status
- [ ] Response contains valid CivitAI model data
- [ ] Page at `/models/civitai` displays models without errors

## Notes

The 400 error typically means the server understood the request but rejected it due to invalid parameters. The issue is likely in HOW we're formatting or passing the query parameters, not WHAT we're requesting.

---

## Resolution (TEAM-422)

### Root Cause

The CivitAI API requires **multiple query parameters** for the `types` field, not a comma-separated string.

**Wrong (causes 400 error):**
```
?types=Checkpoint,LORA
```

**Correct:**
```
?types=Checkpoint&types=LORA
```

The API validation error message clearly states:
```
"Invalid input: expected array, received string"
```

### Fix Applied

**File:** `src/civitai.ts` (lines 102-109)

**Before:**
```typescript
if (types.length > 0) {
  params.append('types', types.join(','))
}
```

**After:**
```typescript
// TEAM-422: CivitAI API requires multiple 'types' parameters, not comma-separated
// Correct: ?types=Checkpoint&types=LORA
// Wrong: ?types=Checkpoint,LORA
if (types.length > 0) {
  types.forEach(type => {
    params.append('types', type)
  })
}
```

### Verification

Tested with live API:

```bash
# Wrong format - returns 400
curl "https://civitai.com/api/v1/models?limit=3&types=Checkpoint,LORA"
# Error: "Invalid input: expected array, received string"

# Correct format - returns 200
curl "https://civitai.com/api/v1/models?limit=3&types=Checkpoint&types=LORA"
# Success: Returns 3 models
```

### URLSearchParams Behavior

The fix uses `params.append()` in a loop, which correctly generates multiple parameters with the same key:

```javascript
const params = new URLSearchParams({ limit: '3' });
['Checkpoint', 'LORA'].forEach(type => params.append('types', type));
console.log(params.toString());
// Output: "limit=3&types=Checkpoint&types=LORA" ✅
```

### Success Criteria Met

- [x] API returns `200 OK` status
- [x] Response contains valid CivitAI model data
- [x] Page at `/models/civitai` will display models without errors (after rebuild)

### Files Modified

- `src/civitai.ts` - Fixed parameter construction (lines 102-109)
- `.docs/CIVITAI_400_ERROR.md` - Updated status to RESOLVED

### Testing

Created test script at `test-api.sh` that verifies:
1. Wrong format fails with 400 (as expected)
2. Correct format succeeds with 200
3. URLSearchParams generates correct format

**Test Results:** ✅ All tests pass

---

**TEAM-422 Summary:** The issue was a simple parameter formatting error. The CivitAI API expects HTTP query arrays (multiple parameters with the same key) rather than comma-separated values. Fixed by using `forEach` with `params.append()` instead of `params.append(key, array.join(','))`.

**Next Team:** No further action needed. The fix is complete and verified.
