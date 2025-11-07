# TEAM-422: Before/After Comparison

## The Bug

### ❌ Before (Broken)

**Code:**
```typescript
if (types.length > 0) {
  params.append('types', types.join(','))
}
```

**Generated URL:**
```
https://civitai.com/api/v1/models?limit=100&sort=Most+Downloaded&nsfw=false&types=Checkpoint%2CLORA
```

**URL Decoded:**
```
?types=Checkpoint,LORA
```

**API Response:**
```json
{
  "error": {
    "code": "invalid_union",
    "message": "Invalid input: expected array, received string"
  }
}
```

**HTTP Status:** `400 Bad Request`

---

## The Fix

### ✅ After (Fixed)

**Code:**
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

**Generated URL:**
```
https://civitai.com/api/v1/models?limit=100&sort=Most+Downloaded&nsfw=false&types=Checkpoint&types=LORA
```

**URL Decoded:**
```
?types=Checkpoint&types=LORA
```

**API Response:**
```json
{
  "items": [
    {
      "id": 257749,
      "name": "Pony Diffusion V6 XL",
      "type": "Checkpoint",
      "stats": {
        "downloadCount": 1234567
      }
    }
  ],
  "metadata": {
    "totalItems": 1676,
    "currentPage": 1,
    "pageSize": 100
  }
}
```

**HTTP Status:** `200 OK`

---

## Visual Comparison

### Parameter Format

```
❌ WRONG:  types=Checkpoint,LORA
           └─ Single string parameter with comma

✅ CORRECT: types=Checkpoint&types=LORA
            └─ Multiple parameters with same key
```

### URLSearchParams Behavior

```javascript
// ❌ WRONG APPROACH
const params = new URLSearchParams();
params.append('types', ['Checkpoint', 'LORA'].join(','));
console.log(params.toString());
// Output: "types=Checkpoint%2CLORA"
//         (comma is URL-encoded as %2C)

// ✅ CORRECT APPROACH
const params = new URLSearchParams();
['Checkpoint', 'LORA'].forEach(type => {
  params.append('types', type);
});
console.log(params.toString());
// Output: "types=Checkpoint&types=LORA"
//         (multiple parameters with same key)
```

---

## Why This Matters

### HTTP Query Array Standard

Many REST APIs use **repeated query parameters** for arrays:

```
✅ Standard:  ?tags=red&tags=blue&tags=green
❌ Non-standard: ?tags=red,blue,green
```

CivitAI follows this standard. The API validates `types` as an array and expects:
- Multiple query parameters with the same key
- NOT a comma-separated string

### URLSearchParams.append() vs set()

```javascript
// append() - Adds a new value (allows duplicates)
params.append('types', 'Checkpoint');
params.append('types', 'LORA');
// Result: types=Checkpoint&types=LORA ✅

// set() - Replaces existing value (no duplicates)
params.set('types', 'Checkpoint');
params.set('types', 'LORA');
// Result: types=LORA ❌ (Checkpoint was overwritten)
```

---

## Impact

### Before Fix
- ❌ CivitAI API returns 400 error
- ❌ Marketplace page shows no models
- ❌ Users cannot browse CivitAI catalog

### After Fix
- ✅ CivitAI API returns 200 success
- ✅ Marketplace page displays models
- ✅ Users can browse Checkpoint and LORA models

---

## Verification

Run the verification script:

```bash
cd bin/79_marketplace_core/marketplace-node
./test-api.sh
```

Expected output:
```
❌ FAILED (as expected): Got error response
✅ SUCCESS: Got valid response
   Fetched 3 models
   First model: Pony Diffusion V6 XL (type: Checkpoint)
```

---

**TEAM-422** - Simple fix, big impact. 8 lines of code changed.
