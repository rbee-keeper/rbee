# Slug System Fix

**Date:** Nov 4, 2025  
**Issue:** 404 errors on model detail pages  
**Root Cause:** Double-dash separator being collapsed to single dash

## Problem

The slug conversion was collapsing `--` (double dash) into `-` (single dash):

```
google/electra-base-discriminator 
  → google-electra-base-discriminator  ❌ (single dash, no separator)
```

This happened because the order of operations was wrong:
1. Replace `/` with `--` → `google--electra-base-discriminator`
2. Collapse multiple dashes → `google-electra-base-discriminator` ❌ (collapsed the separator!)

## Solution

**Fixed order of operations** - preserve slashes until after dash collapsing:

```typescript
export function modelIdToSlug(modelId: string): string {
  return modelId
    .toLowerCase()
    .replace(/[^a-z0-9/-]/g, '-')  // Replace non-alphanumeric (EXCEPT slash)
    .replace(/-+/g, '-')           // Collapse multiple dashes
    .replace(/\//g, '--')          // Replace slashes AFTER collapsing
    .replace(/^-|-$/g, '')         // Remove leading/trailing dashes
}
```

Now it works correctly:
```
google/electra-base-discriminator 
  → google--electra-base-discriminator  ✅ (double dash preserved)
```

## Verification

All URLs now return 200:

```bash
✅ /models/google--electra-base-discriminator (200)
✅ /models/sentence-transformers--all-minilm-l6-v2 (200)
✅ /models/meta-llama--llama-2-7b-chat-hf (200)
✅ /models (200)
```

## Key Insight

**Order matters in string transformations!**

When you have a multi-step transformation:
1. Protect special markers (like `/`) from intermediate steps
2. Apply the marker replacement LAST
3. This prevents intermediate steps from destroying your markers

## Files Changed

- `lib/slugify.ts` - Fixed `modelIdToSlug()` function order of operations

## Result

✅ All model detail pages now work with clean URLs  
✅ Double-dash separator preserved  
✅ SEO-friendly slugs working correctly  
✅ No more 404 errors  

The slug system is now fully functional!
