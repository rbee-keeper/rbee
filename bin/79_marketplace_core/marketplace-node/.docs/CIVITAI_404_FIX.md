# TEAM-460: CivitAI 404 Route Conflict Fix

## Problem

**URL:** `http://localhost:7823/models/civitai/month` returned 404

## Root Cause

**Next.js route priority conflict** between two dynamic routes:

```
/app/models/civitai/
├── [slug]/page.tsx          ← HIGHER PRIORITY (single segment)
└── [...filter]/page.tsx     ← LOWER PRIORITY (catch-all)
```

When visiting `/models/civitai/month`:
1. Next.js matches `[slug]` first (single-segment routes have higher priority)
2. The `[slug]` route expects model IDs like `civitai-12345-model-name`
3. `slugToModelId('month')` produces invalid model ID → 404

## Solution

**Added `filter/` prefix to all filter URLs** to avoid route conflicts:

### Before (Broken)
```
/models/civitai/month        → Caught by [slug] → 404
/models/civitai/week         → Caught by [slug] → 404
/models/civitai/checkpoints  → Caught by [slug] → 404
```

### After (Fixed)
```
/models/civitai/filter/month        → Caught by [...filter] → ✅
/models/civitai/filter/week         → Caught by [...filter] → ✅
/models/civitai/filter/checkpoints  → Caught by [...filter] → ✅
```

## Changes Made

### 1. Updated Filter Paths (`filters.ts`)
```typescript
// Before
{ timePeriod: 'Month', path: 'month' }

// After
{ timePeriod: 'Month', path: 'filter/month' }
```

All filter paths now have at least 2 segments (`filter/month`), so they're caught by `[...filter]` instead of `[slug]`.

### 2. Added Redirect for Old URLs (`[slug]/page.tsx`)
```typescript
const oldFilterKeywords = ['month', 'week', 'checkpoints', 'loras', 'sdxl', 'sd15']
if (oldFilterKeywords.includes(slug)) {
  redirect(`/models/civitai/filter/${slug}`)
}
```

This provides backwards compatibility for any old bookmarks/links.

## Testing

### New URLs (Working)
- ✅ `/models/civitai/filter/month` - Month filter
- ✅ `/models/civitai/filter/week` - Week filter
- ✅ `/models/civitai/filter/checkpoints` - Checkpoints only
- ✅ `/models/civitai/filter/month/checkpoints/sdxl` - Combined filters

### Old URLs (Redirected)
- `/models/civitai/month` → redirects to `/models/civitai/filter/month`
- `/models/civitai/week` → redirects to `/models/civitai/filter/week`

### Model URLs (Unchanged)
- ✅ `/models/civitai/civitai-12345-model-name` - Still works

## Files Modified
- `frontend/apps/marketplace/app/models/civitai/filters.ts` (updated paths)
- `frontend/apps/marketplace/app/models/civitai/[slug]/page.tsx` (added redirect)

## Next.js Route Priority Rules

For future reference, Next.js App Router prioritizes routes:
1. **Static routes** (most specific)
2. **Dynamic single-segment** `[param]` 
3. **Catch-all** `[...param]`
4. **Optional catch-all** `[[...param]]`

**Single-segment dynamic routes ALWAYS win over catch-all routes for single segments.**

## Alternative Solutions Considered

1. **Use optional catch-all** `[[...segments]]` - Requires major restructuring
2. **Rename [slug] to [model-slug]** - Breaks existing URLs
3. **Add filter/ prefix** ✅ - **Chosen (cleanest, preserves model URLs)**

## Build Command
```bash
cd frontend/apps/marketplace
pnpm build
```

The build pre-generates all filter pages with the new URL structure.
