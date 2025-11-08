# TEAM-457: Marketplace Navigation Fix

**Status:** ✅ FIXED  
**Date:** Nov 7, 2025

## Problems Fixed

### 1. ❌ Hardcoded "Back to rbee.dev" URL
**File:** `frontend/apps/marketplace/components/MarketplaceNav.tsx`

**Before:**
```tsx
<Link href="https://rbee.dev">  ❌ Hardcoded production URL
  Back to rbee.dev
</Link>
```

**After:**
```tsx
<Link href={urls.commercial}>  ✅ Uses environment variable
  Back to rbee.dev
</Link>
```

### 2. ❌ Wrong Availability Status

**Before:**
- ✅ LLM Models - Available (correct)
- ❌ SD Models - "Coming soon" (WRONG - they're available!)
- ✅ LLM Workers - Available (WRONG - they're coming soon!)
- ✅ Image Workers - "Coming soon" (correct)

**After:**
- ✅ LLM Models - Available
- ✅ SD Models - Available (removed "Soon" badge, made clickable)
- ✅ LLM Workers - Coming soon (added "Soon" badge, disabled)
- ✅ Image Workers - Coming soon (kept "Soon" badge, disabled)

### 3. ❌ Other Hardcoded URLs

**Before:**
```tsx
<Link href="https://github.com/veighnsche/llama-orch/tree/main/docs">  ❌
<a href="https://github.com/veighnsche/llama-orch">  ❌
```

**After:**
```tsx
<Link href={urls.github.docs}>  ✅
<a href={urls.github.repo}>  ✅
```

## Files Created/Modified

### Created
1. ✅ `frontend/apps/marketplace/lib/env.ts` - Environment configuration (NEW)

### Modified
2. ✅ `frontend/apps/marketplace/components/MarketplaceNav.tsx` - Fixed all URLs and availability

## Environment Configuration

### New lib/env.ts
```typescript
export const env = {
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL || 'https://rbee.dev',
  githubUrl: process.env.NEXT_PUBLIC_GITHUB_URL || 'https://github.com/veighnsche/llama-orch',
  docsUrl: process.env.NEXT_PUBLIC_DOCS_URL || 'https://docs.rbee.dev',
}

export const urls = {
  commercial: env.siteUrl,
  github: {
    repo: env.githubUrl,
    docs: `${env.githubUrl}/tree/main/docs`,
  },
  docs: env.docsUrl,
}
```

### Environment Variables (from .env.local.example)
```bash
# Production (default)
NEXT_PUBLIC_SITE_URL=https://rbee.dev

# Development (uncomment for local dev)
# NEXT_PUBLIC_SITE_URL=http://localhost:7822
```

## Changes Summary

### MarketplaceNav.tsx

**Imports:**
```diff
+ import { urls } from '@/lib/env'
```

**SD Models (Lines 50-57):**
```diff
- <Link className="...cursor-not-allowed" onClick={(e) => e.preventDefault()}>
-   SD Models
-   <span>Soon</span>
- </Link>
+ <Link href="/models?type=sd" className="...hover:text-foreground">
+   SD Models
+ </Link>
```

**LLM Workers (Lines 65-73):**
```diff
- <Link href="/workers" className="...hover:text-foreground">
-   LLM Workers
- </Link>
+ <Link className="...cursor-not-allowed" onClick={(e) => e.preventDefault()}>
+   LLM Workers
+   <span>Soon</span>
+ </Link>
```

**Back to rbee.dev (Line 120):**
```diff
- <Link href="https://rbee.dev">
+ <Link href={urls.commercial}>
```

**GitHub Docs (Line 94):**
```diff
- <Link href="https://github.com/veighnsche/llama-orch/tree/main/docs">
+ <Link href={urls.github.docs}>
```

**GitHub Repo (Line 103):**
```diff
- <a href="https://github.com/veighnsche/llama-orch">
+ <a href={urls.github.repo}>
```

## Correct Status

### ✅ AVAILABLE NOW
- **LLM Models** - Clickable, no badge
- **SD Models** - Clickable, no badge

### 🔜 COMING SOON
- **LLM Workers** - Disabled, "Soon" badge
- **Image Workers** - Disabled, "Soon" badge

## Verification

After restart:

1. **Check "Back to rbee.dev" link:**
   - Development: Should go to `http://localhost:7822`
   - Production: Should go to `https://rbee.dev`

2. **Check Models section:**
   - LLM Models: Clickable ✅
   - SD Models: Clickable ✅

3. **Check Workers section:**
   - LLM Workers: Disabled with "Soon" badge ✅
   - Image Workers: Disabled with "Soon" badge ✅

## Summary

✅ **Created lib/env.ts** for marketplace  
✅ **Fixed 3 hardcoded URLs** (back to rbee.dev, GitHub docs, GitHub repo)  
✅ **Fixed SD Models** - Now available (removed "Soon" badge)  
✅ **Fixed LLM Workers** - Now coming soon (added "Soon" badge)  
✅ **All URLs use environment variables**  

**Marketplace navigation now shows correct availability and uses environment variables!** 🚀
