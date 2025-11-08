# TEAM-422: SSG Compatibility Fix

**Status:** ✅ COMPLETE  
**Date:** 2025-11-07  
**Team:** TEAM-422

## Problem

ModelCard component used `useState` which caused runtime error in SSG pages:
```
useState only works in Client Components. Add the "use client" directive at the top of the file to use it.
```

## Root Cause

The ModelCard component had two pieces of client-side state:
1. `imageError` - Track if image failed to load
2. `isHovered` - Track hover state (unused)

These required client-side JavaScript, breaking SSG pre-rendering.

## Fix Applied

**File:** `frontend/packages/rbee-ui/src/marketplace/organisms/ModelCard/ModelCard.tsx`

### Removed useState

**Before:**
```typescript
export function ModelCard({ model, onAction, actionButton, onClick }: ModelCardProps) {
  const [imageError, setImageError] = React.useState(false)
  const [, setIsHovered] = React.useState(false)

  return (
    <Card 
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {model.imageUrl && !imageError ? (
        <img onError={() => setImageError(true)} />
      ) : (
        <FallbackGradient />
      )}
    </Card>
  )
}
```

**After:**
```typescript
export function ModelCard({ model, onAction, actionButton, onClick }: ModelCardProps) {
  // TEAM-422: Removed useState for SSG compatibility
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <Card>
      {model.imageUrl ? (
        <img loading="lazy" />
      ) : (
        <FallbackGradient />
      )}
    </Card>
  )
}
```

### Changes Made

1. **Removed `useState` calls** - No client-side state
2. **Removed `onMouseEnter/onMouseLeave`** - Hover effects still work via CSS
3. **Removed `onError` handler** - If image fails, browser handles it
4. **Simplified image logic** - Show image if URL exists, otherwise fallback

### Trade-offs

**Lost:**
- ❌ Automatic fallback to gradient if image fails to load
- ❌ Hover state tracking (was unused anyway)

**Kept:**
- ✅ All visual hover effects (via CSS `group-hover:`)
- ✅ Image lazy loading
- ✅ Fallback gradient when no imageUrl
- ✅ All styling and animations

**Gained:**
- ✅ SSG compatibility
- ✅ Faster page loads (no hydration needed)
- ✅ Better SEO (fully pre-rendered)
- ✅ Simpler component

## SSG Verification

### All Marketplace Pages are SSG ✅

**Main Pages:**
- ✅ `/` - Homepage (SSG)
- ✅ `/models` - Models hub redirect (SSG)
- ✅ `/models/huggingface` - HF models list (SSG)
- ✅ `/models/civitai` - CivitAI models list (SSG)
- ✅ `/models/huggingface/[slug]` - HF model detail (SSG)
- ✅ `/models/civitai/[slug]` - CivitAI model detail (SSG)
- ✅ `/workers` - Workers list (SSG)
- ✅ `/workers/[workerId]` - Worker detail (SSG)

**Client-Side Pages (Intentional):**
- ⚠️ `/search` - Dynamic search (requires "use client")

### Client Components (Used in SSG Pages)

These are fine - they add interactivity to pre-rendered pages:

1. **ModelTableWithRouting** - Client wrapper for navigation
   - Page is SSG, component adds routing
   - No data fetching, just navigation

2. **useKeeperInstalled** - Hook for install detection
   - Used in detail pages
   - Doesn't affect SSG rendering

## Architecture

```
┌─────────────────────────────────────────┐
│ SSG Page (Server-Rendered)              │
│                                          │
│  - Fetch data at build time             │
│  - Render ModelCard components          │
│  - Generate static HTML                 │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ Client Component (Hydrated)        │ │
│  │                                    │ │
│  │  - ModelTableWithRouting           │ │
│  │  - Adds navigation handlers        │ │
│  │  - No data fetching                │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

## Benefits of SSG

1. **Performance**
   - Pages load instantly (pre-rendered HTML)
   - No JavaScript required for initial render
   - Faster Time to First Byte (TTFB)

2. **SEO**
   - Search engines see full content immediately
   - No waiting for client-side rendering
   - Better crawlability

3. **User Experience**
   - Content visible immediately
   - Progressive enhancement
   - Works without JavaScript

4. **Cost**
   - Static files are cheap to serve
   - Can use CDN for global distribution
   - No server-side rendering overhead

## Testing

### Verify SSG Build

```bash
cd frontend/apps/marketplace
pnpm build

# Check output - should see:
# ○ /models/civitai (SSG)
# ○ /models/huggingface (SSG)
# ● /search (Client)
```

### Verify No Runtime Errors

1. Visit `/models/civitai`
2. Check browser console - no errors
3. Cards should render correctly
4. Hover effects should work (CSS-based)

## Files Modified

1. **frontend/packages/rbee-ui/src/marketplace/organisms/ModelCard/ModelCard.tsx**
   - Removed `useState` calls (lines 34-35)
   - Removed `onMouseEnter/onMouseLeave` handlers (lines 46-47)
   - Removed `onError` handler (line 58)
   - Simplified image conditional (line 51)

## Success Criteria

- [x] No useState in ModelCard
- [x] No runtime errors on CivitAI page
- [x] Cards render correctly
- [x] Hover effects work (CSS)
- [x] All main pages are SSG
- [x] Only search page is client-side

## Future Enhancements

If we need image error handling in the future:

1. **Option 1:** Use a client wrapper component
   ```tsx
   'use client'
   export function ModelCardWithErrorHandling(props) {
     const [imageError, setImageError] = useState(false)
     return <ModelCard {...props} imageUrl={imageError ? undefined : props.imageUrl} />
   }
   ```

2. **Option 2:** Server-side image validation
   - Check image URLs at build time
   - Only include valid URLs in SSG data

3. **Option 3:** Use Next.js Image component
   - Built-in error handling
   - Automatic optimization
   - But requires configuration

## Recommendation

**Keep it simple.** The current approach works well:
- SSG pages load fast
- Cards look great
- No client-side complexity
- If an image fails, browser shows broken image icon (acceptable)

---

**TEAM-422** - Removed useState from ModelCard for SSG compatibility. All marketplace pages are now fully pre-rendered for maximum performance and SEO.
