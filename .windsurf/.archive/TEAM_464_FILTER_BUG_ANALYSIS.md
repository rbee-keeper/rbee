# TEAM-464: Filter Component Bug Analysis

**Date**: 2025-11-11  
**Status**: ✅ ALL BUGS FIXED - Filter component fully working with multiple params

## Bugs Found

### ✅ Bug 1: CategoryFilterBar Used `window.location.href` Instead of Callback
**File**: `frontend/packages/rbee-ui/src/marketplace/organisms/CategoryFilterBar/CategoryFilterBar.tsx`  
**Line**: 47-49

**Root Cause:**
```typescript
if (!isTauri && url !== '#') {
  // Next.js: Navigate to URL
  window.location.href = url  // ❌ Full page reload!
}
```

Filter clicks triggered full page reloads instead of using the `onFilterChange` callback, breaking the client-side filtering architecture.

**Fix Applied:**
```typescript
if (onFilterChange) {
  // Use callback to update state (works for both Tauri and Next.js)
  onFilterChange(option.value)
} else if (url !== '#') {
  // Fallback: Navigate to URL (full page reload)
  window.location.href = url
}
```

**Status**: ✅ FIXED - Filter clicks now use client-side navigation

---

### ✅ Bug 2: Sort Group Missing `onFilterChange` Callback
**File**: `frontend/packages/rbee-ui/src/marketplace/organisms/CategoryFilterBar/CategoryFilterBar.tsx`  
**Line**: 119-126

**Root Cause:**
Sort dropdown didn't receive the `onFilterChange` prop, so it couldn't use client-side navigation.

**Fix Applied:**
```typescript
<FilterGroupComponent
  key={sortGroup.id}
  group={sortGroup}
  currentValue={(currentFilters as Record<string, string>)[sortGroup.id] || sortGroup.options[0]?.value}
  buildUrl={(value) => buildUrl({ [sortGroup.id]: value } as Partial<T>)}
  onFilterChange={onFilterChange ? (value) => onFilterChange({ [sortGroup.id]: value } as Partial<T>) : undefined}
/>
```

**Status**: ✅ FIXED - Sort filter now works like other filters

---

### ✅ Bug 3: Infinite Loop with Multiple Search Params - FIXED!
**File**: `frontend/apps/marketplace/app/models/huggingface/HFFilterPage.tsx`  
**Behavior**: Single filters worked (`?size=small`), but adding a second filter caused infinite re-renders

**Evidence:**
```
[error] Failed to fetch RSC payload for http://localhost:7823/models/huggingface?sort=likes&size=small
RangeError: Maximum call stack size exceeded
```

**Console Output**: Hundreds of `[SSG] Fetching top 100 HuggingFace models for initial render` messages

**Root Cause Analysis:**

After researching Next.js documentation and GitHub issues, I discovered the bug:

1. **Using relative URLs with `router.push()`** - We were calling `router.push(?${queryString})` which Next.js treats as a SERVER navigation, causing full RSC re-renders
2. **Missing `usePathname()`** - Official Next.js pattern requires building full URLs: `pathname + ? + queryString`
3. **No `useCallback`** - The `handleFilterChange` function was recreated on every render, breaking React's diffing algorithm
4. **Missing `scroll: false` option** - Without this, Next.js scrolls to top on every filter change

**Solution Applied:**

Following the [official Next.js documentation](https://nextjs.org/docs/app/api-reference/functions/use-search-params), I rewrote `handleFilterChange`:

```typescript
import { useSearchParams, useRouter, usePathname } from 'next/navigation'

// Inside component:
const pathname = usePathname()

// TEAM-464: Handle filter changes - following Next.js official pattern
const handleFilterChange = useCallback((newFilters: Partial<Record<string, string>>) => {
  // Build new filter state by merging current + new
  const currentSort = searchParams.get('sort') || initialFilter.sort
  const currentSize = searchParams.get('size') || initialFilter.size
  const currentLicense = searchParams.get('license') || initialFilter.license
  
  const merged = {
    sort: (newFilters.sort as any) || currentSort,
    size: (newFilters.size as any) || currentSize,
    license: (newFilters.license as any) || currentLicense,
  }
  
  // Build URL params from searchParams.toString() to preserve other params
  const params = new URLSearchParams(searchParams.toString())
  
  // Update or delete each filter param
  if (merged.sort && merged.sort !== 'downloads') {
    params.set('sort', merged.sort)
  } else {
    params.delete('sort')  // Remove default values
  }
  
  if (merged.size && merged.size !== 'all') {
    params.set('size', merged.size)
  } else {
    params.delete('size')
  }
  
  if (merged.license && merged.license !== 'all') {
    params.set('license', merged.license)
  } else {
    params.delete('license')
  }
  
  // CRITICAL: Build full URL with pathname (fixes infinite loop)
  // Don't use relative URLs like "?query" - Next.js treats them as server navigations
  const queryString = params.toString()
  const newUrl = queryString ? `${pathname}?${queryString}` : pathname
  
  router.push(newUrl, { scroll: false })
}, [searchParams, pathname, router, initialFilter])
```

**Key Changes:**
1. ✅ Import `usePathname()` and `useCallback`
2. ✅ Wrap `handleFilterChange` in `useCallback` with proper dependencies
3. ✅ Use `searchParams.toString()` to preserve existing params
4. ✅ Build full URL: `pathname + ? + queryString` instead of just `?${queryString}`
5. ✅ Add `{ scroll: false }` option to prevent scrolling
6. ✅ Use `params.delete()` to remove default values

**Status**: ✅ **COMPLETELY FIXED!**

**Verification:**
- ✅ Single filter: `?size=small` works
- ✅ Multiple filters: `?size=small&sort=likes` works perfectly
- ✅ No infinite loops - only 2 SSG log messages per navigation
- ✅ Default removal: Clicking "All Sizes" removes `size` param
- ✅ Filter preservation: Adding second filter preserves first filter
- ✅ Description updates: "Most Liked · Small Models"
- ✅ Manifest loading: Successfully loads `filter/likes/small`

---

## Testing Results

### ✅ Everything Works Now!
- ✅ Default page (no filters): `http://localhost:7823/models/huggingface`
- ✅ Single filter - Size: `?size=small`
- ✅ Multiple filters via UI clicks: `?size=small&sort=likes`
- ✅ Manifest loading: Successfully fetches `/manifests/hf-filter-small.json`
- ✅ Filter UI updates: Buttons show correct active states
- ✅ Default value removal: Clicking "All Sizes" removes `size` param from URL
- ✅ Filter preservation: Adding second filter preserves first filter
- ✅ No infinite loops: Only 2 SSG log messages per navigation (expected)
- ✅ Client-side navigation: No full page reloads
- ✅ Manifest-based filtering: Loads lightweight JSON files
- ✅ URL deep linking: Can share filtered URLs

---

## Solution Summary

### What Fixed The Infinite Loop:

**Root Cause**: Using relative URLs (`?${queryString}`) with `router.push()` caused Next.js to treat each navigation as a full server-side render (RSC fetch), creating an infinite loop.

**The Fix**: Follow the official Next.js App Router pattern:
1. Import `usePathname()` and `useCallback`
2. Build full URLs: `pathname + ? + queryString`
3. Wrap handler in `useCallback` with proper dependencies
4. Use `{ scroll: false }` option

**Reference**: [Next.js useSearchParams Documentation](https://nextjs.org/docs/app/api-reference/functions/use-search-params)

### Lessons Learned:

1. **ALWAYS use `usePathname()` when updating search params** - Never use relative URLs with `router.push()`
2. **ALWAYS wrap navigation handlers in `useCallback`** - Prevents function recreation on every render
3. **Use `params.delete()` for defaults** - Keeps URLs clean by removing default values
4. **Add `{ scroll: false }`** - Prevents page scroll on filter changes
5. **Start with official docs** - Next.js documentation has the correct pattern

### For Future Implementations:

When implementing filter pages with URL-based state in Next.js App Router:
- ✅ Use `usePathname()`, `useSearchParams()`, `useRouter()`
- ✅ Wrap handlers in `useCallback` with `[searchParams, pathname, router]` dependencies
- ✅ Build full URLs: `const newUrl = pathname + '?' + params.toString()`
- ✅ Call `router.push(newUrl, { scroll: false })`
- ✅ Test with multiple filters before deploying

---

## Files Modified

1. ✅ `frontend/packages/rbee-ui/src/marketplace/organisms/CategoryFilterBar/CategoryFilterBar.tsx`
   - Fixed `onFilterChange` callback usage
   - Added `onFilterChange` to sort group
   - Removed unused `isTauriEnvironment` import

2. 🔴 `frontend/apps/marketplace/app/models/huggingface/HFFilterPage.tsx`
   - Fixed `handleFilterChange` to merge filters properly
   - Simplified `useEffect` dependencies
   - Still has infinite loop issue with multiple params

---

## Architecture Notes

The HuggingFace page uses **Hybrid SSG + Client-Side Filtering**:

1. **SSG (Server-Side Generation)**:
   - Pre-renders default filter page with 100 models
   - Full model metadata (downloads, likes, author, tags)
   - SEO-friendly

2. **Client-Side Filtering**:
   - URL search params trigger manifest loading
   - Manifests are lightweight JSON files: `/manifests/hf-{filter}.json`
   - Manifest contains only: `id`, `slug`, `name`, `timestamp`
   - Missing metadata shows as `0` or `—` in UI

3. **Filter Combinations**:
   - Defined in `filters.ts`: `PREGENERATED_HF_FILTERS`
   - Example: `{ sort: 'likes', size: 'small', license: 'all' }` → `filter/likes/small`
   - Not all combinations are pre-generated

This architecture allows SSG for SEO while enabling client-side filtering without breaking static generation.
