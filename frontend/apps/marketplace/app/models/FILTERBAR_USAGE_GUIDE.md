# FilterBar Usage Guide for Model List Pages

**Author**: TEAM-464  
**Date**: 2025-11-11  
**Purpose**: How to properly implement `ModelsFilterBar` / `CategoryFilterBar` in filter list pages

---

## Overview

The FilterBar component enables **client-side filtering with URL-based state** for model catalog pages. It supports:

- ✅ SSG-compatible filtering (pre-rendered pages)
- ✅ Client-side navigation (no full page reloads)
- ✅ URL search parameters for deep linking
- ✅ Manifest-based data loading
- ✅ Multiple filter combinations

---

## Architecture: Hybrid SSG + Client-Side Filtering

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SSG (Server-Side Generation)                             │
│    - Pre-render default filter page                         │
│    - Full model metadata (downloads, likes, tags)           │
│    - SEO-friendly, instant loading                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. User Clicks Filter                                        │
│    - FilterBar calls onChange({ size: 'small' })            │
│    - Page merges with current filters                       │
│    - Updates URL: ?size=small                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. useEffect Detects URL Change                             │
│    - Reads searchParams                                     │
│    - Finds matching filter config                           │
│    - Loads manifest: /manifests/hf-filter-small.json        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Client-Side Update                                        │
│    - setModels(manifestModels)                              │
│    - No page reload, instant update                         │
│    - Manifest has lightweight data (id, name, slug)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Implementation

### 1. Define Filter Configurations

Create a `filters.ts` file with your filter definitions:

```typescript
// app/models/[source]/filters.ts
import type { FilterGroup } from '@rbee/ui/marketplace'

export interface YourFilters {
  sort: 'downloads' | 'likes' | 'recent'
  size: 'all' | 'small' | 'medium' | 'large'
  license: 'all' | 'apache' | 'mit' | 'other'
}

// Left-side filters (categorical)
export const YOUR_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'size',
    label: 'Model Size',
    options: [
      { label: 'All Sizes', value: 'all' },
      { label: 'Small (<7B)', value: 'small' },
      { label: 'Medium (7B-13B)', value: 'medium' },
      { label: 'Large (>13B)', value: 'large' },
    ],
  },
  {
    id: 'license',
    label: 'License',
    options: [
      { label: 'All Licenses', value: 'all' },
      { label: 'Apache 2.0', value: 'apache' },
      { label: 'MIT', value: 'mit' },
      { label: 'Other', value: 'other' },
    ],
  },
]

// Right-side filter (sorting)
export const YOUR_SORT_GROUP: FilterGroup = {
  id: 'sort',
  label: 'Sort By',
  options: [
    { label: 'Most Downloads', value: 'downloads' },
    { label: 'Most Likes', value: 'likes' },
    { label: 'Recently Updated', value: 'recent' },
  ],
}

// Pre-generated filter combinations for SSG
export const PREGENERATED_FILTERS: FilterConfig<YourFilters>[] = [
  // Default (no URL params)
  { filters: { sort: 'downloads', size: 'all', license: 'all' }, path: '' },
  
  // Single filters
  { filters: { sort: 'likes', size: 'all', license: 'all' }, path: 'filter/likes' },
  { filters: { sort: 'downloads', size: 'small', license: 'all' }, path: 'filter/small' },
  
  // Combined filters
  { filters: { sort: 'likes', size: 'small', license: 'all' }, path: 'filter/likes/small' },
  
  // Add more combinations as needed
]
```

---

### 2. Create the Server Component (SSG Page)

```typescript
// app/models/[source]/page.tsx
import { Suspense } from 'react'
import { YourFilterPage } from './YourFilterPage'
import { PREGENERATED_FILTERS } from './filters'
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Your Models | Marketplace',
  description: 'Browse models with instant filtering',
}

export default async function YourModelsPage() {
  // Default filter (downloads, all sizes, all licenses)
  const currentFilter = PREGENERATED_FILTERS[0].filters
  
  // Fetch models for SSG (top 100 for SEO)
  const FETCH_LIMIT = 100
  const models = await fetchYourModels({ limit: FETCH_LIMIT })
  
  // Normalize models for client component
  const normalizedModels = models.map((model) => ({
    id: model.id,
    name: model.name,
    description: model.description,
    author: model.author,
    downloads: model.downloads ?? 0,
    likes: model.likes ?? 0,
    tags: model.tags?.slice(0, 10) ?? [],
  }))
  
  // Pass SSG data to client component (wrapped in Suspense for useSearchParams)
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <YourFilterPage 
        initialModels={normalizedModels} 
        initialFilter={currentFilter} 
      />
    </Suspense>
  )
}
```

---

### 3. Create the Client Component (Filter Page)

```typescript
// app/models/[source]/YourFilterPage.tsx
'use client'

import { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useRouter, usePathname } from 'next/navigation'
import { ModelsFilterBar } from '../ModelsFilterBar'
import { loadFilterManifestClient } from '@/lib/manifests-client'
import {
  YOUR_FILTER_GROUPS,
  YOUR_SORT_GROUP,
  PREGENERATED_FILTERS,
  type YourFilters,
} from './filters'

interface Model {
  id: string
  name: string
  description: string
  author?: string
  downloads: number
  likes: number
  tags: string[]
}

interface Props {
  initialModels: Model[]
  initialFilter: YourFilters
}

export function YourFilterPage({ initialModels, initialFilter }: Props) {
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()  // ⚠️ CRITICAL: Required for building full URLs
  const [models, setModels] = useState<Model[]>(initialModels)
  const [loading, setLoading] = useState(false)
  
  // Build current filter from URL search params
  const currentFilter: YourFilters = {
    sort: (searchParams.get('sort') as any) || initialFilter.sort,
    size: (searchParams.get('size') as any) || initialFilter.size,
    license: (searchParams.get('license') as any) || initialFilter.license,
  }

  // ⚠️ CRITICAL: Handle filter changes - use useCallback and pathname
  // Following official Next.js App Router pattern to avoid infinite loops
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
    
    // ⚠️ CRITICAL: Build full URL with pathname (prevents infinite loop)
    // Don't use relative URLs like "?query" - Next.js treats them as server navigations
    const queryString = params.toString()
    const newUrl = queryString ? `${pathname}?${queryString}` : pathname
    
    router.push(newUrl, { scroll: false })
  }, [searchParams, pathname, router, initialFilter])

  // Load manifest when URL params change
  useEffect(() => {
    // TEAM-464: Capture initial values to avoid dependency issues
    const defaultSort = initialFilter.sort
    const defaultSize = initialFilter.size
    const defaultLicense = initialFilter.license
    const defaultModels = initialModels
    
    async function loadManifest() {
      // Get current filter values from URL
      const sort = searchParams.get('sort') || defaultSort
      const size = searchParams.get('size') || defaultSize
      const license = searchParams.get('license') || defaultLicense
      
      // Find the filter path for current filter config
      const filterConfig = PREGENERATED_FILTERS.find(
        (f) =>
          f.filters.sort === sort &&
          f.filters.size === size &&
          f.filters.license === license
      )

      if (!filterConfig || filterConfig.path === '') {
        // Default filter, use initial models
        setModels(defaultModels)
        return
      }

      setLoading(true)
      try {
        const manifest = await loadFilterManifestClient('your-source', filterConfig.path)
        
        if (manifest) {
          // Convert manifest models to full model objects
          const manifestModels = manifest.models.map((m) => ({
            id: m.id,
            name: m.name,
            description: '',
            author: undefined,
            downloads: 0,
            likes: 0,
            tags: [],
          }))
          setModels(manifestModels)
        } else {
          // Fallback to initial models if manifest not found
          setModels(defaultModels)
        }
      } catch (error) {
        console.error('Failed to load manifest:', error)
        setModels(defaultModels)
      } finally {
        setLoading(false)
      }
    }

    loadManifest()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams])  // Only depend on searchParams

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      <h1 className="text-4xl font-bold mb-8">Your Models</h1>
      
      {/* Filter Bar */}
      <ModelsFilterBar
        groups={YOUR_FILTER_GROUPS}
        sortGroup={YOUR_SORT_GROUP}
        currentFilters={currentFilter}
        onChange={handleFilterChange}  // ⚠️ CRITICAL: Pass onChange callback
      />

      {/* Loading indicator */}
      {loading && <div>Loading filtered models...</div>}

      {/* Model list */}
      <div className="grid gap-4">
        {models.map((model) => (
          <div key={model.id} className="border p-4 rounded">
            <h3>{model.name}</h3>
            <p>{model.description}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
```

---

## Critical Implementation Details

### ✅ DO: Merge Current + New Filters

```typescript
// ✅ CORRECT: Preserve existing filters
const handleFilterChange = (newFilters: Partial<Record<string, string>>) => {
  const merged = { ...currentFilter, ...newFilters }  // Merge!
  
  const params = new URLSearchParams()
  if (merged.sort !== 'downloads') params.set('sort', merged.sort)
  if (merged.size !== 'all') params.set('size', merged.size)
  
  router.push(`?${params.toString()}`)
}
```

```typescript
// ❌ WRONG: Loses existing filters
const handleFilterChange = (newFilters: Partial<Record<string, string>>) => {
  const params = new URLSearchParams()
  
  // Only sets the NEW filter, loses existing ones!
  if (newFilters.sort) params.set('sort', newFilters.sort)
  if (newFilters.size) params.set('size', newFilters.size)
  
  router.push(`?${params.toString()}`)
}
```

### ✅ DO: Pass `onChange` Callback

```typescript
// ✅ CORRECT: FilterBar uses client-side navigation
<ModelsFilterBar
  groups={YOUR_FILTER_GROUPS}
  sortGroup={YOUR_SORT_GROUP}
  currentFilters={currentFilter}
  onChange={handleFilterChange}  // ← Client-side navigation
/>
```

```typescript
// ❌ WRONG: FilterBar will use window.location.href (full page reload)
<ModelsFilterBar
  groups={YOUR_FILTER_GROUPS}
  sortGroup={YOUR_SORT_GROUP}
  currentFilters={currentFilter}
  buildUrlFn="/models/your-source"  // ← Only for fallback
  // Missing onChange! Will cause full page reloads
/>
```

### ✅ DO: Exclude Default Values from URL

```typescript
// ✅ CORRECT: Clean URLs
if (merged.sort && merged.sort !== 'downloads') {
  params.set('sort', merged.sort)
}
// URL: ?size=small (not ?sort=downloads&size=small)
```

```typescript
// ❌ WRONG: Cluttered URLs
if (merged.sort) {
  params.set('sort', merged.sort)
}
// URL: ?sort=downloads&size=small&license=all (ugly!)
```

### ✅ DO: Only Depend on `searchParams` in useEffect

```typescript
// ✅ CORRECT: Capture values inside useEffect
useEffect(() => {
  const defaultSort = initialFilter.sort
  const defaultModels = initialModels
  
  async function loadManifest() {
    const sort = searchParams.get('sort') || defaultSort
    // Use defaultModels, not initialModels
  }
  
  loadManifest()
}, [searchParams])  // Only searchParams
```

```typescript
// ❌ WRONG: Causes infinite loops
useEffect(() => {
  async function loadManifest() {
    const sort = searchParams.get('sort') || initialFilter.sort
    setModels(initialModels)  // Uses prop directly
  }
  
  loadManifest()
}, [searchParams, initialModels, initialFilter])  // Too many deps!
```

---

## Common Pitfalls

### 🔴 Pitfall 1: Using Relative URLs with router.push()

**Symptom**: Infinite loop, hundreds of console logs, "Maximum call stack size exceeded"

**Example**:
```typescript
// ❌ WRONG - Causes infinite loop
const handleFilterChange = (newFilters) => {
  const params = new URLSearchParams()
  params.set('size', newFilters.size)
  
  // This triggers RSC re-fetches in a loop!
  router.push(`?${params.toString()}`)
}
```

**Solution**: Always use `usePathname()` and build full URLs with `useCallback`:
```typescript
// ✅ CORRECT
const pathname = usePathname()

const handleFilterChange = useCallback((newFilters) => {
  const params = new URLSearchParams(searchParams.toString())
  params.set('size', newFilters.size)
  
  const queryString = params.toString()
  const newUrl = queryString ? `${pathname}?${queryString}` : pathname
  
  router.push(newUrl, { scroll: false })
}, [searchParams, pathname, router])
```

---

### 🔴 Pitfall 2: Not Merging Filters

**Symptom**: Clicking a second filter removes the first one

**Example**:
- User clicks "Small" → URL: `?size=small` ✅
- User clicks "Most Likes" → URL: `?sort=likes` ❌ (lost `size=small`)

**Solution**: Use `searchParams.toString()` as base, then modify:
```typescript
// ✅ CORRECT
const params = new URLSearchParams(searchParams.toString())  // Preserve existing
params.set('size', merged.size)  // Update new filter
```

---

### 🔴 Pitfall 3: Missing `onChange` Callback

**Symptom**: Full page reloads when clicking filters

**Example**:
```typescript
// ❌ WRONG
<ModelsFilterBar
  groups={FILTER_GROUPS}
  currentFilters={currentFilter}
  buildUrlFn="/models/source"
  // Missing onChange!
/>
```

**Solution**: Always pass `onChange={handleFilterChange}`

---

### 🔴 Pitfall 4: Infinite Loops from useEffect Dependencies

**Symptom**: Console shows hundreds of render logs, browser freezes

**Example**:
```typescript
// ❌ WRONG
useEffect(() => {
  loadManifest()
}, [searchParams, initialModels, initialFilter])  // Props cause re-renders
```

**Solution**: Only depend on `searchParams`, capture props inside effect

---

### 🔴 Pitfall 5: Not Handling Missing Filter Configs

**Symptom**: Blank page when filter combination doesn't exist

**Example**:
```typescript
// User navigates to ?sort=recent&size=large
// But PREGENERATED_FILTERS doesn't have this combination
// filterConfig is undefined → crash!
```

**Solution**: Always check `if (!filterConfig)` and fallback to initial models

---

## Manifest File Structure

Manifests should be lightweight JSON files served from `/public/manifests/`:

```json
{
  "filter": "filter/likes/small",
  "models": [
    {
      "id": "deepseek-ai/DeepSeek-R1",
      "slug": "deepseek-ai--DeepSeek-R1",
      "name": "deepseek-ai/DeepSeek-R1"
    },
    {
      "id": "black-forest-labs/FLUX.1-dev",
      "slug": "black-forest-labs--FLUX.1-dev",
      "name": "black-forest-labs/FLUX.1-dev"
    }
  ],
  "timestamp": "2025-11-10T23:18:24.593Z"
}
```

**File naming convention**: `{source}-{filter-path}.json`
- Example: `hf-filter-likes-small.json`
- Example: `civitai-filter-checkpoint-nsfw.json`

---

## Testing Checklist

Before deploying a filter page, verify:

- [ ] **Single filter works**: Click one filter, URL updates, content changes
- [ ] **Multiple filters work**: Click two filters, both params in URL
- [ ] **No infinite loops**: Console doesn't show repeated render logs
- [ ] **No full page reloads**: Network tab doesn't show document requests
- [ ] **Default filter works**: Navigate to base URL, shows all models
- [ ] **Missing filter graceful**: Navigate to non-existent filter, shows fallback
- [ ] **URL deep linking**: Copy URL with params, paste in new tab, works
- [ ] **Manifest loading**: Network tab shows manifest fetch, not full page
- [ ] **Filter UI updates**: Active filter buttons show correct state
- [ ] **Back button works**: Browser back/forward navigates filters

---

## Performance Tips

### 1. Pre-generate Common Filter Combinations

Don't try to support every possible combination. Focus on the most common:

```typescript
// ✅ GOOD: Common combinations
{ filters: { sort: 'downloads', size: 'all', license: 'all' }, path: '' },
{ filters: { sort: 'likes', size: 'all', license: 'all' }, path: 'filter/likes' },
{ filters: { sort: 'downloads', size: 'small', license: 'all' }, path: 'filter/small' },
```

```typescript
// ❌ BAD: Exponential combinations
// 3 sorts × 4 sizes × 4 licenses = 48 filter files!
```

### 2. Keep Manifests Small

Only include essential fields:

```typescript
// ✅ GOOD: Minimal manifest
{ id: 'model-id', slug: 'model-slug', name: 'Model Name' }

// ❌ BAD: Full model data
{ id, name, description, tags, downloads, likes, author, ... }
// This defeats the purpose of manifests!
```

### 3. Use Suspense Boundaries

Wrap the filter page in Suspense to prevent blocking:

```typescript
<Suspense fallback={<FilterPageSkeleton />}>
  <YourFilterPage initialModels={models} initialFilter={filter} />
</Suspense>
```

---

## Debugging

### Enable Logging

Add console logs to track filter changes:

```typescript
const handleFilterChange = (newFilters: Partial<Record<string, string>>) => {
  console.log('[FilterChange] New:', newFilters)
  console.log('[FilterChange] Current:', currentFilter)
  
  const merged = { ...currentFilter, ...newFilters }
  console.log('[FilterChange] Merged:', merged)
  
  // ... rest of implementation
}

useEffect(() => {
  console.log('[useEffect] searchParams changed:', searchParams.toString())
  loadManifest()
}, [searchParams])
```

### Check Network Tab

- **Manifest fetch**: Should see `GET /manifests/hf-filter-small.json` (200 OK)
- **No document requests**: Should NOT see `GET /models/huggingface?size=small` (document)
- **RSC payload**: May see `GET /?_rsc=...` (Next.js internal, OK)

### Check Console for Errors

Common errors:
- `Maximum call stack size exceeded` → Infinite loop (check useEffect deps)
- `Failed to fetch manifest` → Manifest file missing or wrong path
- `Cannot read property 'map' of undefined` → Missing null check on manifest

---

## Example: Complete Working Implementation

See the HuggingFace models page for a reference implementation:

- **Server Component**: `frontend/apps/marketplace/app/models/huggingface/page.tsx`
- **Client Component**: `frontend/apps/marketplace/app/models/huggingface/HFFilterPage.tsx`
- **Filter Definitions**: `frontend/apps/marketplace/app/models/huggingface/filters.ts`
- **Manifest Loader**: `frontend/apps/marketplace/lib/manifests-client.ts`

---

## Summary

**Key Principles**:
1. ✅ Always merge current + new filters
2. ✅ Always pass `onChange` callback to FilterBar
3. ✅ Only depend on `searchParams` in useEffect
4. ✅ Exclude default values from URL
5. ✅ Handle missing filter configs gracefully
6. ✅ Keep manifests lightweight
7. ✅ Test with multiple filters before deploying

**Architecture Benefits**:
- 🚀 Instant filtering (no page reloads)
- 🔍 SEO-friendly (SSG default page)
- 🔗 Deep linking (URL-based state)
- 📦 Small bundles (manifest-based data)
- ♿ Accessible (proper ARIA labels)

Follow this guide and your filter pages will be fast, SEO-friendly, and user-friendly!
