# Phase 3: Client-Side Filter Pages

**Status:** 📋 PENDING  
**Dependencies:** Phase 2 (Static Params)  
**Estimated Time:** 2-3 hours

---

## Objectives

1. Load manifests client-side for filter pages
2. Render model grids from manifest data
3. Add loading states
4. Handle missing manifests gracefully

---

## Architecture

```
User visits: /models/civitai/filter/checkpoints
              ↓
Page loads (no SSR, just static HTML shell)
              ↓
Client-side: fetch /manifests/civitai-filter-checkpoints.json
              ↓
Render model grid from manifest data
              ↓
Fast, cached by CDN
```

---

## Implementation Steps

### Step 1: Create Client-Side Manifest Loader

**File:** `lib/manifests-client.ts` (NEW)

```typescript
'use client'

interface ModelManifest {
  filter: string
  models: Array<{
    id: string
    slug: string
    name: string
  }>
  timestamp: string
}

/**
 * Load a filter manifest on the client-side
 * Fetches from /manifests/ directory (served by CDN)
 */
export async function loadFilterManifestClient(
  source: 'civitai' | 'huggingface',
  filter: string
): Promise<ModelManifest | null> {
  const filename = `${source}-${filter.replace(/\//g, '-')}.json`
  const url = `/manifests/${filename}`
  
  try {
    const response = await fetch(url)
    if (!response.ok) {
      console.error(`[manifests] Failed to load ${filename}: ${response.status}`)
      return null
    }
    
    const manifest: ModelManifest = await response.json()
    console.log(`[manifests] Loaded ${manifest.models.length} models for ${filter}`)
    
    return manifest
  } catch (error) {
    console.error(`[manifests] Error loading ${filename}:`, error)
    return null
  }
}

/**
 * Fallback to live API if manifest not found
 */
export async function fetchModelsLive(
  source: 'civitai' | 'huggingface',
  filter: string
): Promise<ModelManifest['models']> {
  console.log(`[manifests] Falling back to live API for ${source}/${filter}`)
  
  if (source === 'civitai') {
    const response = await fetch(`/api/civitai/models?filter=${filter}`)
    const data = await response.json()
    return data.models || []
  } else {
    const response = await fetch(`/api/huggingface/models?filter=${filter}`)
    const data = await response.json()
    return data.models || []
  }
}
```

### Step 2: Create useManifest Hook

**File:** `hooks/useManifest.ts` (NEW)

```typescript
'use client'

import { useEffect, useState } from 'react'
import { loadFilterManifestClient, fetchModelsLive } from '@/lib/manifests-client'

interface UseManifestResult {
  models: Array<{ id: string; slug: string; name: string }>
  loading: boolean
  error: Error | null
}

export function useManifest(
  source: 'civitai' | 'huggingface',
  filter: string
): UseManifestResult {
  const [models, setModels] = useState<UseManifestResult['models']>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  
  useEffect(() => {
    let cancelled = false
    
    async function loadManifest() {
      setLoading(true)
      setError(null)
      
      try {
        // Try to load from manifest first
        const manifest = await loadFilterManifestClient(source, filter)
        
        if (cancelled) return
        
        if (manifest) {
          setModels(manifest.models)
        } else {
          // Fallback to live API
          const liveModels = await fetchModelsLive(source, filter)
          if (!cancelled) {
            setModels(liveModels)
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err : new Error('Failed to load models'))
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }
    
    loadManifest()
    
    return () => {
      cancelled = true
    }
  }, [source, filter])
  
  return { models, loading, error }
}
```

### Step 3: Update CivitAI Filter Page

**File:** `app/models/civitai/[...filter]/page.tsx`

```typescript
'use client'

import { useManifest } from '@/hooks/useManifest'
import { ModelGrid } from '@/components/ModelGrid'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorMessage } from '@/components/ErrorMessage'

export default function CivitAIFilterPage({ params }: { params: { filter: string[] } }) {
  const filterPath = params.filter.join('/')
  const { models, loading, error } = useManifest('civitai', filterPath)
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner />
        <p className="ml-4 text-muted-foreground">Loading models...</p>
      </div>
    )
  }
  
  if (error) {
    return (
      <ErrorMessage 
        title="Failed to load models"
        message={error.message}
        retry={() => window.location.reload()}
      />
    )
  }
  
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">
        CivitAI Models - {filterPath}
      </h1>
      
      <ModelGrid 
        models={models}
        source="civitai"
      />
      
      {models.length === 0 && (
        <p className="text-center text-muted-foreground mt-8">
          No models found for this filter.
        </p>
      )}
    </div>
  )
}
```

### Step 4: Update HuggingFace Filter Page

**File:** `app/models/huggingface/[...filter]/page.tsx`

```typescript
'use client'

import { useManifest } from '@/hooks/useManifest'
import { ModelGrid } from '@/components/ModelGrid'
import { LoadingSpinner } from '@/components/LoadingSpinner'
import { ErrorMessage } from '@/components/ErrorMessage'

export default function HFFilterPage({ params }: { params: { filter: string[] } }) {
  const filterPath = params.filter.join('/')
  const { models, loading, error } = useManifest('huggingface', filterPath)
  
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner />
        <p className="ml-4 text-muted-foreground">Loading models...</p>
      </div>
    )
  }
  
  if (error) {
    return (
      <ErrorMessage 
        title="Failed to load models"
        message={error.message}
        retry={() => window.location.reload()}
      />
    )
  }
  
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">
        HuggingFace Models - {filterPath}
      </h1>
      
      <ModelGrid 
        models={models}
        source="huggingface"
      />
      
      {models.length === 0 && (
        <p className="text-center text-muted-foreground mt-8">
          No models found for this filter.
        </p>
      )}
    </div>
  )
}
```

### Step 5: Create ModelGrid Component

**File:** `components/ModelGrid.tsx` (NEW)

```typescript
'use client'

import Link from 'next/link'
import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'

interface ModelGridProps {
  models: Array<{
    id: string
    slug: string
    name: string
  }>
  source: 'civitai' | 'huggingface'
}

export function ModelGrid({ models, source }: ModelGridProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {models.map((model) => (
        <Link 
          key={model.id}
          href={`/models/${source}/${model.slug}`}
          className="group"
        >
          <Card className="h-full transition-all hover:shadow-lg hover:scale-105">
            <CardHeader>
              <CardTitle className="line-clamp-2 group-hover:text-primary transition-colors">
                {model.name}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                {source === 'civitai' ? 'CivitAI' : 'HuggingFace'}
              </p>
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  )
}
```

---

## Loading States

### Skeleton Loader

```typescript
export function ModelGridSkeleton() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {Array.from({ length: 12 }).map((_, i) => (
        <Card key={i} className="h-full">
          <CardHeader>
            <div className="h-6 bg-muted animate-pulse rounded" />
          </CardHeader>
          <CardContent>
            <div className="h-4 bg-muted animate-pulse rounded w-1/2" />
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
```

---

## Error Handling

### Graceful Degradation

1. **Manifest not found** → Fallback to live API
2. **Live API fails** → Show error message with retry
3. **Network error** → Show offline message

```typescript
if (!manifest) {
  console.warn('[manifests] Manifest not found, falling back to live API')
  const liveModels = await fetchModelsLive(source, filter)
  return liveModels
}
```

---

## Caching Strategy

### CDN Caching

Manifests are static JSON files served from `/public/manifests/`:
- Cached by Cloudflare CDN
- Fast global delivery
- No server-side processing

### Browser Caching

```typescript
// Add cache headers in next.config.ts
async headers() {
  return [
    {
      source: '/manifests/:path*',
      headers: [
        {
          key: 'Cache-Control',
          value: 'public, max-age=3600, stale-while-revalidate=86400',
        },
      ],
    },
  ]
}
```

---

## Testing

### Manual Testing

```bash
# Start dev server
pnpm run dev

# Visit filter pages
open http://localhost:7823/models/civitai/filter/checkpoints
open http://localhost:7823/models/huggingface/likes

# Check network tab for manifest requests
# Verify models load correctly
```

### Validation Checklist

- [ ] Filter pages load manifests client-side
- [ ] Loading states display correctly
- [ ] Error states display correctly
- [ ] Fallback to live API works
- [ ] Model grid renders correctly
- [ ] Links to model detail pages work
- [ ] CDN caching works in production

---

## Success Criteria

✅ Phase 3 is complete when:
1. Filter pages load manifests client-side
2. Loading and error states work
3. Fallback to live API works
4. Model grids render correctly
5. All filter routes work
6. Performance is acceptable (<1s load time)

---

## Next Phase

Once Phase 3 is complete, proceed to:
**[Phase 4: Dev Mode Optimization](./PHASE_4_DEV_MODE.md)**
