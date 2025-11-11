# Phase 2: Update generateStaticParams

**Status:** 📋 PENDING  
**Dependencies:** Phase 1 (Manifest Generation)  
**Estimated Time:** 1-2 hours

---

## Objectives

1. Update model detail pages to read from `all-models.json`
2. Remove expensive API calls from `generateStaticParams`
3. Prerender only unique model pages
4. Remove filter page prerendering

---

## Current State (BEFORE)

### CivitAI Model Page
```typescript
// app/models/civitai/[slug]/page.tsx
export async function generateStaticParams() {
  // ❌ Fetches from API at build time
  const models = await fetchTop100CivitAIModels()
  return models.map(model => ({ slug: `civitai-${model.id}` }))
}
```

### HuggingFace Model Page
```typescript
// app/models/huggingface/[slug]/page.tsx
export async function generateStaticParams() {
  // ❌ Fetches from API at build time
  const models = await fetchTop100HFModels()
  return models.map(model => ({ slug: model.id.replace('/', '--') }))
}
```

### Filter Pages
```typescript
// app/models/civitai/[...filter]/page.tsx
export async function generateStaticParams() {
  // ❌ Generates hundreds of filter combination pages
  return CIVITAI_FILTERS.map(filter => ({ filter: filter.split('/') }))
}
```

---

## Target State (AFTER)

### Shared Manifest Loader

**File:** `lib/manifests.ts` (NEW)

```typescript
import fs from 'node:fs'
import path from 'node:path'

interface CombinedManifest {
  totalModels: number
  civitai: number
  huggingface: number
  models: Array<{
    id: string
    slug: string
    name: string
    source: 'civitai' | 'huggingface'
  }>
  timestamp: string
}

const IS_PROD = process.env.NODE_ENV === 'production'
const MANIFEST_PATH = path.join(process.cwd(), 'public', 'manifests', 'all-models.json')

/**
 * Load all unique models from the combined manifest
 * Only available at build time (SSG)
 */
export async function loadAllModels(): Promise<CombinedManifest['models']> {
  if (!IS_PROD) {
    console.log('[manifests] Skipping in dev mode - using live APIs')
    return []
  }
  
  try {
    const manifestData = await fs.promises.readFile(MANIFEST_PATH, 'utf-8')
    const manifest: CombinedManifest = JSON.parse(manifestData)
    
    console.log(`[manifests] Loaded ${manifest.totalModels} unique models`)
    console.log(`  - CivitAI: ${manifest.civitai}`)
    console.log(`  - HuggingFace: ${manifest.huggingface}`)
    
    return manifest.models
  } catch (error) {
    console.error('[manifests] Failed to load manifest:', error)
    return []
  }
}

/**
 * Load models for a specific source
 */
export async function loadModelsBySource(source: 'civitai' | 'huggingface') {
  const allModels = await loadAllModels()
  return allModels.filter(model => model.source === source)
}

/**
 * Load a specific filter manifest
 * Used by filter pages to render model grids
 */
export async function loadFilterManifest(source: 'civitai' | 'huggingface', filter: string) {
  if (!IS_PROD) {
    return null // Fallback to live API in dev
  }
  
  const filename = `${source}-${filter.replace(/\//g, '-')}.json`
  const manifestPath = path.join(process.cwd(), 'public', 'manifests', filename)
  
  try {
    const data = await fs.promises.readFile(manifestPath, 'utf-8')
    return JSON.parse(data)
  } catch (error) {
    console.error(`[manifests] Failed to load ${filename}:`, error)
    return null
  }
}
```

---

## Implementation Steps

### Step 1: Create Manifest Loader

- [ ] Create `lib/manifests.ts`
- [ ] Implement `loadAllModels()`
- [ ] Implement `loadModelsBySource()`
- [ ] Implement `loadFilterManifest()`
- [ ] Add error handling

### Step 2: Update CivitAI Model Page

**File:** `app/models/civitai/[slug]/page.tsx`

```typescript
import { loadModelsBySource } from '@/lib/manifests'

export async function generateStaticParams() {
  console.log('[SSG] Generating CivitAI model pages from manifest')
  
  // ✅ Read from manifest instead of API
  const models = await loadModelsBySource('civitai')
  
  console.log(`[SSG] Pre-building ${models.length} CivitAI model pages`)
  
  return models.map(model => ({
    slug: model.slug,
  }))
}

// Rest of the page component stays the same
export default async function CivitAIModelPage({ params }: { params: { slug: string } }) {
  // Still fetches full model data at build time
  const model = await fetchCivitAIModel(params.slug)
  return <CivitAIModelDetail model={model} />
}
```

### Step 3: Update HuggingFace Model Page

**File:** `app/models/huggingface/[slug]/page.tsx`

```typescript
import { loadModelsBySource } from '@/lib/manifests'

export async function generateStaticParams() {
  console.log('[SSG] Generating HuggingFace model pages from manifest')
  
  // ✅ Read from manifest instead of API
  const models = await loadModelsBySource('huggingface')
  
  console.log(`[SSG] Pre-building ${models.length} HuggingFace model pages`)
  
  return models.map(model => ({
    slug: model.slug,
  }))
}

// Rest of the page component stays the same
export default async function HFModelPage({ params }: { params: { slug: string } }) {
  // Still fetches full model data at build time
  const model = await fetchHFModel(params.slug)
  return <HFModelDetail model={model} />
}
```

### Step 4: Update Redirect Page

**File:** `app/models/[slug]/page.tsx`

```typescript
import { loadAllModels } from '@/lib/manifests'

export async function generateStaticParams() {
  console.log('[SSG] Generating model redirect pages from manifest')
  
  // ✅ Read from manifest instead of API
  const models = await loadAllModels()
  
  console.log(`[SSG] Pre-building ${models.length} redirect pages`)
  
  return models.map(model => ({
    slug: model.slug,
  }))
}

// Redirect logic stays the same
export default async function ModelRedirectPage({ params }: { params: { slug: string } }) {
  const source = params.slug.startsWith('civitai-') ? 'civitai' : 'huggingface'
  redirect(`/models/${source}/${params.slug}`)
}
```

### Step 5: Remove Filter Page Prerendering

**File:** `app/models/civitai/[...filter]/page.tsx`

```typescript
// ❌ REMOVE THIS
export async function generateStaticParams() {
  return CIVITAI_FILTERS.map(filter => ({ filter: filter.split('/') }))
}

// ✅ MAKE IT DYNAMIC (no prerendering)
// Just remove generateStaticParams entirely
// Next.js will render these pages on-demand

export default async function CivitAIFilterPage({ params }: { params: { filter: string[] } }) {
  // This will now load the manifest client-side (Phase 3)
  const filterPath = params.filter.join('/')
  const manifest = await loadFilterManifest('civitai', filterPath)
  
  return <ModelGrid models={manifest?.models || []} />
}
```

**File:** `app/models/huggingface/[...filter]/page.tsx`

```typescript
// ❌ REMOVE generateStaticParams

// ✅ MAKE IT DYNAMIC
export default async function HFFilterPage({ params }: { params: { filter: string[] } }) {
  const filterPath = params.filter.join('/')
  const manifest = await loadFilterManifest('huggingface', filterPath)
  
  return <ModelGrid models={manifest?.models || []} />
}
```

---

## Dev Mode Handling

### Fallback to Live APIs

```typescript
// lib/manifests.ts
export async function loadAllModels(): Promise<CombinedManifest['models']> {
  if (!IS_PROD) {
    console.log('[manifests] Dev mode - falling back to live APIs')
    
    // Fetch a small subset for dev
    const civitaiModels = await fetchTop20CivitAIModels()
    const hfModels = await fetchTop20HFModels()
    
    return [
      ...civitaiModels.map(m => ({ id: `civitai-${m.id}`, slug: `civitai-${m.id}`, name: m.name, source: 'civitai' as const })),
      ...hfModels.map(m => ({ id: m.id, slug: m.id.replace('/', '--'), name: m.id, source: 'huggingface' as const })),
    ]
  }
  
  // Production: read from manifest
  // ...
}
```

---

## Testing

### Build Test

```bash
# Clean previous build
rm -rf .next out public/manifests

# Generate manifests
NODE_ENV=production pnpm run generate:manifests

# Build site
pnpm run build

# Check output
ls -la out/models/civitai/
ls -la out/models/huggingface/
```

### Validation Checklist

- [ ] Manifests generated successfully
- [ ] Only unique model pages prerendered
- [ ] No filter combination pages prerendered
- [ ] Build completes in <5 minutes
- [ ] Total pages reduced from ~500 to ~200-300
- [ ] Dev mode still works (uses live APIs)

---

## Success Criteria

✅ Phase 2 is complete when:
1. `generateStaticParams` reads from manifests
2. Only unique model pages are prerendered
3. Filter pages are NOT prerendered
4. Build time is significantly reduced
5. Dev mode works without manifests
6. All tests pass

---

## Next Phase

Once Phase 2 is complete, proceed to:
**[Phase 3: Client-Side Filter Pages](./PHASE_3_CLIENT_SIDE_FILTERS.md)**
