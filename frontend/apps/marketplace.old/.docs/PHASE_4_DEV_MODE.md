# Phase 4: Dev Mode Optimization

**Status:** 📋 PENDING  
**Dependencies:** Phase 3 (Client-Side Filters)  
**Estimated Time:** 1 hour

---

## Objectives

1. Skip manifest generation in dev mode
2. Use live API fetching for fast development
3. Add dev mode indicators
4. Ensure smooth developer experience

---

## Problem

Generating manifests takes 2-3 minutes:
- Fetches from CivitAI API (13 filters × ~2s = 26s)
- Fetches from HuggingFace API (9 filters × ~2s = 18s)
- Total: ~45s + processing time

This is **too slow** for development where you want instant feedback.

---

## Solution

### Dev Mode Behavior

```
Dev Mode (NODE_ENV=development):
  ├─ Skip manifest generation
  ├─ Fetch small subset from live APIs
  ├─ Only prerender ~20 model pages
  └─ Fast dev server startup (<10s)

Production Mode (NODE_ENV=production):
  ├─ Generate all manifests
  ├─ Prerender all unique model pages
  └─ Slower build time (acceptable for CI/CD)
```

---

## Implementation Steps

### Step 1: Update Manifest Generator

**File:** `scripts/generate-model-manifests.ts`

```typescript
const IS_PROD = process.env.NODE_ENV === 'production'
const IS_CI = process.env.CI === 'true'

async function generateManifests() {
  // Skip in dev mode
  if (!IS_PROD && !IS_CI) {
    console.log('⏭️  Skipping manifest generation in dev mode')
    console.log('   Pages will use live API fetching')
    return
  }
  
  console.log('📦 Generating model manifests for production...')
  
  // ... rest of generation logic
}
```

### Step 2: Update Manifest Loader (Server)

**File:** `lib/manifests.ts`

```typescript
const IS_PROD = process.env.NODE_ENV === 'production'

export async function loadAllModels(): Promise<CombinedManifest['models']> {
  // Dev mode: fetch small subset from live APIs
  if (!IS_PROD) {
    console.log('[manifests] Dev mode - fetching from live APIs')
    return await fetchDevModels()
  }
  
  // Production: read from manifest
  try {
    const manifestData = await fs.promises.readFile(MANIFEST_PATH, 'utf-8')
    const manifest: CombinedManifest = JSON.parse(manifestData)
    return manifest.models
  } catch (error) {
    console.error('[manifests] Failed to load manifest:', error)
    // Fallback to dev mode behavior
    return await fetchDevModels()
  }
}

/**
 * Fetch a small subset of models for dev mode
 * Fast and doesn't require manifest generation
 */
async function fetchDevModels(): Promise<CombinedManifest['models']> {
  const [civitaiModels, hfModels] = await Promise.all([
    fetchTop20CivitAIModels(),
    fetchTop20HFModels(),
  ])
  
  return [
    ...civitaiModels.map(m => ({
      id: `civitai-${m.id}`,
      slug: `civitai-${m.id}`,
      name: m.name,
      source: 'civitai' as const,
    })),
    ...hfModels.map(m => ({
      id: m.id,
      slug: m.id.replace('/', '--'),
      name: m.id,
      source: 'huggingface' as const,
    })),
  ]
}

async function fetchTop20CivitAIModels() {
  const response = await fetch('https://civitai.com/api/v1/models?limit=20&nsfw=true')
  const data = await response.json()
  return data.items || []
}

async function fetchTop20HFModels() {
  const response = await fetch('https://huggingface.co/api/models?limit=20&full=true&sort=likes')
  const data = await response.json()
  return data || []
}
```

### Step 3: Add Dev Mode Indicator

**File:** `components/DevModeIndicator.tsx` (NEW)

```typescript
'use client'

export function DevModeIndicator() {
  const isDev = process.env.NODE_ENV === 'development'
  
  if (!isDev) return null
  
  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-yellow-500 text-yellow-950 px-4 py-2 rounded-lg shadow-lg flex items-center gap-2">
        <div className="size-2 bg-yellow-950 rounded-full animate-pulse" />
        <span className="text-sm font-medium">Dev Mode - Using Live APIs</span>
      </div>
    </div>
  )
}
```

**File:** `app/layout.tsx`

```typescript
import { DevModeIndicator } from '@/components/DevModeIndicator'

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <DevModeIndicator />
      </body>
    </html>
  )
}
```

### Step 4: Update Package.json Scripts

**File:** `package.json`

```json
{
  "scripts": {
    "dev": "next dev --turbopack -p 7823",
    "build": "NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev next build --webpack",
    "build:dev": "NODE_ENV=development next build",
    "generate:manifests": "tsx scripts/generate-model-manifests.ts",
    "prebuild": "bash scripts/validate-no-force-dynamic.sh && pnpm run generate:manifests"
  }
}
```

**Usage:**
```bash
# Dev mode (fast, no manifests)
pnpm run dev

# Dev build (fast, no manifests)
pnpm run build:dev

# Production build (slow, with manifests)
pnpm run build
```

---

## Environment Variables

### .env.development

```bash
# Dev mode settings
NODE_ENV=development
SKIP_MANIFEST_GENERATION=true
DEV_MODEL_LIMIT=20
```

### .env.production

```bash
# Production settings
NODE_ENV=production
SKIP_MANIFEST_GENERATION=false
```

---

## Performance Comparison

### Before (All Modes Generate Manifests)

```
Dev Server Startup:
  ├─ Generate manifests: 45s
  ├─ Next.js compilation: 10s
  └─ Total: 55s ❌ Too slow!

Production Build:
  ├─ Generate manifests: 45s
  ├─ Next.js build: 3min
  └─ Total: 3min 45s
```

### After (Dev Mode Skips Manifests)

```
Dev Server Startup:
  ├─ Skip manifests: 0s ✅
  ├─ Next.js compilation: 10s
  └─ Total: 10s ✅ Fast!

Production Build:
  ├─ Generate manifests: 45s
  ├─ Next.js build: 2min (fewer pages)
  └─ Total: 2min 45s ✅ Faster!
```

---

## Testing

### Dev Mode Test

```bash
# Start dev server
NODE_ENV=development pnpm run dev

# Should see:
# ⏭️  Skipping manifest generation in dev mode
# [manifests] Dev mode - fetching from live APIs

# Verify:
# - Server starts in <15s
# - ~20 models prerendered
# - Dev mode indicator visible
# - Filter pages work (fetch live)
```

### Production Build Test

```bash
# Production build
NODE_ENV=production pnpm run build

# Should see:
# 📦 Generating model manifests for production...
# [manifests] Loaded 300 unique models

# Verify:
# - All manifests generated
# - ~300 models prerendered
# - Build completes in <5min
```

---

## Troubleshooting

### Issue: Manifests still generated in dev

**Solution:** Check `NODE_ENV`
```bash
echo $NODE_ENV  # Should be "development"
```

### Issue: Dev server too slow

**Solution:** Reduce dev model limit
```typescript
// lib/manifests.ts
async function fetchDevModels() {
  // Reduce from 20 to 10
  const [civitaiModels, hfModels] = await Promise.all([
    fetchTop10CivitAIModels(),
    fetchTop10HFModels(),
  ])
}
```

### Issue: Filter pages broken in dev

**Solution:** Ensure fallback to live API works
```typescript
// hooks/useManifest.ts
if (!manifest) {
  console.log('[dev] Manifest not found, using live API')
  const liveModels = await fetchModelsLive(source, filter)
  setModels(liveModels)
}
```

---

## Success Criteria

✅ Phase 4 is complete when:
1. Dev mode skips manifest generation
2. Dev server starts in <15s
3. Production build generates manifests
4. Dev mode indicator shows correctly
5. Both modes work correctly
6. Developer experience is smooth

---

## Next Phase

Once Phase 4 is complete, proceed to:
**[Phase 5: Testing & Validation](./PHASE_5_TESTING.md)**
