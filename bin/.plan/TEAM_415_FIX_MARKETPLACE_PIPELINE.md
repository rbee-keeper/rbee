# TEAM-415: Fix Marketplace Pipeline Architecture

**Created by:** TEAM-406  
**Date:** 2025-11-05  
**Mission:** Fix marketplace to use proper pipeline: marketplace-node → marketplace-sdk (Rust)  
**Problem:** Marketplace app directly calls HuggingFace API via fetch() - WRONG!  
**Status:** 🔥 CRITICAL FIX REQUIRED

---

## 🚨 Problem Identified

### Current (WRONG) Architecture:

```
Next.js Marketplace App
  ↓
  lib/huggingface.ts (direct fetch to HuggingFace API)
  ↓
  https://huggingface.co/api/models
```

**Issues:**
1. ❌ Bypasses marketplace-node package entirely
2. ❌ Bypasses marketplace-sdk (Rust) entirely
3. ❌ No type safety from Rust
4. ❌ No shared logic with Tauri app
5. ❌ Duplicate code (fetch logic in multiple places)
6. ❌ Can't leverage Rust's performance/safety

---

### Correct Architecture:

```
Next.js Marketplace App
  ↓
  @rbee/marketplace-node (Node.js wrapper)
  ↓
  @rbee/marketplace-sdk (Rust crate - native, NOT WASM for SSG)
  ↓
  HuggingFace API
```

**Benefits:**
1. ✅ Single source of truth (Rust)
2. ✅ Type safety from Rust → TypeScript
3. ✅ Shared logic with Tauri app
4. ✅ Performance (Rust HTTP client)
5. ✅ Easy to add compatibility filtering
6. ✅ Consistent API across all apps

---

## 📋 Files to Fix

### ❌ Files Using WRONG Approach (Direct fetch):

1. `/frontend/apps/marketplace/lib/huggingface.ts` - **DELETE THIS FILE**
2. `/frontend/apps/marketplace/app/models/[slug]/page.tsx` - Import from marketplace-node
3. `/frontend/apps/marketplace/app/models/page.tsx` - Import from marketplace-node
4. `/frontend/apps/marketplace/app/api/models/[id]/route.ts` - **DELETE THIS FILE** (not needed for SSG)

### ✅ Files Using CORRECT Approach:

1. `/bin/99_shared_crates/marketplace-sdk/src/huggingface.rs` - ✅ Already implemented!
2. `/frontend/packages/marketplace-node/src/index.ts` - ❌ Needs implementation (currently TODOs)

---

## 🔧 Implementation Plan

### Phase 1: Implement marketplace-node Functions (2-3 hours)

**File:** `/frontend/packages/marketplace-node/src/index.ts`

**Current State:** All functions return empty arrays with TODO comments

**Required Changes:**

```typescript
// TEAM-415: Implement actual marketplace-node functions using marketplace-sdk

import { HuggingFaceClient } from '@rbee/marketplace-sdk'

// Create singleton client
let hfClient: HuggingFaceClient | null = null

function getHFClient(): HuggingFaceClient {
  if (!hfClient) {
    hfClient = new HuggingFaceClient()
  }
  return hfClient
}

/**
 * List HuggingFace models
 */
export async function listHuggingFaceModels(
  options: SearchOptions = {}
): Promise<Model[]> {
  const client = getHFClient()
  const { limit = 50, sort = 'popular' } = options
  
  // Map sort option to HF API format
  const sortParam = sort === 'popular' ? 'downloads' : 
                    sort === 'recent' ? 'recent' : 
                    'trending'
  
  return await client.listModels(
    null,              // query
    sortParam,         // sort
    null,              // filter_tags
    limit              // limit
  )
}

/**
 * Search HuggingFace models
 */
export async function searchHuggingFaceModels(
  query: string,
  options: SearchOptions = {}
): Promise<Model[]> {
  const client = getHFClient()
  const { limit = 50 } = options
  
  return await client.searchModels(query, limit)
}

/**
 * Get a specific HuggingFace model
 */
export async function getHuggingFaceModel(modelId: string): Promise<Model> {
  const client = getHFClient()
  return await client.getModel(modelId)
}
```

**Checklist:**
- [ ] Remove TODO comments
- [ ] Implement `listHuggingFaceModels()` using marketplace-sdk
- [ ] Implement `searchHuggingFaceModels()` using marketplace-sdk
- [ ] Add `getHuggingFaceModel()` function (missing!)
- [ ] Export new functions
- [ ] Add JSDoc comments

---

### Phase 2: Update marketplace-sdk for Node.js (1-2 hours)

**Problem:** marketplace-sdk's HuggingFaceClient is only available for native Rust (not WASM)

**File:** `/bin/99_shared_crates/marketplace-sdk/src/lib.rs`

**Current:**
```rust
#[cfg(not(target_arch = "wasm32"))]
mod huggingface;

#[cfg(not(target_arch = "wasm32"))]
pub use huggingface::HuggingFaceClient;
```

**This is CORRECT!** For Node.js (SSG), we use native Rust, not WASM.

**Required:** Ensure marketplace-node can import the native Rust client

**Check:** Does marketplace-node's package.json have the right setup?

**File:** `/frontend/packages/marketplace-node/package.json`

```json
{
  "name": "@rbee/marketplace-node",
  "version": "0.1.0",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "dependencies": {
    "@rbee/marketplace-sdk": "workspace:*",
    "@rbee/sdk-loader": "workspace:*"
  },
  "scripts": {
    "build": "tsc",
    "dev": "tsc --watch"
  }
}
```

**Issue:** marketplace-node is trying to load WASM, but for Node.js SSG we should use native Rust!

**Solution:** marketplace-sdk needs to expose a Node.js native addon (napi-rs)

**Alternative (Quick Fix):** Keep using fetch in marketplace-node but centralize it there

---

### Phase 3: Quick Fix - Centralize fetch in marketplace-node (1 hour)

**If native Rust binding is too complex right now, centralize fetch logic in marketplace-node:**

**File:** `/frontend/packages/marketplace-node/src/huggingface.ts` (NEW)

```typescript
// TEAM-415: HuggingFace API client (centralized in marketplace-node)
// TODO: Replace with native Rust client when napi-rs binding is ready

const HF_API_BASE = 'https://huggingface.co/api'

export interface HFModel {
  id: string
  modelId?: string
  author?: string
  downloads?: number
  likes?: number
  tags?: string[]
  pipeline_tag?: string
  lastModified?: string
  createdAt?: string
  siblings?: Array<{ rfilename: string; size: number }>
  cardData?: { model_description?: string }
  description?: string
  config?: any
}

export async function fetchHFModels(
  query?: string,
  sort: string = 'downloads',
  limit: number = 50
): Promise<HFModel[]> {
  let url = `${HF_API_BASE}/models?limit=${limit}`
  
  if (query) {
    url += `&search=${encodeURIComponent(query)}`
  }
  
  url += `&sort=${sort}&direction=-1`
  
  const response = await fetch(url, {
    next: { revalidate: 3600 }
  })
  
  if (!response.ok) {
    throw new Error(`HuggingFace API error: ${response.statusText}`)
  }
  
  return await response.json()
}

export async function fetchHFModel(modelId: string): Promise<HFModel> {
  const response = await fetch(
    `${HF_API_BASE}/models/${modelId}`,
    { next: { revalidate: 3600 } }
  )
  
  if (!response.ok) {
    throw new Error(`Model not found: ${modelId}`)
  }
  
  return await response.json()
}
```

**File:** `/frontend/packages/marketplace-node/src/index.ts` (UPDATE)

```typescript
import { fetchHFModels, fetchHFModel, type HFModel } from './huggingface'
import type { Model } from './types'

function convertHFModel(hf: HFModel): Model {
  const parts = hf.id.split('/')
  const name = parts.length >= 2 ? parts[1] : hf.id
  const author = parts.length >= 2 ? parts[0] : hf.author || null
  
  // Calculate total size
  const totalBytes = hf.siblings?.reduce((sum, file) => sum + (file.size || 0), 0) || 0
  
  return {
    id: hf.id,
    name,
    author,
    description: hf.cardData?.model_description || hf.description || '',
    downloads: hf.downloads || 0,
    likes: hf.likes || 0,
    size: formatBytes(totalBytes),
    tags: hf.tags || [],
    source: 'huggingface',
    createdAt: hf.createdAt,
    lastModified: hf.lastModified,
    config: hf.config,
    siblings: hf.siblings || [],
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
}

export async function listHuggingFaceModels(
  options: SearchOptions = {}
): Promise<Model[]> {
  const { limit = 50, sort = 'popular' } = options
  const sortParam = sort === 'popular' ? 'downloads' : sort
  
  const hfModels = await fetchHFModels(undefined, sortParam, limit)
  return hfModels.map(convertHFModel)
}

export async function searchHuggingFaceModels(
  query: string,
  options: SearchOptions = {}
): Promise<Model[]> {
  const { limit = 50 } = options
  
  const hfModels = await fetchHFModels(query, 'downloads', limit)
  return hfModels.map(convertHFModel)
}

export async function getHuggingFaceModel(modelId: string): Promise<Model> {
  const hfModel = await fetchHFModel(modelId)
  return convertHFModel(hfModel)
}
```

**Checklist:**
- [ ] Create `huggingface.ts` in marketplace-node
- [ ] Implement fetch functions
- [ ] Update `index.ts` to use fetch functions
- [ ] Add type definitions
- [ ] Export all functions

---

### Phase 4: Update Marketplace App to Use marketplace-node (1 hour)

**File:** `/frontend/apps/marketplace/app/models/[slug]/page.tsx`

**Before (WRONG):**
```typescript
import { fetchModel, transformToModelDetailData, getStaticModelIds } from '@/lib/huggingface'
```

**After (CORRECT):**
```typescript
import { getHuggingFaceModel, listHuggingFaceModels } from '@rbee/marketplace-node'

export async function generateStaticParams() {
  const models = await listHuggingFaceModels({ limit: 100 })
  return models.map((model) => ({ 
    slug: modelIdToSlug(model.id) 
  }))
}

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const model = await getHuggingFaceModel(modelId)
    
    return {
      title: `${model.name} | AI Model`,
      description: model.description || `${model.name} - ${model.downloads.toLocaleString()} downloads`,
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}

export default async function ModelPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const model = await getHuggingFaceModel(modelId)
    
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <ModelDetailPageTemplate model={model} showBackButton={false} />
      </div>
    )
  } catch {
    notFound()
  }
}
```

**File:** `/frontend/apps/marketplace/app/models/page.tsx`

**Before (WRONG):**
```typescript
import { fetchTopModels } from '@/lib/huggingface'
```

**After (CORRECT):**
```typescript
import { listHuggingFaceModels } from '@rbee/marketplace-node'

export default async function ModelsPage() {
  const models = await listHuggingFaceModels({ limit: 100 })
  
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">AI Models</h1>
      <ModelGrid models={models} />
    </div>
  )
}
```

**Checklist:**
- [ ] Update `models/[slug]/page.tsx` to use marketplace-node
- [ ] Update `models/page.tsx` to use marketplace-node
- [ ] Remove imports from `@/lib/huggingface`
- [ ] Test SSG build
- [ ] Verify types work correctly

---

### Phase 5: Delete Wrong Files (5 minutes)

- [ ] **DELETE** `/frontend/apps/marketplace/lib/huggingface.ts`
- [ ] **DELETE** `/frontend/apps/marketplace/app/api/models/[id]/route.ts` (not needed for SSG)
- [ ] **DELETE** `/frontend/apps/marketplace/app/api/models/route.ts` (if exists)

---

### Phase 6: Add Type Definitions (30 minutes)

**File:** `/frontend/packages/marketplace-node/src/types.ts` (NEW)

```typescript
// TEAM-415: Shared types for marketplace-node

export interface Model {
  id: string
  name: string
  author: string | null
  description: string
  downloads: number
  likes: number
  size: string
  tags: string[]
  source: 'huggingface' | 'civitai'
  createdAt?: string
  lastModified?: string
  config?: any
  siblings?: Array<{ rfilename: string; size: number }>
}

export interface SearchOptions {
  limit?: number
  sort?: 'popular' | 'recent' | 'trending'
}
```

**Update:** `/frontend/packages/marketplace-node/src/index.ts`

```typescript
export type { Model, SearchOptions } from './types'
```

---

## 🎯 Testing Checklist

### Unit Tests
- [ ] Test `listHuggingFaceModels()` returns models
- [ ] Test `searchHuggingFaceModels()` filters by query
- [ ] Test `getHuggingFaceModel()` returns single model
- [ ] Test error handling (model not found)

### Integration Tests
- [ ] Run `npm run build` in marketplace app
- [ ] Verify SSG generates static pages
- [ ] Check that models are fetched via marketplace-node
- [ ] Verify no direct fetch to HuggingFace in app code

### Manual Tests
- [ ] Visit `/models` page - should show models
- [ ] Visit `/models/[slug]` page - should show model details
- [ ] Check browser network tab - no direct HF API calls from client
- [ ] Verify types work in IDE (autocomplete, type checking)

---

## 📊 Architecture Comparison

### Before (WRONG):
```
┌─────────────────────────────────┐
│   Next.js Marketplace App       │
│                                 │
│  ┌──────────────────────────┐  │
│  │ lib/huggingface.ts       │  │
│  │ (direct fetch)           │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
    https://huggingface.co/api
```

**Problems:**
- No shared logic
- No type safety
- Duplicate code
- Can't leverage Rust

---

### After (CORRECT):
```
┌─────────────────────────────────┐
│   Next.js Marketplace App       │
│                                 │
│  ┌──────────────────────────┐  │
│  │ Import from              │  │
│  │ @rbee/marketplace-node   │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
┌─────────────────────────────────┐
│   @rbee/marketplace-node        │
│   (Node.js wrapper)             │
│                                 │
│  ┌──────────────────────────┐  │
│  │ Centralized fetch logic  │  │
│  │ Type conversions         │  │
│  │ Error handling           │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
    https://huggingface.co/api
```

**Benefits:**
- Single source of truth
- Type safety
- Shared logic
- Easy to add filtering
- Consistent API

---

### Future (IDEAL - with napi-rs):
```
┌─────────────────────────────────┐
│   Next.js Marketplace App       │
│                                 │
│  ┌──────────────────────────┐  │
│  │ Import from              │  │
│  │ @rbee/marketplace-node   │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
┌─────────────────────────────────┐
│   @rbee/marketplace-node        │
│   (Node.js wrapper)             │
│                                 │
│  ┌──────────────────────────┐  │
│  │ Native Rust binding      │  │
│  │ (napi-rs)                │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
┌─────────────────────────────────┐
│   @rbee/marketplace-sdk         │
│   (Rust crate - native)         │
│                                 │
│  ┌──────────────────────────┐  │
│  │ HuggingFaceClient        │  │
│  │ (Rust HTTP client)       │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
              ↓
    https://huggingface.co/api
```

**Benefits:**
- Rust performance
- Rust safety
- Shared with Tauri
- Type safety end-to-end

---

## 🚀 Implementation Order

### Immediate (Phase 3 + 4 + 5): 2-3 hours
1. Create `huggingface.ts` in marketplace-node (centralize fetch)
2. Update `index.ts` in marketplace-node (implement functions)
3. Update marketplace app to use marketplace-node
4. Delete wrong files
5. Test SSG build

### Future (Phase 1 + 2): 4-6 hours
1. Add napi-rs binding to marketplace-sdk
2. Update marketplace-node to use native Rust client
3. Remove fetch logic from marketplace-node
4. Test performance improvement

---

## ✅ Success Criteria

### Phase 3-5 Complete (Quick Fix):
- ✅ marketplace-node has all functions implemented
- ✅ Marketplace app uses marketplace-node (no direct fetch)
- ✅ SSG build works
- ✅ Types work correctly
- ✅ No `lib/huggingface.ts` file exists

### Phase 1-2 Complete (Future - Native Rust):
- ✅ marketplace-node uses native Rust client
- ✅ Performance improvement (Rust HTTP client)
- ✅ Shared logic with Tauri app
- ✅ Type safety end-to-end

---

**TEAM-415 - Fix Marketplace Pipeline**  
**Priority:** 🔥 CRITICAL  
**Effort:** 2-3 hours (quick fix) or 6-9 hours (native Rust)  
**Next:** Start with Phase 3 (Quick Fix - Centralize fetch in marketplace-node)
