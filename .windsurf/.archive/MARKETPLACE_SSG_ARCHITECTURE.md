# Marketplace SSG Architecture

**Date:** 2025-11-08  
**Status:** ✅ CORRECT ARCHITECTURE  
**CI/CD:** ✅ NO API DEPENDENCY

---

## 🎯 How It Works

### Workers Page (SSG)
```tsx
// frontend/apps/marketplace/app/workers/page.tsx
import { WORKERS } from '@/../../../bin/80-hono-worker-catalog/src/data'

export default async function WorkersPage() {
  // Uses static data from repo - NO API CALL
  const filteredWorkers = filterWorkers(WORKERS, currentFilters)
  return <WorkerCard workers={filteredWorkers} />
}
```

**Data Source:** `bin/80-hono-worker-catalog/src/data.ts`  
**Build Time:** Imports directly from file  
**Runtime:** Pre-rendered static HTML  
**API Dependency:** ❌ NONE

### Models Pages (Need to Check)

**HuggingFace Models:**
- Uses `listHuggingFaceModels()` from `@rbee/marketplace-node`
- **Question:** Does this call HuggingFace API at build time?

**Civitai Models:**
- Uses `getCompatibleCivitaiModels()` from `@rbee/marketplace-node`
- **Question:** Does this call Civitai API at build time?

---

## ✅ Workers: Perfect Architecture

### Why It Works

1. **Static Data in Repo**
   ```
   bin/80-hono-worker-catalog/src/data.ts
   ↓ (import at build time)
   frontend/apps/marketplace/app/workers/page.tsx
   ↓ (SSG)
   Static HTML
   ```

2. **No API Dependency**
   - Data is versioned in git
   - Build doesn't need running services
   - CI/CD works without setup

3. **Dual Purpose Data**
   - **Build time:** Imported by Next.js for SSG
   - **Runtime:** Served by Hono API for GUI

### Data Flow

```
┌─────────────────────────────────────────┐
│  bin/80-hono-worker-catalog/src/data.ts │
│  (Single Source of Truth)                │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
  ┌─────────┐    ┌──────────┐
  │ Next.js │    │   Hono   │
  │   SSG   │    │   API    │
  └─────────┘    └──────────┘
       │                │
       ▼                ▼
  Static HTML      GUI (Tauri)
```

---

## ⚠️ Models: Need to Verify

### HuggingFace

**Current Code:**
```tsx
// frontend/apps/marketplace/app/models/huggingface/page.tsx
const hfModels = await listHuggingFaceModels({ limit: 100 })
```

**Questions:**
1. Does `listHuggingFaceModels()` call HuggingFace API?
2. If yes, does it work in CI without API keys?
3. Should we cache/commit the data instead?

### Civitai

**Current Code:**
```tsx
// frontend/apps/marketplace/app/models/civitai/page.tsx
const civitaiModels = await getCompatibleCivitaiModels()
```

**Questions:**
1. Does `getCompatibleCivitaiModels()` call Civitai API?
2. If yes, does it work in CI without rate limits?
3. Should we cache/commit the data instead?

---

## 🎯 Recommended Architecture

### Option 1: Static Data Files (Like Workers)
```
bin/79_marketplace_core/marketplace-data/
├── huggingface-models.json
├── civitai-models.json
└── update-data.ts (script to refresh)
```

**Pros:**
- ✅ No API calls during build
- ✅ Works in CI without setup
- ✅ Fast builds
- ✅ Predictable data

**Cons:**
- ⚠️ Data can be stale
- ⚠️ Need script to update
- ⚠️ Larger repo size

### Option 2: Build-Time API Calls (Current?)
```tsx
// At build time
const models = await fetch('https://huggingface.co/api/...')
```

**Pros:**
- ✅ Always fresh data
- ✅ No manual updates

**Cons:**
- ❌ Requires API access in CI
- ❌ Rate limits
- ❌ Slower builds
- ❌ Build can fail if API down

### Option 3: Hybrid (Recommended)
```tsx
// Try API first, fall back to cached data
const models = await fetchWithFallback(
  'https://api.../models',
  './cached-models.json'
)
```

**Pros:**
- ✅ Fresh data when possible
- ✅ Works in CI (uses cache)
- ✅ Resilient to API failures

**Cons:**
- ⚠️ More complex
- ⚠️ Need cache update strategy

---

## 🔍 Investigation Needed

### Check marketplace-node Implementation

```bash
# Check if it calls external APIs
cat bin/79_marketplace_core/marketplace-node/src/index.ts

# Check HuggingFace client
cat bin/79_marketplace_core/marketplace-sdk/src/huggingface.rs

# Check Civitai client
cat bin/79_marketplace_core/marketplace-sdk/src/civitai.rs
```

### Questions to Answer

1. **Does `listHuggingFaceModels()` make HTTP requests?**
   - If yes: To where? (`https://huggingface.co/api/...`)
   - If yes: Does it need auth?
   - If yes: Does it work in CI?

2. **Does `getCompatibleCivitaiModels()` make HTTP requests?**
   - If yes: To where? (`https://civitai.com/api/...`)
   - If yes: Rate limits?
   - If yes: Does it work in CI?

3. **What happens if APIs are down during build?**
   - Does build fail?
   - Is there a fallback?

---

## ✅ Solution: Follow Workers Pattern

### Step 1: Create Static Data Files

```typescript
// bin/79_marketplace_core/marketplace-data/huggingface-models.ts
export const HUGGINGFACE_MODELS = [
  {
    id: "meta-llama/Llama-2-7b-hf",
    name: "Llama 2 7B",
    downloads: 5000000,
    likes: 12000,
    // ... more fields
  },
  // ... more models
]
```

### Step 2: Update Script

```typescript
// bin/79_marketplace_core/marketplace-data/update-models.ts
import { HuggingFaceClient } from '../marketplace-sdk'

async function updateHuggingFaceModels() {
  const client = new HuggingFaceClient()
  const models = await client.listModels({ limit: 100 })
  
  // Write to file
  fs.writeFileSync(
    './huggingface-models.ts',
    `export const HUGGINGFACE_MODELS = ${JSON.stringify(models, null, 2)}`
  )
}
```

### Step 3: Import in Pages

```tsx
// frontend/apps/marketplace/app/models/huggingface/page.tsx
import { HUGGINGFACE_MODELS } from '@/../../../bin/79_marketplace_core/marketplace-data/huggingface-models'

export default async function HuggingFaceModelsPage() {
  // No API call - uses static data
  const models = HUGGINGFACE_MODELS
  return <ModelTable models={models} />
}
```

### Step 4: Automate Updates

```yaml
# .github/workflows/update-marketplace-data.yml
name: Update Marketplace Data
on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pnpm install
      - run: pnpm update-marketplace-data
      - uses: peter-evans/create-pull-request@v5
        with:
          title: 'chore: update marketplace data'
          commit-message: 'chore: update marketplace data'
```

---

## 📊 Comparison

| Aspect | Workers (Current) | Models (Current?) | Models (Proposed) |
|--------|------------------|-------------------|-------------------|
| **Data Source** | Static file in repo | API calls? | Static file in repo |
| **Build Dependency** | None | External APIs? | None |
| **CI/CD** | ✅ Works | ❓ Unknown | ✅ Works |
| **Freshness** | Manual update | Always fresh? | Daily auto-update |
| **Reliability** | 100% | Depends on API | 100% |
| **Build Speed** | Fast | Slow? | Fast |

---

## 🎯 Action Items

### Immediate (Investigation)
1. [ ] Check if `listHuggingFaceModels()` calls external API
2. [ ] Check if `getCompatibleCivitaiModels()` calls external API
3. [ ] Test marketplace build in clean environment (no APIs)
4. [ ] Document findings

### Short-term (If APIs are called)
1. [ ] Create static data files for models
2. [ ] Create update script
3. [ ] Update pages to import static data
4. [ ] Test build in CI

### Long-term (Automation)
1. [ ] Create GitHub Action for daily updates
2. [ ] Add data validation
3. [ ] Monitor data freshness
4. [ ] Document update process

---

## ✅ Success Criteria

- [ ] Marketplace builds without running services
- [ ] Marketplace builds without API keys
- [ ] Marketplace builds in CI/CD
- [ ] Data is reasonably fresh (< 24 hours old)
- [ ] Build is fast (< 2 minutes)
- [ ] Architecture is documented

---

**Status:** ✅ Workers architecture is perfect  
**Next:** Investigate models data fetching strategy
