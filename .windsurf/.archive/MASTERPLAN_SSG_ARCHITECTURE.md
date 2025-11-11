# Master Plan: SSG Architecture for Marketplace

**Date:** 2025-11-08  
**Status:** 📋 PLANNING PHASE  
**Priority:** CRITICAL - Foundation for all marketplace SSG

---

## 🎯 Goals

1. **No Cross-Crate Imports** - Strict architectural boundaries
2. **Production API as Source of Truth** - Single source, no duplication
3. **CI/CD Compatible** - Builds work without local services
4. **Fast Builds** - Efficient data fetching
5. **Resilient** - Graceful degradation on API failures

---

## 📊 Current State Analysis

### What We Have

#### 1. Worker Catalog (Hono API)
- **Location:** `bin/80-hono-worker-catalog/`
- **Data Source:** `src/data.ts` (static TypeScript)
- **API Endpoints:**
  - `GET /workers` - List all workers
  - `GET /workers/:id` - Get worker details
  - `GET /workers/:id/PKGBUILD` - Get build file
- **Deployment:** Cloudflare Workers
- **Production URL:** TBD (needs to be set)

#### 2. Marketplace (Next.js SSG)
- **Location:** `frontend/apps/marketplace/`
- **Pages:**
  - `/workers` - Workers list (currently uses cross-crate import ❌)
  - `/workers/[workerId]` - Worker details
  - `/models/huggingface` - HuggingFace models
  - `/models/civitai` - Civitai models
- **Build:** Next.js static generation
- **Deployment:** Vercel / Cloudflare Pages

#### 3. Marketplace SDK (Rust + WASM)
- **Location:** `bin/79_marketplace_core/marketplace-sdk/`
- **Clients:**
  - `HuggingFaceClient` - Fetches from HuggingFace API
  - `CivitaiClient` - Fetches from Civitai API
  - `WorkerCatalogClient` - Fetches from worker catalog
- **Compilation:** WASM for browser, native for Tauri

#### 4. Marketplace Node (Node.js wrapper)
- **Location:** `bin/79_marketplace_core/marketplace-node/`
- **Purpose:** Wraps marketplace-sdk WASM for Node.js
- **Used By:** Next.js at build time

### What's Broken

1. **Workers Page:** Cross-crate import violates architecture
2. **Models Pages:** Unknown if they call external APIs at build time
3. **CI/CD:** No defined build order
4. **Environment Variables:** Not standardized
5. **Error Handling:** No fallback strategy

---

## 🏗️ Architectural Principles

### 1. Strict Crate Boundaries
```
❌ NEVER: Import from another crate's source
✅ ALWAYS: Use published APIs (HTTP, npm packages)
```

### 2. Single Source of Truth
```
Production API → Build-Time Fetch → Static Pages
```

### 3. Data Flow Direction
```
Source Data → API → Consumer
(Never: Consumer → Source Files)
```

### 4. Deployment Independence
```
Each service deploys independently
No build-time dependencies on local services
```

---

## 📋 Master Plan

### Phase 1: Infrastructure Setup

#### 1.1 Define Production URLs
```bash
# Environment variables needed
WORKER_CATALOG_URL=https://workers-catalog.rbee.dev
HUGGINGFACE_CACHE_URL=https://cache.rbee.dev/huggingface  # Optional
CIVITAI_CACHE_URL=https://cache.rbee.dev/civitai          # Optional
```

**Decision Points:**
- [ ] What's the production URL for worker catalog?
- [ ] Do we cache HuggingFace/Civitai data?
- [ ] Where do we host the cache?

#### 1.2 Deploy Worker Catalog
```bash
# Deploy to Cloudflare Workers
cd bin/80-hono-worker-catalog
pnpm run deploy

# Verify
curl https://workers-catalog.rbee.dev/workers
```

**Tasks:**
- [ ] Set up Cloudflare Workers project
- [ ] Configure custom domain
- [ ] Add deployment script
- [ ] Test API endpoints

#### 1.3 Set Up CI/CD Secrets
```yaml
# GitHub Secrets needed
CLOUDFLARE_API_TOKEN: xxx
CLOUDFLARE_ACCOUNT_ID: xxx
VERCEL_TOKEN: xxx
VERCEL_ORG_ID: xxx
VERCEL_PROJECT_ID: xxx
```

**Tasks:**
- [ ] Create Cloudflare API token
- [ ] Create Vercel project
- [ ] Add secrets to GitHub

---

### Phase 2: Data Fetching Strategy

#### 2.1 Workers Data (Simple)
```tsx
// At build time
const response = await fetch(`${WORKER_CATALOG_URL}/workers`)
const { workers } = await response.json()

// Generate static pages
return <WorkersPage workers={workers} />
```

**Characteristics:**
- ✅ Small dataset (~10 workers)
- ✅ Rarely changes
- ✅ Fast API response
- ✅ No rate limits
- ✅ We control the API

**Strategy:** Direct fetch from production API

#### 2.2 HuggingFace Models (Complex)
```tsx
// At build time - calls external API
const models = await listHuggingFaceModels({ limit: 100 })
```

**Characteristics:**
- ⚠️ Large dataset (millions of models)
- ⚠️ Frequently changes
- ⚠️ External API (not under our control)
- ⚠️ Rate limits possible
- ⚠️ Slow API response

**Strategy Options:**

**Option A: Direct Fetch (Current?)**
```tsx
// Pros: Always fresh data
// Cons: Slow builds, rate limits, external dependency
const models = await fetch('https://huggingface.co/api/models?limit=100')
```

**Option B: Cached Data Service**
```tsx
// Pros: Fast builds, no rate limits, resilient
// Cons: Need to maintain cache service
const models = await fetch('https://cache.rbee.dev/huggingface/models')
```

**Option C: Static Data File**
```tsx
// Pros: Fastest builds, no external deps
// Cons: Stale data, manual updates
import { MODELS } from './cached-models.json'
```

**Recommendation:** Option B (Cached Data Service)

#### 2.3 Civitai Models (Complex)
```tsx
// At build time - calls external API
const models = await getCompatibleCivitaiModels()
```

**Characteristics:**
- ⚠️ Large dataset
- ⚠️ Frequently changes
- ⚠️ External API
- ⚠️ Rate limits
- ⚠️ NSFW content filtering needed

**Strategy:** Same as HuggingFace (Option B - Cached Data Service)

---

### Phase 3: Implementation Plan

#### 3.1 Workers Page (Simple - Do First)

**Current State:**
```tsx
// ❌ Cross-crate import
import { WORKERS } from '@/../../../bin/80-hono-worker-catalog/src/data'
```

**Target State:**
```tsx
// ✅ Production API fetch
const WORKER_CATALOG_URL = process.env.WORKER_CATALOG_URL || 'https://workers-catalog.rbee.dev'

async function fetchWorkers() {
  const response = await fetch(`${WORKER_CATALOG_URL}/workers`, {
    next: { revalidate: 3600 } // Cache for 1 hour
  })
  
  if (!response.ok) {
    console.error(`Failed to fetch workers: ${response.status}`)
    return [] // Graceful degradation
  }
  
  const data = await response.json()
  return data.workers || []
}

export default async function WorkersPage() {
  const workers = await fetchWorkers()
  // ... render
}
```

**Tasks:**
- [ ] Remove cross-crate import
- [ ] Add `fetchWorkers()` function
- [ ] Add error handling
- [ ] Add environment variable
- [ ] Test locally with production API
- [ ] Test locally with local API
- [ ] Update `.env.example`

**Files to Modify:**
- `frontend/apps/marketplace/app/workers/page.tsx`
- `frontend/apps/marketplace/.env.example`

#### 3.2 Models Pages (Complex - Do After Cache Service)

**Investigation Needed:**
```bash
# Check if these call external APIs
grep -r "fetch\|http" bin/79_marketplace_core/marketplace-sdk/src/huggingface.rs
grep -r "fetch\|http" bin/79_marketplace_core/marketplace-sdk/src/civitai.rs
```

**Questions to Answer:**
1. Do `listHuggingFaceModels()` and `getCompatibleCivitaiModels()` make HTTP requests?
2. If yes, to which URLs?
3. Are there rate limits?
4. How long do the requests take?
5. Can we cache the responses?

**Decision Tree:**
```
IF models pages call external APIs:
  IF APIs are fast and reliable:
    → Keep direct fetch (add retry logic)
  ELSE:
    → Build cache service
ELSE:
  → No changes needed
```

---

### Phase 4: Cache Service (If Needed)

#### 4.1 Architecture

```
┌─────────────────────────────────────────────────────┐
│ Cache Service (Cloudflare Workers)                  │
│                                                     │
│ GET /huggingface/models                            │
│ GET /civitai/models                                │
│                                                     │
│ - Caches responses in KV storage                   │
│ - Updates every 6 hours                            │
│ - Serves stale data if API down                    │
└─────────────────────────────────────────────────────┘
         ↓ (fetch at build time)
┌─────────────────────────────────────────────────────┐
│ Marketplace Build                                   │
│                                                     │
│ - Fast builds (cached data)                        │
│ - No rate limits                                   │
│ - Resilient to API failures                        │
└─────────────────────────────────────────────────────┘
```

#### 4.2 Implementation

**Location:** `bin/81-marketplace-cache/` (new crate)

**Features:**
- Cloudflare Workers + KV storage
- Automatic refresh every 6 hours
- Stale-while-revalidate pattern
- CORS enabled
- Rate limit protection

**Endpoints:**
```
GET /huggingface/models?limit=100
GET /civitai/models?limit=100
GET /health
```

**Tasks:**
- [ ] Create new crate `bin/81-marketplace-cache/`
- [ ] Implement cache service
- [ ] Set up Cloudflare KV
- [ ] Deploy to production
- [ ] Update marketplace to use cache

---

### Phase 5: CI/CD Pipeline

#### 5.1 Build Order

```yaml
jobs:
  # JOB 1: Deploy Worker Catalog
  deploy-worker-catalog:
    - Deploy to Cloudflare Workers
    - Verify API is live
    - Output: https://workers-catalog.rbee.dev

  # JOB 2: Deploy Cache Service (if needed)
  deploy-cache-service:
    needs: [deploy-worker-catalog]
    - Deploy to Cloudflare Workers
    - Verify API is live
    - Output: https://cache.rbee.dev

  # JOB 3: Build Marketplace
  build-marketplace:
    needs: [deploy-worker-catalog, deploy-cache-service]
    env:
      WORKER_CATALOG_URL: https://workers-catalog.rbee.dev
      HUGGINGFACE_CACHE_URL: https://cache.rbee.dev/huggingface
      CIVITAI_CACHE_URL: https://cache.rbee.dev/civitai
    - Fetch data from production APIs
    - Generate static pages
    - Output: Static HTML

  # JOB 4: Deploy Marketplace
  deploy-marketplace:
    needs: [build-marketplace]
    - Deploy to Vercel/Cloudflare Pages
    - Output: https://marketplace.rbee.dev
```

#### 5.2 Environment Variables

**Build Time (CI/CD):**
```bash
WORKER_CATALOG_URL=https://workers-catalog.rbee.dev
HUGGINGFACE_CACHE_URL=https://cache.rbee.dev/huggingface
CIVITAI_CACHE_URL=https://cache.rbee.dev/civitai
```

**Local Development:**
```bash
# Option 1: Use production APIs
WORKER_CATALOG_URL=https://workers-catalog.rbee.dev

# Option 2: Use local APIs
WORKER_CATALOG_URL=http://localhost:8787
```

---

## 🔍 Investigation Tasks

### Before Implementation

#### Task 1: Check HuggingFace Client
```bash
# Check if it makes HTTP requests
cat bin/79_marketplace_core/marketplace-sdk/src/huggingface.rs | grep -A 10 "pub async fn list_models"

# Check marketplace-node wrapper
cat bin/79_marketplace_core/marketplace-node/src/index.ts | grep -A 10 "listHuggingFaceModels"

# Test locally
cd frontend/apps/marketplace
pnpm run build 2>&1 | grep -i "huggingface\|fetch\|http"
```

**Questions:**
- [ ] Does it call `https://huggingface.co/api/...`?
- [ ] How long does it take?
- [ ] Does it work without API keys?
- [ ] Are there rate limits?

#### Task 2: Check Civitai Client
```bash
# Check if it makes HTTP requests
cat bin/79_marketplace_core/marketplace-sdk/src/civitai.rs | grep -A 10 "pub async fn get_compatible_models"

# Test locally
cd frontend/apps/marketplace
pnpm run build 2>&1 | grep -i "civitai\|fetch\|http"
```

**Questions:**
- [ ] Does it call `https://civitai.com/api/...`?
- [ ] How long does it take?
- [ ] Are there rate limits?
- [ ] Does it filter NSFW content?

#### Task 3: Test Current Build
```bash
# Clean build to see all network requests
cd frontend/apps/marketplace
rm -rf .next
pnpm run build 2>&1 | tee build.log

# Check for external API calls
grep -i "fetch\|http\|api" build.log
```

**Expected Output:**
- List of all HTTP requests made during build
- Time taken for each request
- Any errors or rate limit warnings

---

## 📊 Decision Matrix

### Workers Data
| Criteria | Direct Fetch | Cache Service | Static File |
|----------|-------------|---------------|-------------|
| **Freshness** | ✅ Always fresh | ⚠️ 6hr delay | ❌ Manual | 
| **Build Speed** | ✅ Fast (<1s) | ✅ Fast (<1s) | ✅ Instant |
| **Reliability** | ✅ We control | ✅ We control | ✅ Always works |
| **Complexity** | ✅ Simple | ⚠️ Medium | ✅ Simple |
| **Maintenance** | ✅ None | ⚠️ Cache service | ❌ Manual updates |

**Decision:** ✅ Direct Fetch from Production API

### Models Data
| Criteria | Direct Fetch | Cache Service | Static File |
|----------|-------------|---------------|-------------|
| **Freshness** | ✅ Always fresh | ⚠️ 6hr delay | ❌ Manual |
| **Build Speed** | ❌ Slow (10-30s) | ✅ Fast (<1s) | ✅ Instant |
| **Reliability** | ❌ External API | ✅ We control | ✅ Always works |
| **Complexity** | ✅ Simple | ⚠️ Cache service | ✅ Simple |
| **Maintenance** | ⚠️ Rate limits | ⚠️ Cache service | ❌ Manual updates |

**Decision:** ⚠️ TBD (Need investigation results)

---

## 🎯 Implementation Phases

### Phase 1: Workers (Week 1)
- [ ] Deploy worker catalog to production
- [ ] Fix workers page (remove cross-crate import)
- [ ] Test locally
- [ ] Test in CI
- [ ] Deploy to production

**Success Criteria:**
- Workers page builds without cross-crate imports
- Build works in CI without local services
- Page loads with real production data

### Phase 2: Investigation (Week 1)
- [ ] Investigate HuggingFace client
- [ ] Investigate Civitai client
- [ ] Measure build times
- [ ] Document findings
- [ ] Make decision on cache service

**Success Criteria:**
- Know exactly what APIs are called
- Know build time impact
- Have clear decision on architecture

### Phase 3: Models (Week 2)
**If cache service needed:**
- [ ] Build cache service
- [ ] Deploy to production
- [ ] Update models pages
- [ ] Test and deploy

**If direct fetch OK:**
- [ ] Add retry logic
- [ ] Add error handling
- [ ] Test and deploy

**Success Criteria:**
- Models pages build reliably
- Build time < 5 minutes
- Resilient to API failures

### Phase 4: CI/CD (Week 2)
- [ ] Create GitHub Actions workflow
- [ ] Set up secrets
- [ ] Test full pipeline
- [ ] Document process

**Success Criteria:**
- Automated deployment works
- Correct build order enforced
- All pages are static (○)

---

## ✅ Success Criteria

### Technical
- [ ] No cross-crate imports anywhere
- [ ] All marketplace pages are static (○)
- [ ] Build works in CI without local services
- [ ] Build time < 5 minutes
- [ ] Graceful error handling

### Architectural
- [ ] Strict crate boundaries maintained
- [ ] Single source of truth (production APIs)
- [ ] Independent deployment
- [ ] Clear data flow

### Operational
- [ ] Documented build process
- [ ] CI/CD pipeline working
- [ ] Environment variables standardized
- [ ] Monitoring in place

---

## 📝 Open Questions

### Critical (Need Answers Before Implementation)
1. **What's the production URL for worker catalog?**
   - Suggestion: `workers-catalog.rbee.dev`
   - Who sets this up?

2. **Do models pages call external APIs at build time?**
   - Need to investigate
   - Affects architecture decision

3. **Do we need a cache service?**
   - Depends on investigation results
   - Adds complexity but improves reliability

### Important (Can Decide During Implementation)
4. **How often should cache refresh?**
   - Suggestion: Every 6 hours
   - Balance freshness vs API load

5. **What's the fallback strategy if APIs fail?**
   - Return empty array?
   - Use stale cache?
   - Fail the build?

6. **Should we add monitoring?**
   - Track build times
   - Track API failures
   - Alert on issues

---

## 🚀 Next Steps

### Immediate (This Week)
1. **Get answers to critical questions**
   - Production URL for worker catalog
   - Investigate models data fetching

2. **Start Phase 1 (Workers)**
   - Deploy worker catalog
   - Fix workers page
   - Test locally

3. **Start Phase 2 (Investigation)**
   - Check HuggingFace client
   - Check Civitai client
   - Document findings

### Short-term (Next Week)
4. **Implement Phase 3 (Models)**
   - Based on investigation results
   - Build cache service if needed
   - Update models pages

5. **Implement Phase 4 (CI/CD)**
   - Create workflow
   - Test pipeline
   - Deploy to production

---

**Status:** 📋 MASTER PLAN COMPLETE - Ready for review and approval  
**Next:** Get answers to critical questions, then start Phase 1
