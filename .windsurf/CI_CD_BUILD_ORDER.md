# CI/CD Build Order for SSG

**Date:** 2025-11-08  
**Status:** 🎯 REQUIRED FOR SSG  
**Priority:** CRITICAL

---

## 🎯 Correct Build Order

### Step 1: Deploy Worker Catalog to Cloudflare
```bash
# Deploy global worker catalog to production
cd bin/80-hono-worker-catalog
pnpm run deploy

# Result: https://workers.rbee.dev/workers (live API)
```

### Step 2: Build Marketplace (Fetches from Production)
```bash
# Set production API URL
export WORKER_CATALOG_URL=https://workers.rbee.dev

# Build marketplace (fetches workers from production API)
cd frontend/apps/marketplace
pnpm run build

# Result: Static HTML with real production data
```

### Step 3: Deploy Marketplace
```bash
# Deploy static site to Cloudflare Pages / Vercel / etc.
pnpm run deploy
```

---

## ❌ What Was Wrong

### Cross-Crate Import (Violates Architecture)
```tsx
// ❌ WRONG - Imports from another crate
import { WORKERS } from '@/../../../bin/80-hono-worker-catalog/src/data'
```

**Problems:**
- Violates "no cross-crate imports" rule
- Tight coupling between crates
- Can't deploy independently
- Data duplication

### Why It Seemed to Work Locally
- Both crates in same repo
- TypeScript can resolve the path
- No build errors
- But violates architecture principles!

---

## ✅ Correct Architecture

### Production API Fetch
```tsx
// ✅ RIGHT - Fetches from production API
const WORKER_CATALOG_URL = process.env.WORKER_CATALOG_URL || 'https://workers.rbee.dev'

async function fetchWorkers() {
  const response = await fetch(`${WORKER_CATALOG_URL}/workers`)
  const data = await response.json()
  return data.workers
}
```

**Benefits:**
- ✅ No cross-crate dependencies
- ✅ Worker catalog deployed independently
- ✅ Marketplace fetches real production data
- ✅ Single source of truth (production API)
- ✅ Proper separation of concerns

---

## 🔧 Environment Variables

### Local Development
```bash
# .env.local (marketplace)
WORKER_CATALOG_URL=http://localhost:8787  # Local Wrangler dev server
```

### CI/CD
```bash
# GitHub Actions / CI environment
WORKER_CATALOG_URL=https://workers.rbee.dev  # Production API
```

### Production
```bash
# Vercel / Cloudflare Pages
WORKER_CATALOG_URL=https://workers.rbee.dev  # Production API
```

---

## 📋 GitHub Actions Workflow

### .github/workflows/deploy.yml
```yaml
name: Deploy rbee Marketplace

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  # STEP 1: Deploy Worker Catalog First
  deploy-worker-catalog:
    name: Deploy Worker Catalog to Cloudflare
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Deploy Worker Catalog
        working-directory: bin/80-hono-worker-catalog
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
        run: pnpm run deploy
      
      - name: Wait for deployment
        run: sleep 30  # Give Cloudflare time to propagate
      
      - name: Verify deployment
        run: |
          curl -f https://workers.rbee.dev/workers || exit 1
          echo "✅ Worker catalog is live"

  # STEP 2: Build Marketplace (Depends on Worker Catalog)
  build-marketplace:
    name: Build Marketplace SSG
    runs-on: ubuntu-latest
    needs: deploy-worker-catalog  # Wait for worker catalog
    steps:
      - uses: actions/checkout@v4
      
      - uses: pnpm/action-setup@v2
        with:
          version: 8
      
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Build Marketplace
        working-directory: frontend/apps/marketplace
        env:
          WORKER_CATALOG_URL: https://workers.rbee.dev  # Production API
        run: pnpm run build
      
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: marketplace-build
          path: frontend/apps/marketplace/.next

  # STEP 3: Deploy Marketplace
  deploy-marketplace:
    name: Deploy Marketplace to Vercel
    runs-on: ubuntu-latest
    needs: build-marketplace
    steps:
      - uses: actions/checkout@v4
      
      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: marketplace-build
          path: frontend/apps/marketplace/.next
      
      - name: Deploy to Vercel
        uses: amondnet/vercel-action@v25
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
          vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
          working-directory: frontend/apps/marketplace
```

---

## 🔍 Verification

### Check Build Logs
```bash
# Should see:
[INFO] Fetching workers from https://workers.rbee.dev/workers
[INFO] Found 8 workers
[INFO] Generating static pages...
[INFO] ✓ Static pages generated
```

### Check Build Output
```bash
# Should see:
Route (app)                                 Size  First Load JS    
├ ○ /workers                               2.1 kB        234 kB  ✅ Static
├ ○ /workers/llm-worker-rbee-cpu           1.8 kB        232 kB  ✅ Static
└ ○ /workers/llm-worker-rbee-cuda          1.8 kB        232 kB  ✅ Static

○  (Static)  prerendered as static content
```

### Test Production
```bash
# Worker catalog should be live
curl https://workers.rbee.dev/workers

# Marketplace should have static pages
curl https://marketplace.rbee.dev/workers
```

---

## 🚨 Common Issues

### Issue 1: Worker Catalog Not Deployed
```
Error: Failed to fetch workers: ECONNREFUSED
```

**Solution:** Deploy worker catalog first!
```bash
cd bin/80-hono-worker-catalog && pnpm run deploy
```

### Issue 2: Wrong API URL
```
Error: Failed to fetch workers: 404
```

**Solution:** Check `WORKER_CATALOG_URL` environment variable
```bash
echo $WORKER_CATALOG_URL
# Should be: https://workers.rbee.dev
```

### Issue 3: API Not Ready
```
Error: Failed to fetch workers: 503
```

**Solution:** Wait for Cloudflare propagation (30-60 seconds)
```bash
sleep 30
curl https://workers.rbee.dev/workers
```

---

## 📊 Data Flow

```
┌─────────────────────────────────────────────────────┐
│ STEP 1: Deploy Worker Catalog                      │
│                                                     │
│  bin/80-hono-worker-catalog/src/data.ts            │
│  ↓ (deploy to Cloudflare Workers)                  │
│  https://workers.rbee.dev/workers                  │
│  (Production API - Single Source of Truth)         │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ STEP 2: Build Marketplace                          │
│                                                     │
│  frontend/apps/marketplace/app/workers/page.tsx    │
│  ↓ (fetch at build time)                           │
│  GET https://workers.rbee.dev/workers              │
│  ↓ (SSG with real data)                            │
│  Static HTML pages                                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ STEP 3: Deploy Marketplace                         │
│                                                     │
│  Static HTML → Vercel/Cloudflare Pages            │
│  https://marketplace.rbee.dev/workers              │
│  (Instant load, pre-rendered)                      │
└─────────────────────────────────────────────────────┘
```

---

## ✅ Benefits

### Architectural
- ✅ No cross-crate dependencies
- ✅ Proper separation of concerns
- ✅ Independent deployment
- ✅ Single source of truth

### Operational
- ✅ Worker catalog can be updated independently
- ✅ Marketplace rebuilds fetch latest data
- ✅ No data duplication
- ✅ Clear deployment order

### Performance
- ✅ Static pages (instant load)
- ✅ CDN-served
- ✅ No runtime API calls
- ✅ SEO-optimized

---

## 🎯 Local Development

### Option 1: Use Production API
```bash
# .env.local
WORKER_CATALOG_URL=https://workers.rbee.dev

# Build marketplace
pnpm --filter @rbee/marketplace build
```

### Option 2: Run Local Worker Catalog
```bash
# Terminal 1: Start worker catalog
cd bin/80-hono-worker-catalog
pnpm run dev  # Runs on http://localhost:8787

# Terminal 2: Build marketplace
cd frontend/apps/marketplace
WORKER_CATALOG_URL=http://localhost:8787 pnpm run build
```

---

## 📝 Checklist

### Before CI/CD
- [ ] Worker catalog has `pnpm run deploy` script
- [ ] Cloudflare credentials in GitHub Secrets
- [ ] `WORKER_CATALOG_URL` env var configured
- [ ] Marketplace fetches from API (not import)

### CI/CD Setup
- [ ] GitHub Actions workflow created
- [ ] Deploy worker catalog job
- [ ] Build marketplace job (depends on worker catalog)
- [ ] Deploy marketplace job
- [ ] Proper job dependencies (`needs:`)

### Verification
- [ ] Worker catalog deploys successfully
- [ ] API is accessible (curl test)
- [ ] Marketplace build fetches from API
- [ ] Static pages generated
- [ ] No cross-crate imports

---

**Status:** ✅ Architecture fixed - No more cross-crate imports  
**Next:** Set up CI/CD workflow with correct build order
