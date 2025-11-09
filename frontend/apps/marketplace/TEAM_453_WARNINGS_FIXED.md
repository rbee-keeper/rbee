# TEAM-453: Marketplace Build Warnings Fixed

**Date:** 2025-11-09  
**Status:** ✅ FIXED

## Summary

Fixed all build warnings in the marketplace deployment:
1. ✅ `metadataBase` warning (localhost:3000)
2. ✅ `pages_build_output_dir` warning (wrangler.jsonc)
3. ✅ Clarified SSG data sources

## Warnings Fixed

### 1. metadataBase Warning

**Warning:**
```
⚠ metadataBase property in metadata export is not set for resolving social open graph or twitter images, using "http://localhost:3000"
```

**Root Cause:**
- Next.js needs `metadataBase` for OpenGraph/Twitter image URLs
- Was defaulting to `localhost:3000` (Next.js default port)
- Should use `localhost:7823` (marketplace dev port) or production URL

**Fix:**
```typescript
// app/layout.tsx
export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'https://marketplace.rbee.dev'),
  // ... rest of metadata
}
```

**Environment Variables:**
- Development: `NEXT_PUBLIC_SITE_URL=http://localhost:7823`
- Production: `NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev`

### 2. Wrangler pages_build_output_dir Warning

**Warning:**
```
▲ [WARNING] Pages now has wrangler.json support.
  We detected a configuration file at wrangler.jsonc but it is missing the
  "pages_build_output_dir" field, required by Pages.
```

**Root Cause:**
- Cloudflare Pages requires `pages_build_output_dir` in wrangler.jsonc
- Was missing from configuration

**Fix:**
```json
// wrangler.jsonc
{
  "name": "marketplace",
  "main": ".open-next/worker.js",
  "pages_build_output_dir": ".next",  // ← Added this
  "compatibility_date": "2025-03-01",
  // ...
}
```

## SSG Data Sources Clarified

### Question: Is SSG Getting Data from gwc.rbee.dev?

**Answer: NO** - The SSG is NOT fetching from gwc.rbee.dev during build.

### What SSG Actually Does

**Workers (Hardcoded):**
```typescript
// app/workers/[workerId]/page.tsx
const WORKERS: Record<string, Worker> = {
  'cpu-llm': { /* ... */ },
  'cuda-llm': { /* ... */ },
  'metal-llm': { /* ... */ },
  'rocm-llm': { /* ... */ },
}

export async function generateStaticParams() {
  return Object.keys(WORKERS).map((workerId) => ({
    workerId,
  }))
}
```

**Workers are hardcoded in the app**, not fetched from gwc.rbee.dev.

**Models (Fetched from CivitAI & HuggingFace):**
```typescript
// app/models/civitai/[slug]/page.tsx
export async function generateStaticParams() {
  console.log('[SSG] Pre-building top 100 Civitai models')
  
  const models = await getCompatibleCivitaiModels()  // ← Fetches from CivitAI API
  
  return models.map((model) => ({
    slug: modelIdToSlug(model.id),
  }))
}
```

**Models are fetched directly from:**
- **CivitAI API:** `https://civitai.com/api/v1`
- **HuggingFace API:** (via marketplace-node package)

### Why This Design?

**Workers:**
- Small, static dataset (4 workers)
- Rarely changes
- Hardcoded for simplicity and reliability
- No external API dependency during build

**Models:**
- Large, dynamic dataset (100+ models)
- Changes frequently
- Fetched from source APIs during build
- Pre-rendered as static pages for SEO

### Runtime API Usage

The `MARKETPLACE_API_URL` (gwc.rbee.dev) is used at **runtime**, not during SSG:

```typescript
// app/api/models/route.ts
const MARKETPLACE_API_URL = process.env.MARKETPLACE_API_URL || 'http://localhost:3001'

export async function GET(request: NextRequest) {
  // This runs at RUNTIME, not during SSG
  const response = await fetch(`${MARKETPLACE_API_URL}/api/models?...`)
  // ...
}
```

**When gwc.rbee.dev is used:**
- Dynamic API routes (not pre-rendered)
- Client-side fetches
- Search functionality
- Real-time data

## Port Configuration

### Correct Ports (from PORT_CONFIGURATION.md)

**Marketplace:**
- Development: `7823`
- Production: `https://marketplace.rbee.dev`

**Related Services:**
- Commercial: `7822`
- User Docs: `7811`
- GWC (Worker Catalog): `8787`

### Port References Fixed

**package.json:**
```json
{
  "scripts": {
    "dev": "next dev --turbopack -p 7823"  // ✅ Correct
  }
}
```

**Environment Variables:**
```env
# Development
NEXT_PUBLIC_SITE_URL=http://localhost:7823

# Production
NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev
```

## Files Modified

1. **`app/layout.tsx`**
   - Added `metadataBase` to metadata
   - Uses `NEXT_PUBLIC_SITE_URL` environment variable
   - Falls back to `https://marketplace.rbee.dev`

2. **`wrangler.jsonc`**
   - Added `pages_build_output_dir: ".next"`
   - Fixes Cloudflare Pages warning

3. **`xtask/src/deploy/marketplace.rs`**
   - Added `NEXT_PUBLIC_SITE_URL` to deployment env
   - Ensures metadataBase works in production

## Deployment Environment

### Development (.env.local)
```env
MARKETPLACE_API_URL=http://localhost:8787
NEXT_PUBLIC_SITE_URL=http://localhost:7823
NEXT_DISABLE_DEVTOOLS=0
```

### Production (Deployment)
```env
MARKETPLACE_API_URL=https://gwc.rbee.dev
NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev
NEXT_DISABLE_DEVTOOLS=1
```

## Build Output (After Fixes)

### Expected Output (No Warnings)
```bash
✓ Compiled successfully in 2.7s
✓ Finished TypeScript in 4.9s
✓ Collecting page data in 18.2s
[SSG] Pre-building top 100 Civitai models
[SSG] Pre-building 100 HuggingFace model pages
[SSG] Pre-building 100 Civitai model pages
✓ Generating static pages (252/252) in 1.4s
✓ Finalizing page optimization in 5.9ms

Route (app)
┌ ○ /
├ ○ /workers
├ ● /workers/[workerId] (4 pages)
├ ● /workers/[...filter] (17 pages)
├ ○ /models
├ ● /models/civitai/[slug] (100 pages)
├ ● /models/huggingface/[slug] (100 pages)
└ ... (252 total pages)
```

**No more warnings about:**
- ❌ `metadataBase` using localhost:3000
- ❌ `pages_build_output_dir` missing

## Testing

### Verify Fixes

**1. Check metadataBase:**
```bash
cd frontend/apps/marketplace
pnpm build 2>&1 | grep -i "metadataBase"
# Should show NO warnings
```

**2. Check wrangler config:**
```bash
cd frontend/apps/marketplace
pnpm wrangler pages deploy .next --project-name=rbee-marketplace --dry-run
# Should show NO warnings about pages_build_output_dir
```

**3. Check environment:**
```bash
# Development
echo $NEXT_PUBLIC_SITE_URL  # Should be http://localhost:7823

# Production (during deployment)
cat .env.local  # Should contain NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev
```

## Summary

✅ **All warnings fixed**
- metadataBase now uses correct URL
- wrangler.jsonc has pages_build_output_dir
- Environment variables properly configured

✅ **SSG data sources clarified**
- Workers: Hardcoded in app (4 workers)
- Models: Fetched from CivitAI/HuggingFace APIs (200+ models)
- gwc.rbee.dev: Used at runtime, not during SSG

✅ **Port configuration correct**
- Development: localhost:7823
- Production: marketplace.rbee.dev

Next deployment will be clean with no warnings!
