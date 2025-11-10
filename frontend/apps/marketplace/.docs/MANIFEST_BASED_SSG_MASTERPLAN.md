# Manifest-Based SSG Masterplan

**Date:** 2025-11-10  
**Status:** 🚧 IN PROGRESS  
**Goal:** Replace expensive filter/sort page prerendering with JSON manifest system

---

## Problem Statement

Currently, we prerender every filter/sort combination page (e.g., `/models/civitai/AllTime/All/All/downloads/Soft`), which creates:
- ❌ Hundreds of redundant pages
- ❌ Slow build times
- ❌ Broken routes when filter combinations change
- ❌ Duplicate model detail pages

## Solution Overview

**Build Time:**
1. Generate JSON manifests for each filter/sort combination
2. Merge all manifests and deduplicate model IDs
3. Prerender ONLY unique model detail pages
4. Filter/sort pages load manifests client-side

**Dev Time:**
- Skip manifest generation (too slow)
- Fetch live from APIs
- Fast development experience

---

## Architecture

```
Build Time (Production):
┌─────────────────────────────────────────────────────────┐
│ 1. Generate Manifests                                   │
│    ├─ Fetch all CivitAI filter combinations            │
│    ├─ Fetch all HuggingFace filter combinations        │
│    ├─ Save individual JSON manifests                    │
│    └─ Create combined all-models.json                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Next.js Build                                        │
│    ├─ Read all-models.json                             │
│    ├─ Generate static params for unique models only    │
│    └─ Prerender ~200-300 unique model pages            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Filter/Sort Pages (Client-Side)                     │
│    ├─ Load manifest JSON from /manifests/              │
│    ├─ Render model grid from manifest data             │
│    └─ No server-side rendering needed                  │
└─────────────────────────────────────────────────────────┘

Dev Time (Development):
┌─────────────────────────────────────────────────────────┐
│ Skip manifest generation                                │
│ All pages fetch live from APIs                          │
│ Fast dev server startup                                 │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### ✅ Phase 0: Foundation (COMPLETE)
- [x] Create manifest generator script
- [x] Add prebuild hook
- [x] Add tsx dependency

### 🚧 Phase 1: Manifest Generation (IN PROGRESS)
**File:** [PHASE_1_MANIFEST_GENERATION.md](./PHASE_1_MANIFEST_GENERATION.md)
- [ ] Implement CivitAI manifest fetching
- [ ] Implement HuggingFace manifest fetching
- [ ] Test manifest generation
- [ ] Verify JSON output

### 📋 Phase 2: Update generateStaticParams
**File:** [PHASE_2_STATIC_PARAMS.md](./PHASE_2_STATIC_PARAMS.md)
- [ ] Update CivitAI model page
- [ ] Update HuggingFace model page
- [ ] Update redirect page
- [ ] Remove filter page prerendering

### 📋 Phase 3: Client-Side Filter Pages
**File:** [PHASE_3_CLIENT_SIDE_FILTERS.md](./PHASE_3_CLIENT_SIDE_FILTERS.md)
- [ ] Create manifest loader utility
- [ ] Update CivitAI filter pages
- [ ] Update HuggingFace filter pages
- [ ] Add loading states

### 📋 Phase 4: Dev Mode Optimization
**File:** [PHASE_4_DEV_MODE.md](./PHASE_4_DEV_MODE.md)
- [ ] Skip manifest generation in dev
- [ ] Fallback to live API fetching
- [ ] Add dev mode indicators

### 📋 Phase 5: Testing & Validation
**File:** [PHASE_5_TESTING.md](./PHASE_5_TESTING.md)
- [ ] Test build process
- [ ] Verify manifest accuracy
- [ ] Test filter pages
- [ ] Performance benchmarks

### 📋 Phase 6: Deployment
**File:** [PHASE_6_DEPLOYMENT.md](./PHASE_6_DEPLOYMENT.md)
- [ ] Update CI/CD pipeline
- [ ] Deploy to staging
- [ ] Verify production build
- [ ] Deploy to production

---

## Success Metrics

**Before:**
- ~500+ pages prerendered
- 10+ minute build times
- Broken filter routes

**After:**
- ~200-300 unique model pages prerendered
- <5 minute build times
- All filter routes work correctly
- Faster page loads (static JSON)

---

## Files to Modify

### Scripts
- ✅ `scripts/generate-model-manifests.ts` (created)
- ✅ `package.json` (updated)

### Pages (generateStaticParams)
- `app/models/[slug]/page.tsx`
- `app/models/civitai/[slug]/page.tsx`
- `app/models/huggingface/[slug]/page.tsx`

### Filter Pages (client-side)
- `app/models/civitai/[...filter]/page.tsx`
- `app/models/huggingface/[...filter]/page.tsx`

### Utilities
- `lib/manifests.ts` (new)
- `lib/civitai.ts` (update)
- `lib/huggingface.ts` (update)

---

## Risk Mitigation

**Risk:** Manifest data becomes stale
**Mitigation:** Regenerate manifests on every build (fast API calls)

**Risk:** Dev mode too slow
**Mitigation:** Skip manifest generation, use live APIs

**Risk:** Missing models in manifests
**Mitigation:** Fallback to live API if model not in manifest

**Risk:** Build fails if API is down
**Mitigation:** Cache previous manifests, use as fallback

---

## Next Steps

1. Complete Phase 1: Manifest Generation
2. Test manifest output
3. Proceed to Phase 2: Update generateStaticParams

---

**See individual phase documents for detailed implementation steps.**
