# TEAM-462: PAGINATION MASTER PLAN

**Date:** 2025-11-09  
**Status:** 🚀 READY TO IMPLEMENT  
**Objective:** Add pagination to HuggingFace and CivitAI list pages (SSG, no runtime API calls)

---

## 🎯 GOAL

**From:** 100 models per provider (no pagination)  
**To:** 300 models per provider across 3 pages (all SSG)

---

## 📋 IMPLEMENTATION STEPS

### STEP 1: Update HuggingFace API Client (5 min)

**File:** `/bin/79_marketplace_core/marketplace-node/src/huggingface.ts`

**Changes:**
```typescript
// Add offset support for pagination
export async function fetchHFModels(
  query: string | undefined,
  options: { 
    limit?: number
    offset?: number  // NEW - for pagination
    sort?: string
    filter?: string 
  } = {},
): Promise<HFModel[]> {
  const params = new URLSearchParams({
    ...(query && { search: query }),
    limit: String(options.limit || 20),
    ...(options.offset && { offset: String(options.offset) }),  // NEW
    ...(options.sort && { sort: options.sort }),
    ...(options.filter && { filter: options.filter }),
  })
  
  // ... rest of implementation
}
```

---

### STEP 2: Update CivitAI API Client (5 min)

**File:** `/bin/79_marketplace_core/marketplace-node/src/civitai.ts`

**Changes:**
```typescript
// Add page support for pagination
async function fetchCivitAIModels(options = {}) {
  const { 
    query, 
    limit = 20, 
    page = 1,  // NEW - CivitAI uses page numbers
    types = ['Checkpoint', 'LORA'], 
    sort = 'Most Downloaded', 
    nsfw = false 
  } = options
  
  const params = new URLSearchParams({
    limit: String(limit),
    page: String(page),  // NEW
    sort,
    nsfw: String(nsfw),
  })
  
  // ... rest of implementation
}
```

---

### STEP 3: Add Pagination to HuggingFace Main Page (15 min)

**File:** `/frontend/apps/marketplace/app/models/huggingface/page.tsx`

**Changes:**

```typescript
// Add searchParams to page props
interface PageProps {
  searchParams: Promise<{ page?: string }>
}

// Generate static params for pages 1-3
export async function generateStaticParams() {
  return [
    { page: '1' },
    { page: '2' },
    { page: '3' },
  ]
}

export default async function HuggingFaceModelsPage({ searchParams }: PageProps) {
  const params = await searchParams
  const currentPage = parseInt(params.page || '1')
  const perPage = 100
  const offset = (currentPage - 1) * perPage
  
  console.log(`[SSG] Fetching HuggingFace models page ${currentPage} (offset: ${offset})`)

  const hfModels = await listHuggingFaceModels({ 
    limit: perPage,
    offset: offset,
  })

  console.log(`[SSG] Showing ${hfModels.length} HuggingFace models (page ${currentPage})`)

  // ... rest of component
  
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* ... existing content ... */}
      
      {/* ADD PAGINATION UI */}
      <Pagination 
        currentPage={currentPage}
        totalPages={3}
        baseUrl="/models/huggingface"
      />
      
      <ModelTableWithRouting models={models} />
    </div>
  )
}
```

---

### STEP 4: Add Pagination to CivitAI Main Page (15 min)

**File:** `/frontend/apps/marketplace/app/models/civitai/page.tsx`

**Changes:**

```typescript
// Add searchParams to page props
interface PageProps {
  searchParams: Promise<{ page?: string }>
}

// Generate static params for pages 1-3
export async function generateStaticParams() {
  return [
    { page: '1' },
    { page: '2' },
    { page: '3' },
  ]
}

export default async function CivitAIModelsPage({ searchParams }: PageProps) {
  const params = await searchParams
  const currentPage = parseInt(params.page || '1')
  const perPage = 100
  
  console.log(`[SSG] Fetching CivitAI models page ${currentPage}`)

  const civitaiModels = await getCompatibleCivitaiModels({ 
    limit: perPage,
    page: currentPage,
  })

  console.log(`[SSG] Showing ${civitaiModels.length} CivitAI models (page ${currentPage})`)

  // ... rest of component
  
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* ... existing content ... */}
      
      {/* ADD PAGINATION UI */}
      <Pagination 
        currentPage={currentPage}
        totalPages={3}
        baseUrl="/models/civitai"
      />
      
      {/* Model grid */}
    </div>
  )
}
```

---

### STEP 5: Create Pagination Component (10 min)

**File:** `/frontend/apps/marketplace/components/Pagination.tsx` (NEW)

```typescript
import Link from 'next/link'

interface PaginationProps {
  currentPage: number
  totalPages: number
  baseUrl: string
}

export function Pagination({ currentPage, totalPages, baseUrl }: PaginationProps) {
  return (
    <div className="flex items-center justify-center gap-2 my-8">
      {/* Previous button */}
      {currentPage > 1 && (
        <Link
          href={`${baseUrl}?page=${currentPage - 1}`}
          className="px-4 py-2 border rounded hover:bg-muted"
        >
          ← Previous
        </Link>
      )}
      
      {/* Page numbers */}
      {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
        <Link
          key={page}
          href={`${baseUrl}?page=${page}`}
          className={`px-4 py-2 border rounded ${
            page === currentPage 
              ? 'bg-primary text-primary-foreground font-bold' 
              : 'hover:bg-muted'
          }`}
        >
          {page}
        </Link>
      ))}
      
      {/* Next button */}
      {currentPage < totalPages && (
        <Link
          href={`${baseUrl}?page=${currentPage + 1}`}
          className="px-4 py-2 border rounded hover:bg-muted"
        >
          Next →
        </Link>
      )}
    </div>
  )
}
```

---

### STEP 6: Update Index Exports (5 min)

**File:** `/bin/79_marketplace_core/marketplace-node/src/index.ts`

**Verify exports include pagination params:**

```typescript
export { fetchHFModels, fetchHFModel } from './huggingface'
export { fetchCivitAIModels, fetchCivitAIModel } from './civitai'

// Make sure listHuggingFaceModels and getCompatibleCivitaiModels 
// accept offset/page parameters
```

---

### STEP 7: Test Build (5 min)

```bash
cd frontend/apps/marketplace
pnpm run build
```

**Expected output:**
```
✓ Generating static pages (450/450)
```

**Pages generated:**
- HuggingFace: 3 main pages + 30 filter pages (10 filters × 3 pages) + 300 detail pages = 333 pages
- CivitAI: 3 main pages + 27 filter pages (9 filters × 3 pages) + 300 detail pages = 330 pages
- Workers: ~30 pages
- Other: ~5 pages
- **Total: ~700 pages**

---

## 🎯 SUCCESS CRITERIA

- [ ] HuggingFace page 1 shows models 1-100
- [ ] HuggingFace page 2 shows models 101-200
- [ ] HuggingFace page 3 shows models 201-300
- [ ] CivitAI page 1 shows models 1-100
- [ ] CivitAI page 2 shows models 101-200
- [ ] CivitAI page 3 shows models 201-300
- [ ] Pagination UI shows on all pages
- [ ] All pages are SSG (no force-dynamic)
- [ ] Build completes successfully
- [ ] Total ~700 static pages generated

---

## ⚠️ IMPORTANT NOTES

### DO NOT:
- ❌ Add `force-dynamic` - pagination is SSG
- ❌ Make runtime API calls - all data fetched at build time
- ❌ Use client-side pagination - pages are pre-rendered

### DO:
- ✅ Fetch different data for each page at BUILD TIME
- ✅ Pre-render all 3 pages as static HTML
- ✅ Use Next.js `searchParams` for page numbers
- ✅ Generate static params for all pages

---

## 📊 BEFORE vs AFTER

### Before
- 100 HuggingFace models (1 page)
- 100 CivitAI models (1 page)
- 255 total pages

### After
- 300 HuggingFace models (3 pages)
- 300 CivitAI models (3 pages)
- ~700 total pages
- All still SSG, no runtime API calls

---

## 🚀 EXECUTION ORDER

1. Update `huggingface.ts` (add offset)
2. Update `civitai.ts` (add page)
3. Create `Pagination.tsx` component
4. Update HuggingFace main page
5. Update CivitAI main page
6. Test build
7. Verify all pages work

**ESTIMATED TIME: 1 hour**

---

**NOW IMPLEMENT THIS. NO MORE ASKING. JUST DO IT.**
