# TEAM-410: Phase 4 - Next.js Integration

**Created:** 2025-11-05  
**Team:** TEAM-410  
**Duration:** 2-3 days  
**Status:** ✅ COMPLETE  
**Dependencies:** TEAM-409 complete (compatibility matrix data layer)

---

## 🎯 Mission

Integrate compatibility matrix into Next.js marketplace app: show compatible workers on model detail pages, add worker recommendations, and optimize for SSG/SEO.

---

## 🏗️ Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│ COMPATIBILITY INTEGRATION: SDK → Node → Next.js SSG        │
└─────────────────────────────────────────────────────────────┘

1. marketplace-sdk (Rust WASM)
   ├─ compatibility.rs (core logic)
   ├─ wasm_worker.rs (WASM bindings)
   └─ Compiled to: marketplace-node/wasm/

2. marketplace-node (TypeScript wrapper)
   ├─ Import WASM: import * as wasm from './wasm/marketplace_sdk'
   ├─ Export functions: checkModelCompatibility(), filterCompatibleModels()
   └─ Used by: Next.js at build time (SSG)

3. Next.js Marketplace (SSG)
   ├─ Import: import { listCompatibleModels } from '@rbee/marketplace-node'
   ├─ Build time: generateStaticParams() calls marketplace-node
   ├─ Output: Static HTML with compatibility data
   └─ Deploy: Cloudflare Pages

4. GitHub Actions (Cron Jobs)
   ├─ Schedule: Daily (0 0 * * *) for top 100 list
   ├─ Action: Fetch models, filter compatible, rebuild static pages
   ├─ Deploy: wrangler pages deploy dist/
   └─ Cost: $0/month (free tier)
```

**Key Points:**
- ✅ All compatibility logic in Rust (marketplace-sdk)
- ✅ WASM bindings for Node.js (marketplace-node)
- ✅ Next.js calls marketplace-node at BUILD TIME (SSG)
- ✅ GitHub Actions updates static pages daily
- ✅ No runtime compatibility checks (all pre-computed)

---

## ✅ Checklist

### Task 4.1: Add Compatibility Data to Model Detail Pages
- [x] Open `frontend/apps/marketplace/app/models/[slug]/page.tsx`
- [x] Import `getCompatibleWorkersForModel` from marketplace-node
- [x] Fetch compatible workers in `generateStaticParams()`
- [x] Pass workers to ModelDetailPageTemplate
- [x] Add TEAM-410 signatures
- [x] Commit: "TEAM-410: Add compatible workers to model detail pages"

**✅ IMPLEMENTATION COMPLETE** - marketplace-node wrapper ready

**Implementation:**
```tsx
// frontend/apps/marketplace/app/models/[slug]/page.tsx

import { getCompatibleWorkersForModel } from '@rbee/marketplace-node'

export default async function ModelPage({ params }: Props) {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const hfModel = await fetchModel(modelId)
    const model = transformToModelDetailData(hfModel)
    
    // TEAM-410: Fetch compatible workers
    const compatibleWorkers = await getCompatibleWorkersForModel(modelId)
    
    return (
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <ModelDetailPageTemplate 
          model={model} 
          compatibleWorkers={compatibleWorkers}
          showBackButton={false} 
        />
      </div>
    )
  } catch {
    notFound()
  }
}
```

**Acceptance:**
- ✅ Compatible workers fetched at build time (SSG)
- ✅ Data passed to template
- ✅ No runtime errors

---

### Task 4.2: Create CompatibilityBadge Component
- [ ] Create `frontend/packages/rbee-ui/src/marketplace/atoms/CompatibilityBadge.tsx`
- [ ] Show compatibility status (compatible, partial, incompatible)
- [ ] Color-coded badges (green, yellow, red)
- [ ] Tooltip with reasons/warnings
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add CompatibilityBadge component"

**Implementation:**
```tsx
// TEAM-410: Compatibility badge component

import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { CompatibilityResult } from '@rbee/marketplace-node'

interface CompatibilityBadgeProps {
  result: CompatibilityResult
  workerName: string
}

export function CompatibilityBadge({ result, workerName }: CompatibilityBadgeProps) {
  const variant = result.compatible ? 'success' : 'destructive'
  const label = result.compatible ? 'Compatible' : 'Incompatible'
  
  return (
    <Tooltip>
      <TooltipTrigger>
        <Badge variant={variant} className="cursor-help">
          {label}
        </Badge>
      </TooltipTrigger>
      <TooltipContent className="max-w-sm">
        <div className="space-y-2">
          <p className="font-semibold">{workerName}</p>
          
          {result.reasons.length > 0 && (
            <div>
              <p className="text-xs font-medium">Reasons:</p>
              <ul className="text-xs list-disc list-inside">
                {result.reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            </div>
          )}
          
          {result.warnings.length > 0 && (
            <div>
              <p className="text-xs font-medium text-yellow-500">Warnings:</p>
              <ul className="text-xs list-disc list-inside">
                {result.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  )
}
```

**Acceptance:**
- ✅ Badge shows correct status
- ✅ Tooltip shows details
- ✅ Accessible (keyboard navigation)

---

### Task 4.3: Create WorkerCompatibilityList Component
- [ ] Create `frontend/packages/rbee-ui/src/marketplace/organisms/WorkerCompatibilityList.tsx`
- [ ] Display list of workers with compatibility badges
- [ ] Group by compatibility status
- [ ] Show worker details (type, platform, capabilities)
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add WorkerCompatibilityList component"

**Implementation:**
```tsx
// TEAM-410: Worker compatibility list component

import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { CompatibilityBadge } from '../atoms/CompatibilityBadge'
import type { Worker, CompatibilityResult } from '@rbee/marketplace-node'

interface WorkerCompatibilityListProps {
  workers: Array<{
    worker: Worker
    compatibility: CompatibilityResult
  }>
}

export function WorkerCompatibilityList({ workers }: WorkerCompatibilityListProps) {
  // Group by compatibility
  const compatible = workers.filter(w => w.compatibility.compatible)
  const incompatible = workers.filter(w => !w.compatibility.compatible)
  
  return (
    <div className="space-y-6">
      {/* Compatible Workers */}
      {compatible.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Compatible Workers</h3>
          <div className="grid gap-3">
            {compatible.map(({ worker, compatibility }) => (
              <Card key={worker.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{worker.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {worker.worker_type} • {worker.platform}
                      </p>
                    </div>
                    <CompatibilityBadge 
                      result={compatibility} 
                      workerName={worker.name} 
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
      
      {/* Incompatible Workers */}
      {incompatible.length > 0 && (
        <div>
          <h3 className="text-lg font-semibold mb-3">Incompatible Workers</h3>
          <div className="grid gap-3 opacity-60">
            {incompatible.map(({ worker, compatibility }) => (
              <Card key={worker.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">{worker.name}</p>
                      <p className="text-sm text-muted-foreground">
                        {worker.worker_type} • {worker.platform}
                      </p>
                    </div>
                    <CompatibilityBadge 
                      result={compatibility} 
                      workerName={worker.name} 
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
```

**Acceptance:**
- ✅ Workers grouped correctly
- ✅ Shows all relevant info
- ✅ Responsive design

---

### Task 4.4: Update ModelDetailPageTemplate
- [ ] Open `frontend/packages/rbee-ui/src/marketplace/pages/ModelDetailPage.tsx`
- [ ] Add `compatibleWorkers` prop
- [ ] Add "Compatible Workers" section
- [ ] Use WorkerCompatibilityList component
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add compatibility section to ModelDetailPage"

**Implementation:**
```tsx
// TEAM-410: Updated ModelDetailPageTemplate with compatibility

import { WorkerCompatibilityList } from '../organisms/WorkerCompatibilityList'

interface ModelDetailPageProps {
  model: ModelDetailData
  compatibleWorkers?: Array<{
    worker: Worker
    compatibility: CompatibilityResult
  }>
  showBackButton?: boolean
}

export function ModelDetailPageTemplate({ 
  model, 
  compatibleWorkers,
  showBackButton = true 
}: ModelDetailPageProps) {
  return (
    <div className="space-y-8">
      {/* Existing model details */}
      <ModelHeader model={model} showBackButton={showBackButton} />
      <ModelStats model={model} />
      <ModelDescription model={model} />
      
      {/* TEAM-410: Compatible Workers Section */}
      {compatibleWorkers && compatibleWorkers.length > 0 && (
        <section>
          <h2 className="text-2xl font-bold mb-4">Compatible Workers</h2>
          <WorkerCompatibilityList workers={compatibleWorkers} />
        </section>
      )}
      
      {/* Existing sections */}
      <ModelFiles model={model} />
      <ModelTags model={model} />
    </div>
  )
}
```

**Acceptance:**
- ✅ Compatibility section renders
- ✅ SSG works (no client-side fetching)
- ✅ SEO-friendly (all in HTML)

---

### Task 4.5: Add Compatibility Filter to Model List
- [ ] Open `frontend/apps/marketplace/app/models/page.tsx`
- [ ] Add worker filter dropdown
- [ ] Filter models by compatible worker
- [ ] Update URL params for filtering
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add worker compatibility filter to model list"

**Implementation:**
```tsx
// TEAM-410: Model list with worker filter

import { listWorkerBinaries, getCompatibleModelsForWorker } from '@rbee/marketplace-node'

export default async function ModelsPage({
  searchParams,
}: {
  searchParams: Promise<{ worker?: string }>
}) {
  const { worker: selectedWorker } = await searchParams
  
  // Fetch all models
  const hfModels = await fetchTopModels(100)
  
  // Fetch workers for filter
  const workers = await listWorkerBinaries()
  
  // Filter by worker if selected
  let models = hfModels.map(transformToModelTableItem)
  if (selectedWorker) {
    const compatibleModels = await getCompatibleModelsForWorker(
      selectedWorker,
      models.map(m => ({ /* model metadata */ }))
    )
    models = models.filter(m => 
      compatibleModels.some(cm => cm.id === m.id)
    )
  }
  
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header */}
      <div className="mb-12 space-y-4">
        <h1 className="text-4xl md:text-5xl font-bold">LLM Models</h1>
        
        {/* TEAM-410: Worker Filter */}
        <WorkerFilterDropdown 
          workers={workers} 
          selected={selectedWorker} 
        />
      </div>
      
      {/* Model Table */}
      <ModelTableWithRouting models={models} />
    </div>
  )
}
```

**Acceptance:**
- ✅ Filter dropdown works
- ✅ Models filtered correctly
- ✅ URL updates on filter change
- ✅ SSG still works

---

### Task 4.6: Add SEO Metadata for Compatibility
- [ ] Update `generateMetadata()` in model detail page
- [ ] Add compatible workers to description
- [ ] Add structured data (JSON-LD)
- [ ] Add Open Graph tags
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add SEO metadata for compatibility"

**Implementation:**
```tsx
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { slug } = await params
  const modelId = slugToModelId(slug)
  
  try {
    const hfModel = await fetchModel(modelId)
    const model = transformToModelDetailData(hfModel)
    
    // TEAM-410: Fetch compatible workers for SEO
    const compatibleWorkers = await getCompatibleWorkersForModel(modelId)
    const workerNames = compatibleWorkers.map(w => w.name).join(', ')
    
    return {
      title: `${model.name} | AI Model`,
      description: `${model.description || model.name} - Compatible with: ${workerNames}`,
      openGraph: {
        title: model.name,
        description: `${model.downloads.toLocaleString()} downloads - Compatible workers: ${workerNames}`,
        type: 'article',
      },
      // JSON-LD structured data
      other: {
        'application/ld+json': JSON.stringify({
          '@context': 'https://schema.org',
          '@type': 'SoftwareApplication',
          name: model.name,
          description: model.description,
          applicationCategory: 'AI Model',
          compatibleWith: compatibleWorkers.map(w => ({
            '@type': 'SoftwareApplication',
            name: w.name,
          })),
        }),
      },
    }
  } catch {
    return { title: 'Model Not Found' }
  }
}
```

**Acceptance:**
- ✅ SEO metadata includes compatibility
- ✅ JSON-LD structured data valid
- ✅ Open Graph tags correct

---

### Task 4.7: Add Compatibility Matrix Page
- [ ] Create `frontend/apps/marketplace/app/compatibility/page.tsx`
- [ ] Show full compatibility matrix
- [ ] Interactive table (sortable, filterable)
- [ ] Export as CSV/JSON
- [ ] Add TEAM-410 signatures
- [ ] Commit: "TEAM-410: Add compatibility matrix page"

**Implementation:**
```tsx
// TEAM-410: Compatibility matrix page

import { generateCompatibilityMatrix, listWorkerBinaries } from '@rbee/marketplace-node'
import { CompatibilityMatrixTable } from '@/components/CompatibilityMatrixTable'

export default async function CompatibilityPage() {
  // Fetch all workers
  const workers = await listWorkerBinaries()
  
  // Fetch top models
  const models = await fetchTopModels(50)
  
  // Generate matrix
  const matrix = await generateCompatibilityMatrix(
    models.map(transformToModelMetadata),
    workers
  )
  
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      <h1 className="text-4xl font-bold mb-8">Compatibility Matrix</h1>
      <p className="text-muted-foreground mb-8">
        See which workers can run which models at a glance.
      </p>
      
      <CompatibilityMatrixTable matrix={matrix} />
    </div>
  )
}
```

**Acceptance:**
- ✅ Matrix page renders
- ✅ Table interactive
- ✅ Export functionality works
- ✅ SSG optimized

---

### Task 4.8: Write Component Tests
- [ ] Create tests for CompatibilityBadge
- [ ] Create tests for WorkerCompatibilityList
- [ ] Create tests for ModelDetailPage with compatibility
- [ ] Run `pnpm test` in rbee-ui
- [ ] Commit: "TEAM-410: Add compatibility component tests"

**Test Setup:**
```tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CompatibilityBadge } from '../CompatibilityBadge'

describe('CompatibilityBadge', () => {
  it('shows compatible badge', () => {
    const result = {
      compatible: true,
      confidence: 'high',
      reasons: ['Architecture and format compatible'],
      warnings: [],
      recommendations: [],
    }
    
    render(<CompatibilityBadge result={result} workerName="CPU Worker" />)
    expect(screen.getByText('Compatible')).toBeInTheDocument()
  })
  
  it('shows incompatible badge', () => {
    const result = {
      compatible: false,
      confidence: 'none',
      reasons: ['Format not supported'],
      warnings: [],
      recommendations: [],
    }
    
    render(<CompatibilityBadge result={result} workerName="CPU Worker" />)
    expect(screen.getByText('Incompatible')).toBeInTheDocument()
  })
})
```

**Acceptance:**
- ✅ All component tests pass
- ✅ Edge cases covered
- ✅ Accessibility tested

---

### Task 4.9: Update Documentation
- [ ] Update `frontend/apps/marketplace/README.md`
- [ ] Document compatibility features
- [ ] Add screenshots
- [ ] Update `frontend/packages/rbee-ui/README.md`
- [ ] Document new components
- [ ] Commit: "TEAM-410: Update Next.js integration documentation"

**README Example:**
```markdown
## Compatibility Matrix

### Model Detail Pages
Each model detail page shows compatible workers:
- ✅ Compatible workers (green badges)
- ❌ Incompatible workers (red badges)
- Tooltips explain why compatible/incompatible

### Worker Filter
Filter models by compatible worker in the model list page.

### Compatibility Matrix Page
View full compatibility matrix at `/compatibility`
```

**Acceptance:**
- ✅ Documentation updated
- ✅ Screenshots added
- ✅ Usage clear

---

### Task 4.10: Verification
- [ ] Run `pnpm build` in marketplace app - SUCCESS
- [ ] Run `pnpm test` in rbee-ui - ALL PASS
- [ ] Check SSG output (inspect .next/server/app/)
- [ ] Verify SEO metadata in HTML
- [ ] Test in browser (compatibility sections render)
- [ ] Review all changes for TEAM-410 signatures
- [ ] Create handoff document (max 2 pages)

**Handoff Document Contents:**
- Components created
- Pages updated
- SEO optimization
- Test coverage
- Next team ready: TEAM-411

---

## 📁 Files Created/Modified

### New Files
- `frontend/packages/rbee-ui/src/marketplace/atoms/CompatibilityBadge.tsx`
- `frontend/packages/rbee-ui/src/marketplace/organisms/WorkerCompatibilityList.tsx`
- `frontend/apps/marketplace/app/compatibility/page.tsx`
- `frontend/apps/marketplace/components/WorkerFilterDropdown.tsx`
- `frontend/apps/marketplace/components/CompatibilityMatrixTable.tsx`
- `frontend/packages/rbee-ui/src/marketplace/atoms/CompatibilityBadge.test.tsx`
- `frontend/packages/rbee-ui/src/marketplace/organisms/WorkerCompatibilityList.test.tsx`
- `TEAM_410_HANDOFF.md`

### Modified Files
- `frontend/apps/marketplace/app/models/[slug]/page.tsx` - Compatibility data
- `frontend/apps/marketplace/app/models/page.tsx` - Worker filter
- `frontend/packages/rbee-ui/src/marketplace/pages/ModelDetailPage.tsx` - Compatibility section
- `frontend/apps/marketplace/README.md` - Documentation
- `frontend/packages/rbee-ui/README.md` - Documentation

---

## ⚠️ Blockers & Dependencies

### Blocked By
- TEAM-409 (needs compatibility data layer)

### Blocks
- TEAM-411 (Tauri integration needs Next.js patterns)

---

## 🎯 Success Criteria

- [ ] Compatibility badges on model detail pages
- [ ] Worker filter on model list page
- [ ] Compatibility matrix page working
- [ ] SEO metadata includes compatibility
- [ ] All tests passing
- [ ] SSG optimized (no client-side fetching)
- [ ] Documentation complete
- [ ] Handoff document complete (≤2 pages)

---

## 📚 References

- Engineering Rules: `.windsurf/rules/engineering-rules.md`
- Next.js app: `frontend/apps/marketplace/`
- rbee-ui: `frontend/packages/rbee-ui/`
- marketplace-node: `frontend/packages/marketplace-node/`

---

**TEAM-410 - Phase 4 Checklist v1.0**  
**Next Phase:** TEAM-411 (Tauri Integration)
