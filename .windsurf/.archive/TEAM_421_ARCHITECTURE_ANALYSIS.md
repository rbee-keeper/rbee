# TEAM-421: Architecture Analysis - Data vs Presentation Layer

**Date:** 2025-11-06  
**Status:** Analysis Complete

---

## Current State Analysis

### ✅ GOOD: Presentation Layer Separation

#### rbee-ui Package (Shared Presentation)
```
frontend/packages/rbee-ui/src/marketplace/
├── pages/
│   ├── ModelsPage/          ✅ DUMB - just renders template
│   ├── ModelDetailPage/     ✅ DUMB - just renders template  
│   └── WorkersPage/         ✅ DUMB - just renders template
├── templates/
│   ├── ModelDetailPageTemplate/  ✅ Pure presentation
│   ├── WorkerListTemplate/       ✅ Pure presentation
│   └── ArtifactDetailPageTemplate/ ✅ Pure presentation
```

**Status:** ✅ **PERFECT** - No data fetching, pure presentation

---

### ❌ PROBLEM: Missing WorkerDetailPage in rbee-ui

**What exists:**
- ✅ `WorkersPage` (list) - DUMB wrapper
- ✅ `WorkerListTemplate` - Pure presentation
- ❌ **NO `WorkerDetailPage`** - Missing!
- ❌ **NO `WorkerDetailPageTemplate`** - Missing!

**What's being used instead:**
- Tauri: `WorkerDetailsPage` uses `ArtifactDetailPageTemplate` directly ✅
- Next.js: Custom inline presentation ❌ (DUPLICATION!)

---

### 🔴 CRITICAL: Next.js Worker Page Has Inline Presentation

**File:** `/home/vince/Projects/llama-orch/frontend/apps/marketplace/app/workers/[workerId]/page.tsx`

**Problem:** 228 lines of **INLINE JSX** - not using shared templates!

```tsx
// ❌ BAD: Presentation logic in Next.js page
export default async function WorkerDetailPage({ params }: PageProps) {
  const worker = WORKERS[workerId];

  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl">
      {/* 100+ lines of inline JSX */}
      <h1>{worker.name}</h1>
      <div className="grid gap-8 md:grid-cols-2">
        {/* More inline presentation */}
      </div>
    </div>
  );
}
```

**Should be:**
```tsx
// ✅ GOOD: Use shared template
export default async function WorkerDetailPage({ params }: PageProps) {
  const worker = WORKERS[workerId];

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <WorkerDetailWithInstall worker={worker} />
    </div>
  );
}
```

---

### ✅ GOOD: Next.js Model Page Architecture

**File:** `/home/vince/Projects/llama-orch/frontend/apps/marketplace/app/models/[slug]/page.tsx`

```tsx
// ✅ Data layer (SSG)
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const model = await getHuggingFaceModel(modelId);
  return {
    title: `${model.name} | AI Model`,
    description: model.description,
  };
}

// ✅ Data fetching
export default async function ModelPage({ params }: Props) {
  const model = await getHuggingFaceModel(modelId);
  
  // ✅ Delegates to client component
  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <ModelDetailWithInstall model={model} />
    </div>
  );
}
```

**File:** `/home/vince/Projects/llama-orch/frontend/apps/marketplace/components/ModelDetailWithInstall.tsx`

```tsx
'use client'

// ✅ Client wrapper with conversion CTA
export function ModelDetailWithInstall({ model }: Props) {
  return (
    <div className="space-y-6">
      {/* Conversion CTA */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h3>One-Click Installation</h3>
        <InstallButton modelId={model.id} />
      </div>

      {/* ✅ Uses shared template */}
      <ModelDetailPageTemplate
        model={model}
        showBackButton={false}
      />
    </div>
  );
}
```

**Status:** ✅ **PERFECT** - Clean separation, uses shared template

---

### ✅ GOOD: Tauri Pages Architecture

#### ModelDetailsPage (Tauri)
**File:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/pages/ModelDetailsPage.tsx`

```tsx
export function ModelDetailsPage() {
  // ✅ Data layer (React Query + Tauri commands)
  const { data: rawModel } = useQuery({
    queryFn: async () => {
      return await invoke<Model>("marketplace_get_model", { modelId });
    },
  });

  // ✅ Control layer (environment-aware actions)
  const actions = useArtifactActions();

  // ✅ Uses shared template
  return (
    <PageContainer>
      <ModelDetailPageTemplate
        model={model}
        onDownload={() => actions.downloadModel(model.id)}
      />
    </PageContainer>
  );
}
```

**Status:** ✅ **PERFECT** - Data layer separate, uses shared template

#### WorkerDetailsPage (Tauri)
**File:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/ui/src/pages/WorkerDetailsPage.tsx`

```tsx
export function WorkerDetailsPage() {
  // ✅ Data layer (React Query + Tauri commands)
  const { data: worker } = useQuery({
    queryFn: async () => {
      const workers = await invoke<WorkerCatalogEntry[]>("marketplace_list_workers");
      return workers.find((w) => w.id === workerId);
    },
  });

  // ✅ Control layer (environment-aware actions)
  const actions = useArtifactActions();

  // ✅ Uses shared template (ArtifactDetailPageTemplate)
  return (
    <PageContainer>
      <ArtifactDetailPageTemplate
        name={worker.name}
        primaryAction={{
          label: actions.getButtonLabel('install'),
          onClick: () => actions.installWorker(worker.id),
        }}
        mainContent={mainContent}
      />
    </PageContainer>
  );
}
```

**Status:** ✅ **PERFECT** - Data layer separate, uses shared template

---

## Architecture Summary

### ✅ What's Working

1. **rbee-ui templates** - Pure presentation, no data fetching
2. **Tauri pages** - Clean data/presentation separation
3. **Next.js model pages** - Clean data/presentation separation
4. **Environment-aware actions** - Works correctly

### ❌ What's Broken

1. **Next.js worker page** - 228 lines of inline JSX (DUPLICATION!)
2. **Missing WorkerDetailPage** - No shared template for worker details
3. **No conversion CTAs** - Next.js pages missing "Install rbee" prompts

---

## Required Actions

### Priority 1: Create WorkerDetailPage Template (HIGH)

**Create:**
1. `frontend/packages/rbee-ui/src/marketplace/pages/WorkerDetailPage/WorkerDetailPage.tsx`
2. `frontend/packages/rbee-ui/src/marketplace/pages/WorkerDetailPage/WorkerDetailPageProps.tsx`
3. `frontend/packages/rbee-ui/src/marketplace/pages/WorkerDetailPage/index.ts`

**OR** (simpler):
1. Just use `ArtifactDetailPageTemplate` directly (already works in Tauri!)

### Priority 2: Refactor Next.js Worker Page (HIGH)

**Create:**
`frontend/apps/marketplace/components/WorkerDetailWithInstall.tsx`

```tsx
'use client'

import { ArtifactDetailPageTemplate, useArtifactActions } from '@rbee/ui/marketplace';

export function WorkerDetailWithInstall({ worker }: Props) {
  const actions = useArtifactActions();

  return (
    <div className="space-y-6">
      {/* Conversion CTA */}
      <InstallCTA artifactType="worker" />

      {/* Shared template */}
      <ArtifactDetailPageTemplate
        name={worker.name}
        description={worker.description}
        artifactType="worker"
        primaryAction={{
          label: actions.getButtonLabel('install'),
          onClick: () => actions.installWorker(worker.id),
        }}
        mainContent={<WorkerDetailsCards worker={worker} />}
      />
    </div>
  );
}
```

**Update:**
`frontend/apps/marketplace/app/workers/[workerId]/page.tsx`

```tsx
export default async function WorkerDetailPage({ params }: PageProps) {
  const worker = WORKERS[workerId];

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <WorkerDetailWithInstall worker={worker} />
    </div>
  );
}
```

### Priority 3: Add Conversion CTAs (MEDIUM)

**Create:**
`frontend/packages/rbee-ui/src/marketplace/components/InstallCTA.tsx`

```tsx
import { getEnvironment } from '@rbee/ui/utils';

export function InstallCTA({ artifactType }: { artifactType: 'model' | 'worker' }) {
  const env = getEnvironment();

  // Only show in Next.js (not in Tauri)
  if (env === 'tauri') return null;

  return (
    <Card className="bg-gradient-to-r from-blue-50 to-purple-50">
      <div className="p-6 text-center">
        <h3>Install rbee to {artifactType === 'model' ? 'download' : 'install'}</h3>
        <Button href="/download">Download rbee</Button>
      </div>
    </Card>
  );
}
```

**Use in:**
- `ModelDetailWithInstall.tsx` (replace existing CTA)
- `WorkerDetailWithInstall.tsx` (new)

---

## Decision: WorkerDetailPage vs ArtifactDetailPageTemplate

### Option A: Create WorkerDetailPage Template
**Pros:**
- Consistent with ModelDetailPage
- Specific to workers

**Cons:**
- More code
- ArtifactDetailPageTemplate already works

### Option B: Use ArtifactDetailPageTemplate Directly ✅ RECOMMENDED
**Pros:**
- Already works in Tauri
- Less code
- More flexible

**Cons:**
- None

**Decision:** Use `ArtifactDetailPageTemplate` directly (like Tauri does)

---

## Final Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Next.js (marketplace.rbee.ai)                               │
├─────────────────────────────────────────────────────────────┤
│ app/models/[slug]/page.tsx                                  │
│   ├─ generateMetadata() → SEO (title, description, OG)     │
│   ├─ getHuggingFaceModel() → Data fetching                 │
│   └─ <ModelDetailWithInstall /> → Client wrapper           │
│                                                             │
│ components/ModelDetailWithInstall.tsx ('use client')        │
│   ├─ <InstallCTA /> → Conversion prompt                    │
│   └─ <ModelDetailPageTemplate /> → Shared presentation     │
├─────────────────────────────────────────────────────────────┤
│ app/workers/[workerId]/page.tsx                             │
│   ├─ generateMetadata() → SEO                              │
│   ├─ WORKERS[id] → Static data                             │
│   └─ <WorkerDetailWithInstall /> → Client wrapper          │
│                                                             │
│ components/WorkerDetailWithInstall.tsx ('use client')       │
│   ├─ <InstallCTA /> → Conversion prompt                    │
│   └─ <ArtifactDetailPageTemplate /> → Shared presentation  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Tauri (rbee-keeper)                                         │
├─────────────────────────────────────────────────────────────┤
│ pages/ModelDetailsPage.tsx                                  │
│   ├─ useQuery() + invoke() → Data fetching                 │
│   ├─ useArtifactActions() → Control layer                  │
│   └─ <ModelDetailPageTemplate /> → Shared presentation     │
├─────────────────────────────────────────────────────────────┤
│ pages/WorkerDetailsPage.tsx                                 │
│   ├─ useQuery() + invoke() → Data fetching                 │
│   ├─ useArtifactActions() → Control layer                  │
│   └─ <ArtifactDetailPageTemplate /> → Shared presentation  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ rbee-ui (Shared Presentation Layer)                         │
├─────────────────────────────────────────────────────────────┤
│ templates/                                                  │
│   ├─ ModelDetailPageTemplate → Pure presentation           │
│   ├─ ArtifactDetailPageTemplate → Pure presentation        │
│   └─ WorkerListTemplate → Pure presentation                │
│                                                             │
│ components/                                                 │
│   └─ InstallCTA → Environment-aware conversion prompt      │
│                                                             │
│ hooks/                                                      │
│   └─ useArtifactActions → Environment-aware actions        │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Phase 1: Fix Next.js Worker Page
- [ ] Create `components/WorkerDetailWithInstall.tsx`
- [ ] Create `components/InstallCTA.tsx`
- [ ] Refactor `app/workers/[workerId]/page.tsx` to use wrapper
- [ ] Remove 200+ lines of inline JSX
- [ ] Test Next.js build

### Phase 2: Add Conversion CTAs
- [ ] Update `ModelDetailWithInstall.tsx` to use `InstallCTA`
- [ ] Update `WorkerDetailWithInstall.tsx` to use `InstallCTA`
- [ ] Verify CTAs only show in Next.js (not Tauri)

### Phase 3: Enhance SEO
- [ ] Improve `generateMetadata()` in model page
- [ ] Improve `generateMetadata()` in worker page
- [ ] Add Open Graph images
- [ ] Add JSON-LD structured data

---

## Success Criteria

✅ **No presentation duplication** - All pages use shared templates  
✅ **Clean data/presentation separation** - Data fetching separate from UI  
✅ **Environment-aware CTAs** - Conversion prompts only in Next.js  
✅ **SEO in Next.js only** - Metadata handled in page wrappers  
✅ **Reusable components** - Same templates work in Tauri and Next.js  

---

## Next Steps

**START HERE:**
1. Create `InstallCTA.tsx` component
2. Create `WorkerDetailWithInstall.tsx` wrapper
3. Refactor Next.js worker page to use wrapper
4. Test in both environments

**Ready to implement?** 🚀
