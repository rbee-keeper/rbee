# Worker Filtering System - Implementation Plan

**Date:** 2025-11-07  
**Status:** PLANNING  
**Goal:** Create reusable filtering system for workers (similar to CivitAI models)

---

## 📊 Current State Analysis

### Worker Data Source
- **Location:** `/home/vince/Projects/llama-orch/bin/80-hono-worker-catalog/src/data.ts`
- **Count:** 6 workers (not 8 as mentioned - need to verify if 2 more exist)
  - 3 LLM workers: CPU, CUDA, Metal
  - 3 SD workers: CPU, CUDA, Metal
- **Missing:** ROCm variants mentioned in current workers page

### Worker Properties (from WorkerCatalogEntry)
```typescript
{
  id: string                    // "llm-worker-rbee-cpu"
  implementation: "rust" | "python" | "cpp"
  workerType: "cpu" | "cuda" | "metal"
  version: string               // "0.1.0"
  platforms: Platform[]         // ["linux", "macos", "windows"]
  architectures: Architecture[] // ["x86_64", "aarch64"]
  name: string                  // "LLM Worker (CPU)"
  description: string
  license: string
  supportedFormats: string[]    // ["gguf", "safetensors"]
  maxContextLength?: number     // 32768
  supportsStreaming: boolean
  supportsBatching: boolean
  // ... build/source info
}
```

### Current CivitAI Filter System
**Files:**
- `app/models/civitai/filters.ts` - Filter definitions & helpers
- `app/models/civitai/FilterBar.tsx` - UI component (3 filter groups)
- `app/models/civitai/[...filter]/page.tsx` - Dynamic route handler
- `app/models/civitai/page.tsx` - Default page

**Architecture:**
1. **Filter Config** - Defines available filters and pre-generated combinations
2. **SSG Pre-generation** - Popular filter combos pre-rendered at build time
3. **Link-based Navigation** - No client-side state, pure SSG
4. **Filter Bar Component** - Renders filter pills with active state

---

## 🎯 Worker Filtering Requirements

### Filter Dimensions

#### 1. **Worker Category** (Primary)
- **LLM Workers** - Language model inference
- **Image Workers** - Stable Diffusion inference
- **All Workers** - Default view

#### 2. **Backend Type** (workerType)
- **CPU** - No GPU required
- **CUDA** - NVIDIA GPUs
- **Metal** - Apple Silicon
- **All Backends** - Default

#### 3. **Platform** (Operating System)
- **Linux**
- **macOS**
- **Windows**
- **All Platforms** - Default

#### 4. **Architecture** (Optional - Advanced)
- **x86_64** - Intel/AMD
- **aarch64** - ARM64
- **All Architectures** - Default

---

## 📐 Reusable Component Architecture

### Phase 1: Extract Generic Filter System

#### 1.1 Create Generic Filter Types
**File:** `frontend/apps/marketplace/lib/filters/types.ts`

```typescript
export interface FilterOption<T = string> {
  label: string
  value: T
}

export interface FilterGroup<T = string> {
  id: string
  label: string
  options: FilterOption<T>[]
}

export interface FilterConfig<T = Record<string, string>> {
  filters: T
  path: string
}

export interface FilterSystemConfig<T = Record<string, string>> {
  groups: FilterGroup[]
  pregeneratedFilters: FilterConfig<T>[]
  buildUrl: (filters: Partial<T>) => string
  buildParams: (filters: T) => Record<string, any>
}
```

#### 1.2 Create Generic Filter Bar Component
**File:** `frontend/packages/rbee-ui/src/marketplace/organisms/FilterBar/FilterBar.tsx`

```typescript
export interface FilterBarProps<T = Record<string, string>> {
  groups: FilterGroup[]
  currentFilters: T
  buildUrl: (filters: Partial<T>) => string
  className?: string
}

export function FilterBar<T = Record<string, string>>({
  groups,
  currentFilters,
  buildUrl,
  className
}: FilterBarProps<T>) {
  return (
    <div className={cn("space-y-6 mb-8", className)}>
      {groups.map((group) => (
        <FilterGroup
          key={group.id}
          group={group}
          currentValue={currentFilters[group.id]}
          buildUrl={(value) => buildUrl({ ...currentFilters, [group.id]: value })}
        />
      ))}
    </div>
  )
}
```

#### 1.3 Create Filter Group Molecule
**File:** `frontend/packages/rbee-ui/src/molecules/FilterGroup/FilterGroup.tsx`

```typescript
export interface FilterGroupProps {
  group: FilterGroup
  currentValue: string
  buildUrl: (value: string) => string
}

export function FilterGroup({ group, currentValue, buildUrl }: FilterGroupProps) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
        {group.label}
      </h3>
      <div className="flex flex-wrap gap-2">
        {group.options.map((option) => {
          const isActive = currentValue === option.value
          const url = buildUrl(option.value)
          
          return (
            <Link
              key={option.value}
              href={url}
              className={cn(
                "px-4 py-2 rounded-full text-sm font-medium transition-all",
                isActive 
                  ? "bg-primary text-primary-foreground shadow-md" 
                  : "bg-muted hover:bg-muted/80 text-muted-foreground hover:text-foreground"
              )}
            >
              {option.label}
            </Link>
          )
        })}
      </div>
    </div>
  )
}
```

---

### Phase 2: Worker-Specific Implementation

#### 2.1 Worker Filter Definitions
**File:** `app/workers/filters.ts`

```typescript
import type { FilterSystemConfig } from '@/lib/filters/types'

export interface WorkerFilters {
  category: 'all' | 'llm' | 'image'
  backend: 'all' | 'cpu' | 'cuda' | 'metal'
  platform: 'all' | 'linux' | 'macos' | 'windows'
}

export const WORKER_FILTER_GROUPS = [
  {
    id: 'category',
    label: 'Worker Category',
    options: [
      { label: 'All Workers', value: 'all' },
      { label: 'LLM Workers', value: 'llm' },
      { label: 'Image Workers', value: 'image' },
    ],
  },
  {
    id: 'backend',
    label: 'Backend Type',
    options: [
      { label: 'All Backends', value: 'all' },
      { label: 'CPU', value: 'cpu' },
      { label: 'CUDA (NVIDIA)', value: 'cuda' },
      { label: 'Metal (Apple)', value: 'metal' },
    ],
  },
  {
    id: 'platform',
    label: 'Platform',
    options: [
      { label: 'All Platforms', value: 'all' },
      { label: 'Linux', value: 'linux' },
      { label: 'macOS', value: 'macos' },
      { label: 'Windows', value: 'windows' },
    ],
  },
]

// Pre-generated filter combinations (SSG)
export const PREGENERATED_WORKER_FILTERS: FilterConfig<WorkerFilters>[] = [
  // Default
  { filters: { category: 'all', backend: 'all', platform: 'all' }, path: '' },
  
  // By category
  { filters: { category: 'llm', backend: 'all', platform: 'all' }, path: 'filter/llm' },
  { filters: { category: 'image', backend: 'all', platform: 'all' }, path: 'filter/image' },
  
  // By backend
  { filters: { category: 'all', backend: 'cpu', platform: 'all' }, path: 'filter/cpu' },
  { filters: { category: 'all', backend: 'cuda', platform: 'all' }, path: 'filter/cuda' },
  { filters: { category: 'all', backend: 'metal', platform: 'all' }, path: 'filter/metal' },
  
  // By platform
  { filters: { category: 'all', backend: 'all', platform: 'linux' }, path: 'filter/linux' },
  { filters: { category: 'all', backend: 'all', platform: 'macos' }, path: 'filter/macos' },
  { filters: { category: 'all', backend: 'all', platform: 'windows' }, path: 'filter/windows' },
  
  // Popular combinations
  { filters: { category: 'llm', backend: 'cuda', platform: 'linux' }, path: 'filter/llm/cuda/linux' },
  { filters: { category: 'llm', backend: 'metal', platform: 'macos' }, path: 'filter/llm/metal/macos' },
  { filters: { category: 'image', backend: 'cuda', platform: 'linux' }, path: 'filter/image/cuda/linux' },
]

export function buildWorkerFilterUrl(filters: Partial<WorkerFilters>): string {
  const found = PREGENERATED_WORKER_FILTERS.find(
    f => 
      f.filters.category === (filters.category || 'all') &&
      f.filters.backend === (filters.backend || 'all') &&
      f.filters.platform === (filters.platform || 'all')
  )
  
  return found?.path ? `/workers/${found.path}` : '/workers'
}

export function getWorkerFilterFromPath(path: string): WorkerFilters {
  const found = PREGENERATED_WORKER_FILTERS.find(f => f.path === path)
  return found?.filters || { category: 'all', backend: 'all', platform: 'all' }
}

export function filterWorkers(
  workers: WorkerCatalogEntry[], 
  filters: WorkerFilters
): WorkerCatalogEntry[] {
  return workers.filter(worker => {
    // Category filter
    if (filters.category !== 'all') {
      const isLLM = worker.id.startsWith('llm-')
      const isImage = worker.id.startsWith('sd-')
      if (filters.category === 'llm' && !isLLM) return false
      if (filters.category === 'image' && !isImage) return false
    }
    
    // Backend filter
    if (filters.backend !== 'all' && worker.workerType !== filters.backend) {
      return false
    }
    
    // Platform filter
    if (filters.platform !== 'all' && !worker.platforms.includes(filters.platform)) {
      return false
    }
    
    return true
  })
}
```

#### 2.2 Worker Page with Filtering
**File:** `app/workers/page.tsx`

```typescript
import { WORKERS } from '@rbee/worker-catalog'
import { WorkerCard } from '@rbee/ui/marketplace'
import { FilterBar } from '@rbee/ui/marketplace/organisms/FilterBar'
import { WORKER_FILTER_GROUPS, PREGENERATED_WORKER_FILTERS, filterWorkers } from './filters'

export default async function WorkersPage() {
  const currentFilters = PREGENERATED_WORKER_FILTERS[0].filters
  const filteredWorkers = filterWorkers(WORKERS, currentFilters)
  
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          Workers
        </h1>
        <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
          {filteredWorkers.length} workers available
        </p>
      </div>

      <FilterBar
        groups={WORKER_FILTER_GROUPS}
        currentFilters={currentFilters}
        buildUrl={buildWorkerFilterUrl}
      />

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredWorkers.map((worker) => (
          <WorkerCard key={worker.id} worker={worker} />
        ))}
      </div>
    </div>
  )
}
```

#### 2.3 Dynamic Filter Route
**File:** `app/workers/[...filter]/page.tsx`

```typescript
export async function generateStaticParams() {
  return PREGENERATED_WORKER_FILTERS
    .filter(f => f.path !== '')
    .map(f => ({ filter: f.path.split('/') }))
}

export default async function FilteredWorkersPage({ params }: PageProps) {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilters = getWorkerFilterFromPath(filterPath)
  const filteredWorkers = filterWorkers(WORKERS, currentFilters)
  
  // ... render with FilterBar and WorkerCard
}
```

---

## 🔄 Migration Path for CivitAI

### Refactor CivitAI to Use Generic Components

**Before:**
```typescript
// app/models/civitai/FilterBar.tsx
// Custom implementation with hardcoded structure
```

**After:**
```typescript
import { FilterBar } from '@rbee/ui/marketplace/organisms/FilterBar'
import { CIVITAI_FILTER_GROUPS, buildCivitaiFilterUrl } from './filters'

export function CivitaiFilterBar({ currentFilter }: Props) {
  return (
    <FilterBar
      groups={CIVITAI_FILTER_GROUPS}
      currentFilters={currentFilter}
      buildUrl={buildCivitaiFilterUrl}
    />
  )
}
```

---

## 📋 Implementation Checklist

### Phase 1: Generic Components (2-3 hours)
- [ ] Create `lib/filters/types.ts` - Generic filter types
- [ ] Create `rbee-ui/molecules/FilterGroup/FilterGroup.tsx` - Filter group molecule
- [ ] Create `rbee-ui/marketplace/organisms/FilterBar/FilterBar.tsx` - Generic filter bar
- [ ] Add to rbee-ui exports

### Phase 2: Worker Implementation (3-4 hours)
- [ ] Verify all 8 workers exist in data.ts (currently only 6)
- [ ] Create `app/workers/filters.ts` - Worker filter definitions
- [ ] Update `app/workers/page.tsx` - Use FilterBar component
- [ ] Create `app/workers/[...filter]/page.tsx` - Dynamic route
- [ ] Update WorkerCard to handle all worker types

### Phase 3: CivitAI Refactor (1-2 hours)
- [ ] Refactor `app/models/civitai/filters.ts` to match new pattern
- [ ] Update `app/models/civitai/FilterBar.tsx` to use generic FilterBar
- [ ] Test all filter combinations work

### Phase 4: Testing & Polish (1-2 hours)
- [ ] Test SSG generation for all filter paths
- [ ] Verify SEO metadata for filtered pages
- [ ] Test mobile responsiveness
- [ ] Add loading states if needed

---

## 🎨 Component Hierarchy

```
FilterBar (organism)
├── FilterGroup (molecule) × N
    ├── Link (Next.js) × N
    └── Active state styling
```

---

## 🚀 Benefits

1. **Reusability** - Same filtering system for workers, models, any catalog
2. **Type Safety** - Generic types ensure consistency
3. **SSG Compatible** - Pre-generated routes for SEO
4. **Maintainability** - Single source of truth for filter UI
5. **Consistency** - Same UX across all marketplace pages

---

## ⚠️ Open Questions

1. **Worker Count** - User mentioned 8 workers, but data.ts only has 6. Missing ROCm variants?
2. **Architecture Filter** - Should this be exposed in UI or kept as advanced filter?
3. **Implementation Filter** - Rust/Python/C++ - useful for users?
4. **Capabilities Filter** - Streaming, batching, formats - too technical?

---

## 📊 Estimated Effort

- **Phase 1 (Generic):** 2-3 hours
- **Phase 2 (Workers):** 3-4 hours  
- **Phase 3 (Refactor):** 1-2 hours
- **Phase 4 (Testing):** 1-2 hours

**Total:** 7-11 hours (1-1.5 days)

---

## 🎯 Success Criteria

- ✅ Workers page shows all 8 workers (verify count)
- ✅ Filter by category (LLM/Image/All)
- ✅ Filter by backend (CPU/CUDA/Metal/All)
- ✅ Filter by platform (Linux/macOS/Windows/All)
- ✅ All filter combinations pre-generated (SSG)
- ✅ CivitAI uses same FilterBar component
- ✅ Consistent styling across both pages
- ✅ Mobile responsive
- ✅ SEO metadata for all filtered pages
