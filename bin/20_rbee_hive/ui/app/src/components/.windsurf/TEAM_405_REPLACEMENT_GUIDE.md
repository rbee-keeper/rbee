# TEAM-405: Replacement Guide - MarketplaceSearch Component

**Date:** Nov 4, 2025  
**Status:** DESIGN SPEC  
**Purpose:** Guide for future team implementing MarketplaceSearch component

---

## 🎯 Mission

Create a **separate MarketplaceSearch component** that uses `@rbee/marketplace-sdk` to search external marketplaces (HuggingFace, CivitAI, Worker Catalog) and trigger downloads to populate the local catalog.

---

## 📐 Architecture

### Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│ LOCAL CATALOG MANAGEMENT (ModelManagement, WorkerManagement)│
├─────────────────────────────────────────────────────────────┤
│ • Lists artifacts from ~/.cache/rbee/                       │
│ • Operations: Load, Unload, Delete, Spawn                   │
│ • Data source: Backend (ModelList, WorkerListInstalled)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ MARKETPLACE SEARCH (MarketplaceSearch - NEW COMPONENT)      │
├─────────────────────────────────────────────────────────────┤
│ • Searches external marketplaces                            │
│ • Operations: Search, Browse, Download                      │
│ • Data source: marketplace-sdk (HuggingFaceClient, etc.)    │
└─────────────────────────────────────────────────────────────┘

                           ↓ Download
                           
┌─────────────────────────────────────────────────────────────┐
│ LOCAL CATALOG (Filesystem)                                  │
├─────────────────────────────────────────────────────────────┤
│ • ~/.cache/rbee/models/                                     │
│ • ~/.cache/rbee/workers/                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Component Structure

### File Layout

```
MarketplaceSearch/
├── index.tsx                    # Main component with tabs
├── types.ts                     # Marketplace-specific types
├── ModelSearch.tsx              # HuggingFace model search
├── WorkerSearch.tsx             # Worker catalog search
├── FilterPanel.tsx              # Reused from old ModelManagement
├── SearchResultCard.tsx         # Individual result card
└── README.md                    # Documentation
```

### Main Component (index.tsx)

```typescript
// TEAM-XXX: MarketplaceSearch - Browse and download artifacts

import { useState } from 'react'
import { Search, Package } from 'lucide-react'
import { Card, Tabs, TabsList, TabsTrigger, TabsContent } from '@rbee/ui/atoms'
import { ModelSearch } from './ModelSearch'
import { WorkerSearch } from './WorkerSearch'

type MarketplaceTab = 'models' | 'workers'

export function MarketplaceSearch() {
  const [activeTab, setActiveTab] = useState<MarketplaceTab>('models')

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Search className="h-5 w-5" />
          Marketplace Search
        </CardTitle>
        <CardDescription>
          Browse and download models and workers from external marketplaces
        </CardDescription>
      </CardHeader>

      <CardContent>
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as MarketplaceTab)}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="models">
              <Search className="h-4 w-4 mr-2" />
              Models (HuggingFace)
            </TabsTrigger>
            <TabsTrigger value="workers">
              <Package className="h-4 w-4 mr-2" />
              Workers (Catalog)
            </TabsTrigger>
          </TabsList>

          <TabsContent value="models">
            <ModelSearch />
          </TabsContent>

          <TabsContent value="workers">
            <WorkerSearch />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
```

---

## 🔌 Integration with marketplace-sdk

### ModelSearch Component

```typescript
// TEAM-XXX: Model search using marketplace-sdk

import { useState, useEffect } from 'react'
import { HuggingFaceClient } from '@rbee/marketplace-sdk'
import type { Model, ModelFilters } from '@rbee/marketplace-sdk'
import { useModelOperations } from '@rbee/rbee-hive-react'
import { FilterPanel } from './FilterPanel'
import { SearchResultCard } from './SearchResultCard'

export function ModelSearch() {
  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<ModelFilters>({
    search: '',
    category: 'text-generation',
    sort: 'Popular',
    limit: 50,
  })
  const [results, setResults] = useState<Model[]>([])
  const [loading, setLoading] = useState(false)

  const { downloadModel } = useModelOperations()

  // Search HuggingFace using marketplace-sdk
  useEffect(() => {
    if (!query || query.length < 2) {
      setResults([])
      return
    }

    const searchModels = async () => {
      setLoading(true)
      try {
        const client = new HuggingFaceClient()
        const models = await client.list_models({
          search: query,
          category: filters.category,
          sort: filters.sort,
          limit: filters.limit,
        })
        setResults(models)
      } catch (error) {
        console.error('HuggingFace search error:', error)
      } finally {
        setLoading(false)
      }
    }

    // Debounce search by 500ms
    const timeoutId = setTimeout(searchModels, 500)
    return () => clearTimeout(timeoutId)
  }, [query, filters])

  const handleDownload = (modelId: string) => {
    // Trigger download operation
    downloadModel({ modelId })
  }

  return (
    <div className="space-y-4">
      {/* Search Input */}
      <Input
        type="text"
        placeholder="Search HuggingFace models..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />

      <div className="grid grid-cols-12 gap-4">
        {/* Filters Sidebar */}
        <div className="col-span-3">
          <FilterPanel filters={filters} onFiltersChange={setFilters} />
        </div>

        {/* Search Results */}
        <div className="col-span-9 space-y-2">
          {loading && <Spinner />}
          {results.map((model) => (
            <SearchResultCard
              key={model.id}
              model={model}
              onDownload={handleDownload}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
```

### WorkerSearch Component

```typescript
// TEAM-XXX: Worker search using marketplace-sdk

import { useState, useEffect } from 'react'
import { WorkerClient } from '@rbee/marketplace-sdk'
import type { Worker } from '@rbee/marketplace-sdk'
import { useWorkerOperations } from '@rbee/rbee-hive-react'

export function WorkerSearch() {
  const [workers, setWorkers] = useState<Worker[]>([])
  const [loading, setLoading] = useState(false)

  const { installWorker } = useWorkerOperations()

  // Fetch workers from catalog
  useEffect(() => {
    const fetchWorkers = async () => {
      setLoading(true)
      try {
        const client = new WorkerClient('http://localhost:8787')
        const workers = await client.list_workers()
        setWorkers(workers)
      } catch (error) {
        console.error('Worker catalog error:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchWorkers()
  }, [])

  const handleInstall = (workerId: string) => {
    // Trigger install operation
    installWorker(workerId)
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {loading && <Spinner />}
      {workers.map((worker) => (
        <Card key={worker.id}>
          <CardHeader>
            <CardTitle>{worker.name}</CardTitle>
            <CardDescription>{worker.description}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div>Type: {worker.worker_type}</div>
              <div>Version: {worker.version}</div>
              <div>Platform: {worker.platform.join(', ')}</div>
            </div>
          </CardContent>
          <CardFooter>
            <Button onClick={() => handleInstall(worker.id)}>
              Install Worker
            </Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}
```

---

## 🔄 Data Flow

### Download Flow

```
User searches marketplace
    ↓
MarketplaceSearch component
    ↓
marketplace-sdk (HuggingFaceClient/WorkerClient)
    ↓
Search results displayed
    ↓
User clicks "Download"
    ↓
useModelOperations().downloadModel() or useWorkerOperations().installWorker()
    ↓
Backend operation (ModelDownload/WorkerInstall)
    ↓
Artifact downloaded to ~/.cache/rbee/
    ↓
Local catalog updated (model-catalog/worker-catalog)
    ↓
ModelManagement/WorkerManagement shows downloaded artifact
```

### Key Points

1. **MarketplaceSearch** searches external APIs (HuggingFace, Worker Catalog)
2. **Download operations** populate local catalog
3. **ModelManagement/WorkerManagement** list from local catalog
4. **Single source of truth:** marketplace-sdk for search, backend operations for catalog

---

## 📦 Reusable Components

### FilterPanel (from old ModelManagement)

Can be reused as-is:

```typescript
// TEAM-381: Filter panel for marketplace search
// TEAM-405: Moved from ModelManagement to MarketplaceSearch

export interface FilterState {
  formats: string[]
  architectures: string[]
  maxSize: string
  openSourceOnly: boolean
  sortBy: 'downloads' | 'likes' | 'recent'
}

export function FilterPanel({ filters, onFiltersChange }: FilterPanelProps) {
  // ... existing implementation from old ModelManagement/FilterPanel.tsx
}
```

### SearchResultCard (new)

```typescript
// TEAM-XXX: Search result card for marketplace items

interface SearchResultCardProps {
  model: Model
  onDownload: (modelId: string) => void
}

export function SearchResultCard({ model, onDownload }: SearchResultCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{model.name}</CardTitle>
        <CardDescription>{model.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex gap-2">
          <Badge>{model.size}</Badge>
          <Badge>{model.downloads.toLocaleString()} downloads</Badge>
          <Badge>{model.likes.toLocaleString()} likes</Badge>
        </div>
      </CardContent>
      <CardFooter>
        <Button onClick={() => onDownload(model.id)}>
          Download Model
        </Button>
      </CardFooter>
    </Card>
  )
}
```

---

## ✅ Acceptance Criteria

### Functionality

- [ ] Search HuggingFace models via marketplace-sdk
- [ ] Search Worker Catalog via marketplace-sdk
- [ ] Filter and sort search results
- [ ] Download models to local catalog
- [ ] Install workers to local catalog
- [ ] Show download/install progress
- [ ] Handle errors gracefully

### Integration

- [ ] Downloaded models appear in ModelManagement
- [ ] Installed workers appear in WorkerManagement
- [ ] Download operations use backend (ModelDownload)
- [ ] Install operations use backend (WorkerInstall)

### Code Quality

- [ ] Uses marketplace-sdk (no duplicate API clients)
- [ ] TypeScript types from marketplace-sdk
- [ ] Proper error handling
- [ ] Loading states
- [ ] Empty states
- [ ] TEAM-XXX signatures on all code

---

## 🚧 Dependencies

### Required

1. **marketplace-sdk (TEAM-402):**
   - ✅ Types defined (Model, Worker, ModelFilters)
   - 🚧 HuggingFaceClient implementation
   - 🚧 WorkerClient implementation

2. **Backend Operations:**
   - ✅ ModelDownload operation
   - ✅ WorkerInstall operation

3. **React Hooks:**
   - ✅ useModelOperations().downloadModel()
   - ✅ useWorkerOperations().installWorker()

### Optional Enhancements

- [ ] CivitAI search (marketplace-sdk CivitAIClient)
- [ ] Download progress tracking
- [ ] Batch downloads
- [ ] Search history
- [ ] Favorites/bookmarks

---

## 📊 Estimated Effort

**Total:** 12-16 hours

**Breakdown:**
- Component structure: 2 hours
- ModelSearch implementation: 4 hours
- WorkerSearch implementation: 3 hours
- FilterPanel integration: 1 hour
- SearchResultCard: 2 hours
- Testing: 2 hours
- Documentation: 2 hours

---

## 🔗 References

1. **marketplace-sdk:** `bin/99_shared_crates/marketplace-sdk/README.md`
2. **Old SearchResultsView:** `bin/20_rbee_hive/ui/app/src/components/ModelManagement/SearchResultsView.tsx` (deleted)
3. **Old WorkerCatalogView:** `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx` (deleted)
4. **TEAM-402 Memory:** Marketplace SDK implementation
5. **TEAM-405 Evidence:** `.windsurf/TEAM_405_EVIDENCE.md`

---

## 🎓 Key Principles

### Single Source of Truth

- ✅ marketplace-sdk for external API calls
- ✅ Backend operations for catalog management
- ❌ No duplicate API clients in components

### Separation of Concerns

- ✅ MarketplaceSearch = Browse external marketplaces
- ✅ ModelManagement/WorkerManagement = Manage local catalog
- ❌ Don't mix marketplace search with local management

### RULE ZERO Compliance

- ✅ Use marketplace-sdk (don't reimplement)
- ✅ Break APIs if needed (pre-1.0)
- ✅ Delete dead code immediately
- ✅ One way to do things

---

**This guide provides everything needed to implement MarketplaceSearch correctly.**
