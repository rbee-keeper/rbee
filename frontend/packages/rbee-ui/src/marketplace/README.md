# Marketplace Components

**TEAM-401:** Marketplace-specific components for rbee.

## Structure

```
marketplace/
├── organisms/   (Reusable cards and controls)
│   ├── ModelCard/
│   ├── WorkerCard/
│   ├── MarketplaceGrid/
│   └── FilterBar/
├── templates/   (Page sections)
│   ├── ModelListTemplate/
│   ├── ModelDetailTemplate/
│   └── WorkerListTemplate/
└── pages/       (Complete pages)
    ├── ModelsPage/
    ├── ModelDetailPage/
    └── WorkersPage/
```

## Usage

### In Next.js (marketplace site)

```tsx
import { ModelsPage, defaultModelsPageProps } from '@rbee/ui/marketplace/pages/ModelsPage'

export default function Page() {
  return <ModelsPage {...defaultModelsPageProps} />
}
```

### In Tauri (Keeper app)

```tsx
import { ModelCard } from '@rbee/ui/marketplace/organisms/ModelCard'

export function MarketplaceTab() {
  const models = useModels() // From marketplace SDK
  return (
    <div className="grid grid-cols-3 gap-6">
      {models.map(model => (
        <ModelCard key={model.id} model={model} onAction={handleDownload} />
      ))}
    </div>
  )
}
```

## Components

### Organisms

#### ModelCard
Displays a single model with image, tags, stats, and action button.

```tsx
<ModelCard
  model={{
    id: 'llama-3.2-1b',
    name: 'Llama 3.2 1B',
    description: 'Fast and efficient small language model',
    author: 'Meta',
    imageUrl: '/models/llama-3.2-1b.jpg',
    tags: ['llm', 'chat', 'small'],
    downloads: 125000,
    likes: 3400,
    size: '1.2 GB'
  }}
  onAction={(id) => console.log('Download', id)}
/>
```

#### WorkerCard
Displays a worker binary with platform/architecture badges.

```tsx
<WorkerCard
  worker={{
    id: 'llama-cpp-cuda',
    name: 'llama.cpp CUDA',
    description: 'CUDA-accelerated inference worker',
    version: '1.0.0',
    platform: ['Linux', 'Windows'],
    architecture: ['x86_64'],
    workerType: 'cuda'
  }}
  onAction={(id) => console.log('Install', id)}
/>
```

#### MarketplaceGrid
Generic grid with loading/error/empty states.

```tsx
<MarketplaceGrid
  items={models}
  renderItem={(model) => <ModelCard model={model} />}
  isLoading={loading}
  error={error}
  columns={3}
/>
```

#### FilterBar
Search and sort controls with debounced input and optional filter chips.

```tsx
<FilterBar
  search={search}
  onSearchChange={setSearch}
  sort={sort}
  onSortChange={setSort}
  sortOptions={[
    { value: 'popular', label: 'Most Popular' },
    { value: 'recent', label: 'Recently Added' }
  ]}
  onClearFilters={handleClear}
  filterChips={[
    { id: 'llm', label: 'LLM', active: true },
    { id: 'vision', label: 'Vision', active: false },
    { id: 'audio', label: 'Audio', active: false }
  ]}
  onFilterChipToggle={(id) => toggleFilter(id)}
/>
```

### Templates

#### ModelListTemplate
Complete model browsing interface with filters and grid.

```tsx
<ModelListTemplate
  title="AI Models"
  description="Browse and download models"
  models={models}
  filters={{ search: '', sort: 'popular' }}
  onFilterChange={handleFilterChange}
  onModelAction={handleDownload}
/>
```

#### ModelDetailTemplate
Detailed model view with specs and related models.

```tsx
<ModelDetailTemplate
  model={modelData}
  installButton={<CustomButton />}
  relatedModels={relatedModels}
/>
```

#### WorkerListTemplate
Worker browsing interface (similar to ModelListTemplate).

```tsx
<WorkerListTemplate
  title="Inference Workers"
  workers={workers}
  filters={{ search: '', sort: 'name' }}
  onFilterChange={handleFilterChange}
  onWorkerAction={handleInstall}
/>
```

### Pages

#### ModelsPage
Complete models page (DUMB - just renders template).

```tsx
<ModelsPage
  template={{
    title: 'AI Models',
    models: modelsData,
    filters: { search: '', sort: 'popular' }
  }}
/>
```

#### ModelDetailPage
Complete model detail page.

```tsx
<ModelDetailPage
  template={{
    model: modelData,
    relatedModels: relatedData
  }}
/>
```

#### WorkersPage
Complete workers page.

```tsx
<WorkersPage
  template={{
    title: 'Inference Workers',
    workers: workersData,
    filters: { search: '', sort: 'name' }
  }}
/>
```

## Principles

1. **DUMB COMPONENTS** - No data fetching, only props
2. **REUSE ATOMS/MOLECULES** - Don't recreate Button, Card, Badge, etc.
3. **CONSISTENT STYLING** - Follow rbee-ui patterns (Card structure, spacing)
4. **SSG-READY** - All data in Props files (perfect for Next.js SSG)
5. **TYPED** - Full TypeScript support with exported types
6. **FLEXIBLE** - Works in both Next.js (marketplace site) and Tauri (Keeper app)

## Styling

All components use:
- Tailwind CSS classes
- Design tokens from `@rbee/ui/tokens`
- Consistent spacing (p-6, gap-6, etc.)
- Responsive layouts (mobile-first)
- Dark mode support (via Tailwind dark: prefix)

## Testing

Components are tested with:
- Unit tests (Vitest)
- Component tests (Playwright)
- Storybook stories for visual testing

Run tests:
```bash
pnpm test                    # Unit tests
pnpm test:ct                 # Component tests
pnpm storybook              # Visual testing
```

## Common Patterns

### Pagination

Use the existing `Pagination` atom with `MarketplaceGrid`:

```tsx
import { 
  Pagination, 
  PaginationContent, 
  PaginationItem, 
  PaginationLink,
  PaginationPrevious,
  PaginationNext,
  PaginationEllipsis
} from '@rbee/ui/atoms/Pagination'
import { MarketplaceGrid } from '@rbee/ui/marketplace/organisms/MarketplaceGrid'

function ModelsWithPagination() {
  const [page, setPage] = useState(1)
  const totalPages = 10
  
  return (
    <MarketplaceGrid
      items={models}
      renderItem={(model) => <ModelCard model={model} />}
      pagination={
        <Pagination>
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious 
                href="#" 
                onClick={() => setPage(p => Math.max(1, p - 1))}
              />
            </PaginationItem>
            
            {[1, 2, 3].map((p) => (
              <PaginationItem key={p}>
                <PaginationLink 
                  href="#" 
                  isActive={page === p}
                  onClick={() => setPage(p)}
                >
                  {p}
                </PaginationLink>
              </PaginationItem>
            ))}
            
            <PaginationItem>
              <PaginationEllipsis />
            </PaginationItem>
            
            <PaginationItem>
              <PaginationNext 
                href="#" 
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      }
    />
  )
}
```

### Filter Chips

Use filter chips for category/tag filtering:

```tsx
import { FilterBar, type FilterChip } from '@rbee/ui/marketplace/organisms/FilterBar'

function ModelsWithFilters() {
  const [filters, setFilters] = useState<FilterChip[]>([
    { id: 'llm', label: 'LLM', active: false },
    { id: 'vision', label: 'Vision', active: false },
    { id: 'audio', label: 'Audio', active: false },
    { id: 'embedding', label: 'Embedding', active: false }
  ])
  
  const toggleFilter = (id: string) => {
    setFilters(prev => prev.map(f => 
      f.id === id ? { ...f, active: !f.active } : f
    ))
  }
  
  return (
    <FilterBar
      search={search}
      onSearchChange={setSearch}
      sort={sort}
      onSortChange={setSort}
      sortOptions={sortOptions}
      onClearFilters={() => {
        setSearch('')
        setSort('popular')
        setFilters(prev => prev.map(f => ({ ...f, active: false })))
      }}
      filterChips={filters}
      onFilterChipToggle={toggleFilter}
    />
  )
}
```

### SSG with Next.js
```tsx
// app/models/page.tsx
import { ModelsPage } from '@rbee/ui/marketplace/pages/ModelsPage'
import { getModels } from '@/lib/api'

export default async function Page() {
  const models = await getModels()
  
  return (
    <ModelsPage
      template={{
        title: 'AI Models',
        models,
        filters: { search: '', sort: 'popular' }
      }}
    />
  )
}
```

### Dynamic with Tauri
```tsx
// src/pages/MarketplacePage.tsx
import { ModelCard } from '@rbee/ui/marketplace/organisms/ModelCard'
import { useMarketplaceSDK } from '@/hooks/useMarketplaceSDK'

export function MarketplacePage() {
  const { models, download } = useMarketplaceSDK()
  
  return (
    <div className="grid grid-cols-3 gap-6">
      {models.map(model => (
        <ModelCard
          key={model.id}
          model={model}
          onAction={download}
        />
      ))}
    </div>
  )
}
```

## Architecture

- **Organisms** = Reusable marketplace-specific components
- **Templates** = Page sections combining organisms
- **Pages** = Complete pages combining templates (DUMB, just props)

This follows the atomic design pattern used throughout rbee-ui.
