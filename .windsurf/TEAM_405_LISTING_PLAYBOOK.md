# TEAM-405: Listing Features Playbook

**Date:** Nov 4, 2025  
**Status:** 📚 REFERENCE  
**Mission:** Complete guide for implementing filters, sorting, and pagination in marketplace

---

## 🎯 Available Components

### 1. **FilterBar** (Organism)
**Location:** `@rbee/ui/marketplace/organisms/FilterBar`

**Features:**
- Search input with debounce (300ms)
- Sort dropdown
- Filter chips (toggleable)
- Clear filters button
- Responsive layout

**Props:**
```tsx
interface FilterBarProps {
  search: string
  onSearchChange: (value: string) => void
  sort: string
  onSortChange: (value: string) => void
  sortOptions: Array<{ value: string; label: string }>
  onClearFilters: () => void
  filterChips?: FilterChip[]
  onFilterChipToggle?: (chipId: string) => void
}

interface FilterChip {
  id: string
  label: string
  active: boolean
}
```

**Usage:**
```tsx
<FilterBar
  search={search}
  onSearchChange={setSearch}
  sort={sort}
  onSortChange={setSort}
  sortOptions={[
    { value: 'popular', label: 'Most Popular' },
    { value: 'recent', label: 'Recently Added' },
    { value: 'downloads', label: 'Most Downloads' },
  ]}
  onClearFilters={() => {
    setSearch('')
    setSort('popular')
  }}
  filterChips={[
    { id: 'text-gen', label: 'Text Generation', active: true },
    { id: 'vision', label: 'Vision', active: false },
  ]}
  onFilterChipToggle={(id) => toggleFilter(id)}
/>
```

---

### 2. **Pagination** (Atom)
**Location:** `@rbee/ui/atoms/Pagination`

**Components:**
- `Pagination` - Container
- `PaginationContent` - List wrapper
- `PaginationItem` - List item
- `PaginationLink` - Page link
- `PaginationPrevious` - Previous button
- `PaginationNext` - Next button
- `PaginationEllipsis` - "..." indicator

**Usage:**
```tsx
<Pagination>
  <PaginationContent>
    <PaginationItem>
      <PaginationPrevious href="#" />
    </PaginationItem>
    <PaginationItem>
      <PaginationLink href="#" isActive>1</PaginationLink>
    </PaginationItem>
    <PaginationItem>
      <PaginationLink href="#">2</PaginationLink>
    </PaginationItem>
    <PaginationItem>
      <PaginationEllipsis />
    </PaginationItem>
    <PaginationItem>
      <PaginationLink href="#">10</PaginationLink>
    </PaginationItem>
    <PaginationItem>
      <PaginationNext href="#" />
    </PaginationItem>
  </PaginationContent>
</Pagination>
```

---

### 3. **MarketplaceGrid** (Organism)
**Location:** `@rbee/ui/marketplace/organisms/MarketplaceGrid`

**Features:**
- Responsive grid (1/2/3 columns)
- Loading skeletons
- Empty states
- Error handling

**Props:**
```tsx
interface MarketplaceGridProps<T> {
  items: T[]
  renderItem: (item: T) => React.ReactNode
  isLoading?: boolean
  error?: string
  emptyMessage?: string
  emptyDescription?: string
  columns?: 1 | 2 | 3
}
```

---

### 4. **ModelListTemplate** (Template)
**Location:** `@rbee/ui/marketplace/templates/ModelListTemplate`

**Complete template with:**
- Header (title + description)
- FilterBar integration
- MarketplaceGrid integration
- All wired up

**Props:**
```tsx
interface ModelListTemplateProps {
  title: string
  description?: string
  models: Model[]
  filters?: { search: string; sort: string }
  sortOptions?: Array<{ value: string; label: string }>
  onFilterChange?: (filters: { search: string; sort: string }) => void
  onModelAction?: (modelId: string) => void
  isLoading?: boolean
  error?: string
  emptyMessage?: string
  emptyDescription?: string
}
```

---

## 🎨 Implementation Patterns

### Pattern 1: Simple Filtering (Client-Side)

**Use when:** Small dataset (<100 items), no backend filtering

```tsx
const [search, setSearch] = useState('')
const [sort, setSort] = useState('popular')

// Filter and sort
const filteredModels = useMemo(() => {
  let result = models.filter(m => 
    m.name.toLowerCase().includes(search.toLowerCase())
  )
  
  // Sort
  switch (sort) {
    case 'popular':
      return result.sort((a, b) => b.downloads - a.downloads)
    case 'recent':
      return result.sort((a, b) => b.createdAt - a.createdAt)
    default:
      return result
  }
}, [models, search, sort])
```

---

### Pattern 2: Server-Side Filtering

**Use when:** Large dataset, backend API supports filtering

```tsx
const [filters, setFilters] = useState({ search: '', sort: 'popular' })

const { data: models, isLoading } = useQuery({
  queryKey: ['models', filters],
  queryFn: async () => {
    return await invoke('marketplace_list_models', {
      query: filters.search || null,
      sort: filters.sort,
      limit: 50,
    })
  },
})
```

---

### Pattern 3: Pagination (Client-Side)

**Use when:** Want to show items in pages, all data loaded

```tsx
const [page, setPage] = useState(1)
const itemsPerPage = 20

const paginatedItems = useMemo(() => {
  const start = (page - 1) * itemsPerPage
  return items.slice(start, start + itemsPerPage)
}, [items, page])

const totalPages = Math.ceil(items.length / itemsPerPage)
```

---

### Pattern 4: Pagination (Server-Side)

**Use when:** Large dataset, backend supports pagination

```tsx
const [page, setPage] = useState(1)
const limit = 20

const { data, isLoading } = useQuery({
  queryKey: ['models', page],
  queryFn: async () => {
    return await invoke('marketplace_list_models', {
      offset: (page - 1) * limit,
      limit,
    })
  },
})
```

---

## 🔧 Recommended Implementation for Marketplace

### Step 1: Add Sorting to Backend

**Update Tauri command:**
```rust
#[tauri::command]
pub async fn marketplace_list_models(
    query: Option<String>,
    sort: Option<String>,  // NEW
    limit: Option<u32>,
) -> Result<Vec<Model>, String> {
    // ... existing code ...
}
```

**Update HuggingFace client:**
```rust
pub async fn list_models(
    &self,
    query: Option<String>,
    sort: Option<String>,  // NEW
    limit: Option<u32>,
) -> Result<Vec<Model>> {
    let mut url = format!("{}/models?limit={}", HF_API_BASE, limit.unwrap_or(50));
    
    if let Some(q) = query {
        url.push_str(&format!("&search={}", urlencoding::encode(&q)));
    }
    
    // NEW: Add sort parameter
    if let Some(s) = sort {
        match s.as_str() {
            "downloads" => url.push_str("&sort=downloads"),
            "likes" => url.push_str("&sort=likes"),
            "recent" => url.push_str("&sort=lastModified"),
            _ => url.push_str("&sort=downloads"), // default
        }
    }
    
    // ... rest of code ...
}
```

---

### Step 2: Update Frontend with FilterBar

```tsx
// TEAM-405: Marketplace with filters and sorting
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { PageContainer } from "@rbee/ui/molecules";
import { FilterBar } from "@rbee/ui/marketplace/organisms/FilterBar";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@rbee/ui/atoms";
import { Badge } from "@rbee/ui/atoms";
import { Download, Heart } from "lucide-react";
import type { Model } from "@/generated/bindings";

export function MarketplaceLlmModels() {
  const navigate = useNavigate();
  const [filters, setFilters] = useState({
    search: '',
    sort: 'downloads',
  });

  // Fetch models with filters
  const { data: rawModels = [], isLoading, error } = useQuery({
    queryKey: ["marketplace", "llm-models", filters],
    queryFn: async () => {
      const result = await invoke<Model[]>("marketplace_list_models", {
        query: filters.search || null,
        sort: filters.sort,
        limit: 50,
      });
      return result;
    },
    staleTime: 5 * 60 * 1000,
  });

  const sortOptions = [
    { value: 'downloads', label: 'Most Downloads' },
    { value: 'likes', label: 'Most Likes' },
    { value: 'recent', label: 'Recently Added' },
    { value: 'alphabetical', label: 'Alphabetical' },
  ];

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <PageContainer
      title="LLM Models"
      description="Discover and download state-of-the-art language models"
      padding="default"
    >
      {/* Filters */}
      <div className="mb-6">
        <FilterBar
          search={filters.search}
          onSearchChange={(search) => setFilters({ ...filters, search })}
          sort={filters.sort}
          onSortChange={(sort) => setFilters({ ...filters, sort })}
          sortOptions={sortOptions}
          onClearFilters={() => setFilters({ search: '', sort: 'downloads' })}
        />
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="space-y-2">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="h-14 rounded-md border bg-muted/20 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="rounded-lg border overflow-x-auto">
          <Table className="table-fixed">
            <TableHeader>
              <TableRow>
                <TableHead className="w-[40%]">Model</TableHead>
                <TableHead className="w-[12%]">Author</TableHead>
                <TableHead className="w-[12%] text-right">
                  <Download className="size-3 inline mr-1" />
                  DL
                </TableHead>
                <TableHead className="w-[12%] text-right">
                  <Heart className="size-3 inline mr-1" />
                  Likes
                </TableHead>
                <TableHead className="w-[24%]">Tags</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rawModels.map((model) => (
                <TableRow
                  key={model.id}
                  className="cursor-pointer"
                  onClick={() => navigate(`/marketplace/llm-models/${encodeURIComponent(model.id)}`)}
                >
                  <TableCell>
                    <div className="min-w-0">
                      <div className="font-semibold truncate">{model.name}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {model.description}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell className="truncate">{model.author || "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(model.downloads)}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {formatNumber(model.likes)}
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {model.tags.slice(0, 2).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {model.tags.length > 2 && (
                        <Badge variant="outline" className="text-xs">
                          +{model.tags.length - 2}
                        </Badge>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </PageContainer>
  );
}
```

---

### Step 3: Add Pagination (Optional)

**With Pagination component:**
```tsx
const [page, setPage] = useState(1)
const itemsPerPage = 20

// After table
<div className="mt-6">
  <Pagination>
    <PaginationContent>
      <PaginationItem>
        <PaginationPrevious 
          onClick={() => setPage(p => Math.max(1, p - 1))}
          className={page === 1 ? 'pointer-events-none opacity-50' : ''}
        />
      </PaginationItem>
      
      {[...Array(totalPages)].map((_, i) => (
        <PaginationItem key={i}>
          <PaginationLink
            onClick={() => setPage(i + 1)}
            isActive={page === i + 1}
          >
            {i + 1}
          </PaginationLink>
        </PaginationItem>
      ))}
      
      <PaginationItem>
        <PaginationNext 
          onClick={() => setPage(p => Math.min(totalPages, p + 1))}
          className={page === totalPages ? 'pointer-events-none opacity-50' : ''}
        />
      </PaginationItem>
    </PaginationContent>
  </Pagination>
</div>
```

---

## 📋 Feature Checklist

### Essential (MVP)
- [x] Search (debounced)
- [ ] Sort by downloads
- [ ] Sort by likes
- [ ] Sort by recent
- [ ] Clear filters button

### Nice to Have
- [ ] Filter chips (by tags)
- [ ] Pagination
- [ ] Results count
- [ ] Loading skeletons
- [ ] Empty states

### Advanced
- [ ] Advanced filters (modal)
- [ ] Save filter presets
- [ ] URL state sync
- [ ] Infinite scroll
- [ ] Virtual scrolling (for 1000+ items)

---

## 🎯 Quick Start

**Minimal implementation (5 minutes):**
1. Import `FilterBar` from `@rbee/ui/marketplace/organisms/FilterBar`
2. Add state: `const [filters, setFilters] = useState({ search: '', sort: 'downloads' })`
3. Pass filters to React Query
4. Render `<FilterBar />` above table

**Full implementation (30 minutes):**
1. Update backend to support sorting
2. Add FilterBar with all sort options
3. Add filter chips for tags
4. Add pagination
5. Add loading/empty states

---

## 📚 References

**Components:**
- FilterBar: `frontend/packages/rbee-ui/src/marketplace/organisms/FilterBar/`
- Pagination: `frontend/packages/rbee-ui/src/atoms/Pagination/`
- ModelListTemplate: `frontend/packages/rbee-ui/src/marketplace/templates/ModelListTemplate/`

**Examples:**
- FilterBar stories: `FilterBar.stories.tsx`
- Pagination stories: `Pagination.stories.tsx`
- Table with pagination: `Table.stories.tsx` (WithPagination story)

---

**TEAM-405: Complete playbook for listing features! 📚**
