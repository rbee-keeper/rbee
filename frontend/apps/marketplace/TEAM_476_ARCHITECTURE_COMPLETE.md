# TEAM-476: Marketplace Architecture - COMPLETE

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Rule Zero:** Applied - Dead code deleted, clean architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Page (Server Component)                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ModelPageContainer (DATA + CONTROL layer)                   │ │
│ │                                                             │ │
│ │ ┌─────────────────┐  ┌──────────────────────────────────┐ │ │
│ │ │ PageHeader      │  │ FilterBar (LEFT: filters,        │ │ │
│ │ │ - Title         │  │           RIGHT: sort + clear)   │ │ │
│ │ │ - Subtitle      │  │                                  │ │ │
│ │ └─────────────────┘  └──────────────────────────────────┘ │ │
│ │                                                             │ │
│ │ ┌─────────────────────────────────────────────────────────┐ │ │
│ │ │ fetchModels() - SSR data fetching                       │ │ │
│ │ └─────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Presentation Layer (children render prop)                   │ │
│ │                                                             │ │
│ │ HuggingFace: TABLE                                          │ │
│ │ CivitAI: IMAGE CARD GRID                                    │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components Created

### 1. `/lib/fetchModels.ts` - Server-side data fetching

```typescript
export const fetchModels = cache(async (vendor, filters) => {
  const adapter = getAdapter(vendor)
  return await adapter.fetchModels(filters)
})
```

**Purpose:** SSR data fetching with React cache

### 2. `/components/PageHeader.tsx` - Reusable header

```typescript
<PageHeader title="HuggingFace Models" subtitle="Browse language models">
  {/* Optional children (e.g., filters) */}
</PageHeader>
```

**Purpose:** Consistent title + subtitle across all pages

### 3. `/components/ModelPageContainer.tsx` - Data + Control layer

```typescript
<ModelPageContainer
  vendor="huggingface"
  title="HuggingFace Models"
  subtitle="Browse language models from HuggingFace Hub"
  filters={filters}
  filterBar={<FilterBar ... />}
>
  {({ models, pagination }) => (
    /* Presentation layer */
  )}
</ModelPageContainer>
```

**Purpose:** 
- Fetches data (SSR)
- Renders header
- Renders filter bar
- Delegates presentation to children

## Pages Implemented

### HuggingFace - TABLE Presentation

**File:** `/app/models/huggingface/page.tsx`

```typescript
export default async function HuggingFaceModelsPage({ searchParams }) {
  const filters: HuggingFaceListModelsParams = {
    ...(searchParams.search && { search: searchParams.search }),
    ...(searchParams.sort && { sort: searchParams.sort as any }),
    limit: 50,
  }

  return (
    <ModelPageContainer
      vendor="huggingface"
      title="HuggingFace Models"
      subtitle="Browse language models from HuggingFace Hub"
      filters={filters}
      filterBar={<FilterBar ... />}
    >
      {({ models }) => (
        <table>
          {/* TABLE presentation */}
          <tbody>
            {models.map(model => (
              <tr key={model.id}>
                <td>{model.name}</td>
                <td>{model.author}</td>
                <td>{model.downloads}</td>
                <td>{model.likes}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </ModelPageContainer>
  )
}
```

**Presentation:** Table with columns (Model, Author, Downloads, Likes, Type)

### CivitAI - IMAGE CARD Presentation

**File:** `/app/models/civitai/page.tsx`

```typescript
export default async function CivitAIModelsPage({ searchParams }) {
  const filters: CivitAIListModelsParams = {
    ...(searchParams.query && { query: searchParams.query }),
    ...(searchParams.sort && { sort: searchParams.sort as any }),
    limit: 50,
  }

  return (
    <ModelPageContainer
      vendor="civitai"
      title="CivitAI Models"
      subtitle="Browse image generation models from CivitAI"
      filters={filters}
      filterBar={<FilterBar ... />}
    >
      {({ models }) => (
        <div className="grid grid-cols-4 gap-6">
          {models.map(model => (
            <div key={model.id} className="card">
              {/* IMAGE + metadata */}
              <Image src={model.imageUrl} alt={model.name} />
              <h3>{model.name}</h3>
              <p>{model.author}</p>
              <div>⬇️ {model.downloads} ❤️ {model.likes}</div>
            </div>
          ))}
        </div>
      )}
    </ModelPageContainer>
  )
}
```

**Presentation:** Responsive grid (1-4 columns) with image cards

## Separation of Concerns

| Layer | Responsibility | Location |
|-------|---------------|----------|
| **Data** | Fetch models from API | `fetchModels()` |
| **Control** | Filters, sorting, pagination | `ModelPageContainer` + `FilterBar` |
| **Presentation** | How models are displayed | Page children (table vs cards) |

## Benefits

✅ **SSR** - Models rendered on server, better SEO  
✅ **Reusable** - Same container for both vendors  
✅ **Flexible** - Different presentation per vendor  
✅ **Type-safe** - Vendor-specific filters  
✅ **Clean** - No dead code, no backwards compatibility  
✅ **Cached** - React cache prevents duplicate fetches  

## Deleted Files (RULE ZERO)

❌ `/components/ModelListContainer.tsx` - Old useEffect approach  
❌ `/components/ModelList.tsx` - Unnecessary wrapper  
❌ `/EXAMPLE_SSR_PAGE.tsx` - Example file  

## Filter Components Used

From `@rbee/ui/marketplace`:

- **FilterBar** - Main container (LEFT: filters, RIGHT: sort)
- **FilterSearch** - Text search with debounce
- **FilterDropdown** - Single-select dropdown
- **FilterMultiSelect** - Multi-select with checkboxes
- **SortDropdown** - Sort selection (RIGHT side)

## Next Steps

1. ✅ SSR architecture complete
2. ⏭️ Add client-side filtering (URL params + router.push)
3. ⏭️ Add pagination (Server Actions)
4. ⏭️ Add loading.tsx for route-level loading
5. ⏭️ Add error.tsx for route-level errors

---

**TEAM-476 RULE ZERO:** Clean architecture. No dead code. Separation of concerns. SSR by default!
