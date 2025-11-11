# TEAM-476: Model List Container

**Date:** 2025-11-11  
**Status:** ✅ COMPLETE  
**Purpose:** Reusable data layer container for both CivitAI and HuggingFace model lists

## Component: `ModelListContainer`

**Location:** `/src/components/ModelListContainer.tsx`

### Purpose

Handles all data fetching logic for model lists, allowing presentation components to focus purely on rendering.

### Props

```typescript
interface ModelListContainerProps {
  source: 'civitai' | 'huggingface'
  children: (props: ModelListRenderProps) => ReactNode
}
```

### Render Props Pattern

```typescript
interface ModelListRenderProps {
  models: MarketplaceModel[]
  loading: boolean
  error: Error | null
  pagination: {
    page: number
    limit: number
    total?: number
    hasNext: boolean
  }
  refetch: () => void
}
```

## Usage

### CivitAI Page Example

```typescript
import { ModelListContainer } from '@/components/ModelListContainer'

export default function CivitAIModelsPage() {
  return (
    <ModelListContainer source="civitai">
      {({ models, loading, error, pagination, refetch }) => (
        <div>
          {loading && <div>Loading...</div>}
          {error && <div>Error: {error.message}</div>}
          
          {/* Card Grid View */}
          <div className="grid grid-cols-3 gap-6">
            {models.map(model => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
          
          {/* Pagination */}
          <Pagination {...pagination} />
        </div>
      )}
    </ModelListContainer>
  )
}
```

### HuggingFace Page Example

```typescript
import { ModelListContainer } from '@/components/ModelListContainer'

export default function HuggingFaceModelsPage() {
  return (
    <ModelListContainer source="huggingface">
      {({ models, loading, error, pagination, refetch }) => (
        <div>
          {loading && <div>Loading...</div>}
          {error && <div>Error: {error.message}</div>}
          
          {/* Table View */}
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Author</th>
                <th>Downloads</th>
                <th>Likes</th>
              </tr>
            </thead>
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
          
          {/* Pagination */}
          <Pagination {...pagination} />
        </div>
      )}
    </ModelListContainer>
  )
}
```

## Features

✅ **Unified Data Layer** - Same container for both sources  
✅ **Render Props Pattern** - Children receive data via function  
✅ **Loading States** - `loading` boolean for spinners  
✅ **Error Handling** - `error` object for error messages  
✅ **Pagination** - Page, limit, total, hasNext  
✅ **Refetch** - Manual refetch function  
✅ **Dynamic Imports** - Lazy loads adapters  
✅ **Type Safe** - Full TypeScript support  

## Data Flow

```
ModelListContainer
  ↓ (source prop)
  ├─ source === 'civitai'
  │   ↓
  │   fetchCivitAIModels({ limit, page })
  │   ↓
  │   MarketplaceModel[]
  │
  └─ source === 'huggingface'
      ↓
      fetchHuggingFaceModels({ limit, sort, direction })
      ↓
      MarketplaceModel[]
  
  ↓ (render props)
  children({ models, loading, error, pagination, refetch })
  ↓
  Presentation Component (Card Grid or Table)
```

## State Management

**Internal State:**
- `models: MarketplaceModel[]` - Fetched models
- `loading: boolean` - Loading state
- `error: Error | null` - Error state
- `pagination` - Page, limit, total, hasNext

**Effects:**
- Fetches on mount
- Refetches when `source` or `page` changes

## Adapter Integration

**CivitAI:**
```typescript
const { fetchCivitAIModels } = await import('@rbee/marketplace-core/adapters/civitai')
response = await fetchCivitAIModels({ limit, page })
```

**HuggingFace:**
```typescript
const { fetchHuggingFaceModels } = await import('@rbee/marketplace-core/adapters/huggingface')
response = await fetchHuggingFaceModels({ limit, sort, direction })
```

## Benefits

✅ **Separation of Concerns** - Data layer separate from presentation  
✅ **Reusability** - Same container for both sources  
✅ **Flexibility** - Render props allow any presentation  
✅ **Type Safety** - Full TypeScript support  
✅ **Error Handling** - Centralized error handling  
✅ **Loading States** - Centralized loading states  

## Next Steps

1. ✅ Container created
2. ⏭️ Update CivitAI page to use container
3. ⏭️ Update HuggingFace page to use container
4. ⏭️ Create presentation components (Card, Table)
5. ⏭️ Add pagination controls

---

**TEAM-476 RULE ZERO:** Container handles data, children handle presentation. Clean separation of concerns.
