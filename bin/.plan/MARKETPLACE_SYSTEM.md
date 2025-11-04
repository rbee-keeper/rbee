# Marketplace System: Models & Workers

**Date:** 2025-11-04  
**Status:** 🎯 DESIGN SPEC  
**Purpose:** Unified marketplace UI for browsing and downloading

---

## 🎯 Three Marketplaces

### 1. HuggingFace Models Marketplace

**URL:** `/marketplace/huggingface`  
**Icon:** 🤗  
**Purpose:** Browse and download LLM models (GGUF format)

**Features:**
- Search by model name
- Filter by architecture (Llama, Mistral, Phi, etc.)
- Filter by size (7B, 13B, 70B, etc.)
- Filter by quantization (Q4_K_M, Q5_K_M, F16, etc.)
- Sort by popularity, date, size
- Download with progress tracking
- See what's already downloaded

### 2. CivitAI Models Marketplace

**URL:** `/marketplace/civitai`  
**Icon:** 🎨  
**Purpose:** Browse and download SD models (SafeTensors/GGUF)

**Features:**
- Search by model name
- Filter by SD version (v1.5, v2.1, XL, Turbo)
- Filter by style (Realistic, Anime, 3D, etc.)
- Filter by NSFW (toggle)
- Sort by popularity, trending, date
- Download with progress tracking
- See what's already downloaded
- Preview images

### 3. Worker Catalog Marketplace

**URL:** `/marketplace/workers`  
**Icon:** 👷  
**Purpose:** Browse and install worker binaries

**Features:**
- Browse available workers (LLM, SD)
- Filter by device type (CPU, CUDA, Metal)
- See installation status
- Install with progress tracking
- View build logs
- See worker metadata

---

## 🏗️ Shared Marketplace Template

### Component Structure

```
MarketplaceTemplate
├── MarketplaceHeader
│   ├── Title & Icon
│   ├── Search bar
│   └── View toggle (Grid/List)
├── MarketplaceSidebar
│   ├── Filters
│   │   ├── Category
│   │   ├── Size/Version
│   │   ├── Sort options
│   │   └── Clear filters
│   └── Active filters chips
└── MarketplaceContent
    ├── Loading state
    ├── Empty state
    ├── Error state
    └── Results grid/list
        └── MarketplaceCard
            ├── Image/Icon
            ├── Title & Author
            ├── Description
            ├── Tags
            ├── Stats (downloads, likes, etc.)
            ├── Status badge
            └── Action button
```

### Shared Types

**File:** `src/types/marketplace.ts`

```typescript
export type MarketplaceType = 'huggingface' | 'civitai' | 'workers'

export interface MarketplaceItem {
  id: string
  name: string
  description: string
  author?: string
  imageUrl?: string
  tags: string[]
  stats: {
    downloads?: number
    likes?: number
    rating?: number
  }
  
  // Status
  status: 'available' | 'downloading' | 'downloaded' | 'installed'
  downloadProgress?: number
  
  // Type-specific metadata
  metadata: Record<string, any>
}

export interface MarketplaceFilters {
  search: string
  category?: string
  size?: string
  sort: 'popular' | 'recent' | 'trending' | 'name'
  nsfw?: boolean
}

export interface MarketplaceConfig {
  type: MarketplaceType
  title: string
  icon: string
  apiEndpoint: string
  filters: FilterConfig[]
  cardTemplate: 'model' | 'worker'
}
```

### Template Component

**File:** `src/components/MarketplaceTemplate.tsx`

```tsx
import { useState } from 'react'
import { Search, Grid3x3, List, Filter } from 'lucide-react'
import type { MarketplaceItem, MarketplaceFilters, MarketplaceConfig } from '@/types/marketplace'

interface MarketplaceTemplateProps {
  config: MarketplaceConfig
  items: MarketplaceItem[]
  loading: boolean
  error?: string
  onDownload: (item: MarketplaceItem) => void
  onInstall?: (item: MarketplaceItem) => void
}

export function MarketplaceTemplate({
  config,
  items,
  loading,
  error,
  onDownload,
  onInstall,
}: MarketplaceTemplateProps) {
  const [view, setView] = useState<'grid' | 'list'>('grid')
  const [filters, setFilters] = useState<MarketplaceFilters>({
    search: '',
    sort: 'popular',
  })
  const [showFilters, setShowFilters] = useState(true)
  
  const filteredItems = applyFilters(items, filters)
  
  return (
    <div className="marketplace-template">
      {/* Header */}
      <div className="marketplace-header">
        <div className="title-section">
          <span className="icon">{config.icon}</span>
          <h1>{config.title}</h1>
        </div>
        
        <div className="search-section">
          <div className="search-bar">
            <Search size={20} />
            <input
              type="text"
              placeholder="Search..."
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            />
          </div>
          
          <div className="view-toggles">
            <button
              className={view === 'grid' ? 'active' : ''}
              onClick={() => setView('grid')}
            >
              <Grid3x3 size={20} />
            </button>
            <button
              className={view === 'list' ? 'active' : ''}
              onClick={() => setView('list')}
            >
              <List size={20} />
            </button>
            <button onClick={() => setShowFilters(!showFilters)}>
              <Filter size={20} />
            </button>
          </div>
        </div>
      </div>
      
      <div className="marketplace-body">
        {/* Sidebar Filters */}
        {showFilters && (
          <div className="marketplace-sidebar">
            <h3>Filters</h3>
            
            {config.filters.map(filter => (
              <FilterSection
                key={filter.id}
                filter={filter}
                value={filters[filter.id]}
                onChange={(value) => setFilters({ ...filters, [filter.id]: value })}
              />
            ))}
            
            <button
              className="clear-filters"
              onClick={() => setFilters({ search: '', sort: 'popular' })}
            >
              Clear All
            </button>
          </div>
        )}
        
        {/* Content Area */}
        <div className="marketplace-content">
          {loading && <LoadingState />}
          
          {error && <ErrorState message={error} />}
          
          {!loading && !error && filteredItems.length === 0 && (
            <EmptyState message="No items found" />
          )}
          
          {!loading && !error && filteredItems.length > 0 && (
            <div className={`results-${view}`}>
              {filteredItems.map(item => (
                <MarketplaceCard
                  key={item.id}
                  item={item}
                  template={config.cardTemplate}
                  onDownload={() => onDownload(item)}
                  onInstall={onInstall ? () => onInstall(item) : undefined}
                />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
```

### Marketplace Card

**File:** `src/components/MarketplaceCard.tsx`

```tsx
import { Download, Check, Loader2, ExternalLink } from 'lucide-react'
import type { MarketplaceItem } from '@/types/marketplace'

interface MarketplaceCardProps {
  item: MarketplaceItem
  template: 'model' | 'worker'
  onDownload: () => void
  onInstall?: () => void
}

export function MarketplaceCard({ item, template, onDownload, onInstall }: MarketplaceCardProps) {
  return (
    <div className="marketplace-card">
      {/* Image/Icon */}
      <div className="card-image">
        {item.imageUrl ? (
          <img src={item.imageUrl} alt={item.name} />
        ) : (
          <div className="placeholder-icon">
            {template === 'model' ? '📦' : '👷'}
          </div>
        )}
        
        {/* Status Badge */}
        <div className={`status-badge ${item.status}`}>
          {item.status === 'downloaded' && <Check size={16} />}
          {item.status === 'downloading' && <Loader2 size={16} className="animate-spin" />}
          {item.status === 'installed' && <Check size={16} />}
        </div>
      </div>
      
      {/* Content */}
      <div className="card-content">
        <h3 className="card-title">{item.name}</h3>
        {item.author && <p className="card-author">by {item.author}</p>}
        <p className="card-description">{item.description}</p>
        
        {/* Tags */}
        <div className="card-tags">
          {item.tags.slice(0, 3).map(tag => (
            <span key={tag} className="tag">{tag}</span>
          ))}
        </div>
        
        {/* Stats */}
        <div className="card-stats">
          {item.stats.downloads && (
            <span>⬇️ {formatNumber(item.stats.downloads)}</span>
          )}
          {item.stats.likes && (
            <span>❤️ {formatNumber(item.stats.likes)}</span>
          )}
          {item.stats.rating && (
            <span>⭐ {item.stats.rating.toFixed(1)}</span>
          )}
        </div>
      </div>
      
      {/* Actions */}
      <div className="card-actions">
        {item.status === 'available' && (
          <button className="btn-primary" onClick={onDownload}>
            <Download size={16} />
            Download
          </button>
        )}
        
        {item.status === 'downloading' && (
          <div className="download-progress">
            <div className="progress-bar">
              <div 
                className="progress-fill" 
                style={{ width: `${item.downloadProgress}%` }}
              />
            </div>
            <span>{item.downloadProgress}%</span>
          </div>
        )}
        
        {item.status === 'downloaded' && (
          <button className="btn-success" disabled>
            <Check size={16} />
            Downloaded
          </button>
        )}
        
        {item.status === 'installed' && onInstall && (
          <button className="btn-success" disabled>
            <Check size={16} />
            Installed
          </button>
        )}
        
        <button className="btn-ghost">
          <ExternalLink size={16} />
          Details
        </button>
      </div>
    </div>
  )
}
```

---

## 🤗 HuggingFace Marketplace

### Configuration

```typescript
const huggingfaceConfig: MarketplaceConfig = {
  type: 'huggingface',
  title: 'HuggingFace Models',
  icon: '🤗',
  apiEndpoint: '/api/marketplace/huggingface',
  cardTemplate: 'model',
  filters: [
    {
      id: 'category',
      label: 'Architecture',
      type: 'select',
      options: [
        { value: 'llama', label: 'Llama' },
        { value: 'mistral', label: 'Mistral' },
        { value: 'phi', label: 'Phi' },
        { value: 'gemma', label: 'Gemma' },
      ],
    },
    {
      id: 'size',
      label: 'Parameter Count',
      type: 'select',
      options: [
        { value: '1b', label: '1B-3B' },
        { value: '7b', label: '7B' },
        { value: '13b', label: '13B' },
        { value: '70b', label: '70B+' },
      ],
    },
    {
      id: 'quantization',
      label: 'Quantization',
      type: 'select',
      options: [
        { value: 'Q4_K_M', label: 'Q4_K_M' },
        { value: 'Q5_K_M', label: 'Q5_K_M' },
        { value: 'Q8_0', label: 'Q8_0' },
        { value: 'F16', label: 'F16' },
      ],
    },
  ],
}
```

### Data Source

**Backend endpoint:** `http://localhost:7835/api/marketplace/huggingface`

**Returns:**
```json
{
  "items": [
    {
      "id": "TheBloke/Llama-2-7B-Chat-GGUF",
      "name": "Llama 2 7B Chat",
      "description": "Meta's Llama 2 model fine-tuned for chat",
      "author": "TheBloke",
      "tags": ["llama", "7b", "chat", "Q4_K_M"],
      "stats": {
        "downloads": 1234567,
        "likes": 5678
      },
      "status": "available",
      "metadata": {
        "architecture": "llama",
        "size": "7b",
        "quantization": "Q4_K_M",
        "context_length": 4096,
        "file_size": "4.1 GB"
      }
    }
  ]
}
```

---

## 🎨 CivitAI Marketplace

### Configuration

```typescript
const civitaiConfig: MarketplaceConfig = {
  type: 'civitai',
  title: 'CivitAI Models',
  icon: '🎨',
  apiEndpoint: '/api/marketplace/civitai',
  cardTemplate: 'model',
  filters: [
    {
      id: 'category',
      label: 'SD Version',
      type: 'select',
      options: [
        { value: 'v1.5', label: 'SD 1.5' },
        { value: 'v2.1', label: 'SD 2.1' },
        { value: 'xl', label: 'SDXL' },
        { value: 'turbo', label: 'SDXL Turbo' },
      ],
    },
    {
      id: 'style',
      label: 'Style',
      type: 'select',
      options: [
        { value: 'realistic', label: 'Realistic' },
        { value: 'anime', label: 'Anime' },
        { value: '3d', label: '3D' },
        { value: 'artistic', label: 'Artistic' },
      ],
    },
    {
      id: 'nsfw',
      label: 'Show NSFW',
      type: 'toggle',
    },
  ],
}
```

### Data Source

**Backend endpoint:** `http://localhost:7835/api/marketplace/civitai`

**Uses CivitAI API:** `https://civitai.com/api/v1/models`

**Returns:**
```json
{
  "items": [
    {
      "id": "civitai:101055",
      "name": "SDXL Turbo",
      "description": "Fast single-step image generation",
      "author": "Stability AI",
      "imageUrl": "https://...",
      "tags": ["sdxl", "turbo", "fast"],
      "stats": {
        "downloads": 987654,
        "likes": 4321,
        "rating": 4.8
      },
      "status": "available",
      "metadata": {
        "version": "xl_turbo",
        "file_size": "6.9 GB",
        "format": "safetensors"
      }
    }
  ]
}
```

---

## 👷 Worker Catalog Marketplace

### Configuration

```typescript
const workerCatalogConfig: MarketplaceConfig = {
  type: 'workers',
  title: 'Worker Catalog',
  icon: '👷',
  apiEndpoint: 'http://localhost:8502/workers',  // 80-hono-worker-catalog
  cardTemplate: 'worker',
  filters: [
    {
      id: 'category',
      label: 'Worker Type',
      type: 'select',
      options: [
        { value: 'llm', label: 'LLM Workers' },
        { value: 'sd', label: 'SD Workers' },
      ],
    },
    {
      id: 'device',
      label: 'Device',
      type: 'select',
      options: [
        { value: 'cpu', label: 'CPU' },
        { value: 'cuda', label: 'CUDA (NVIDIA)' },
        { value: 'metal', label: 'Metal (Apple)' },
      ],
    },
  ],
}
```

### Data Source

**Backend:** `bin/80-hono-worker-catalog/src/index.ts`

**Endpoint:** `http://localhost:8502/workers`

**Returns:**
```json
{
  "workers": [
    {
      "id": "llm-worker-rbee-cpu",
      "name": "LLM Worker (CPU)",
      "description": "Run LLM inference on CPU",
      "tags": ["llm", "cpu"],
      "stats": {
        "downloads": 123
      },
      "status": "available",
      "metadata": {
        "type": "llm",
        "device": "cpu",
        "platform": "linux",
        "version": "0.1.0",
        "pkgbuild_url": "http://localhost:8502/workers/llm-worker-rbee-cpu/pkgbuild"
      }
    }
  ]
}
```

---

## 🎯 Sidebar Integration

### Update KeeperSidebar

**File:** `src/components/KeeperSidebar.tsx`

**Add new section:**
```tsx
const marketplaceNavigation = [
  {
    title: 'HuggingFace',
    icon: PackageIcon,
    onClick: () => openTab('marketplace', 'HuggingFace', '🤗', '/marketplace/huggingface', {
      marketplaceType: 'huggingface'
    }),
  },
  {
    title: 'CivitAI',
    icon: ImageIcon,
    onClick: () => openTab('marketplace', 'CivitAI', '🎨', '/marketplace/civitai', {
      marketplaceType: 'civitai'
    }),
  },
  {
    title: 'Workers',
    icon: HardHatIcon,
    onClick: () => openTab('marketplace', 'Workers', '👷', '/marketplace/workers', {
      marketplaceType: 'workers'
    }),
  },
]

// In render:
<div className="space-y-2">
  <h3 className="text-xs font-semibold uppercase px-2">
    Marketplaces
  </h3>
  <nav className="space-y-1">
    {marketplaceNavigation.map((item) => (
      <button
        key={item.title}
        onClick={item.onClick}
        className="flex items-center gap-3 px-3 py-2 ..."
      >
        <item.icon className="w-4 h-4" />
        <span>{item.title}</span>
      </button>
    ))}
  </nav>
</div>
```

---

## 🚀 Implementation Timeline

**Phase 1: Template (2 days)**
- Create MarketplaceTemplate component
- Create MarketplaceCard component
- Create filter components
- Test with mock data

**Phase 2: HuggingFace (1 day)**
- Integrate with HuggingFace API
- Test model search/filter
- Test download flow

**Phase 3: CivitAI (2 days)**
- Integrate with CivitAI API
- Handle preview images
- Test download flow
- Handle NSFW toggle

**Phase 4: Workers (1 day)**
- Integrate with worker-catalog service
- Test worker installation
- Test build logs

**Total: 6 days**

---

**Unified marketplace system with consistent UX!** 🚀
