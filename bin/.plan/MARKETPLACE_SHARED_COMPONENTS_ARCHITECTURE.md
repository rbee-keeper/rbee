# Marketplace Shared Components Architecture

**Date:** 2025-11-04  
**Status:** 🎯 ARCHITECTURAL DECISION  
**Purpose:** ONE set of components, TWO deployment modes (SSG for SEO + SPA for Keeper)

---

## 🎯 The Vision

### SEO Goldmine
```
marketplace.rbee.dev/models/llama-3.2-1b
marketplace.rbee.dev/models/sdxl-turbo
marketplace.rbee.dev/models/mistral-7b
marketplace.rbee.dev/models/flux-dev
marketplace.rbee.dev/workers/llm-worker-cuda
```

**Every model page:**
- ✅ Pre-rendered (SSG)
- ✅ Google indexes: "Llama 3.2 1B + rbee"
- ✅ Google indexes: "SDXL Turbo + rbee"
- ✅ Google indexes: "Mistral 7B + rbee"
- ✅ Backlinks to rbee from HuggingFace/CivitAI searches

**Result:** Massive SEO boost! 🚀

### Same Components, Two Modes

```
┌─────────────────────────────────────────────────────────┐
│  @rbee/marketplace-components (Shared Package)          │
│  ├─> ModelCard.tsx                                      │
│  ├─> WorkerCard.tsx                                     │
│  ├─> MarketplaceGrid.tsx                                │
│  ├─> FilterSidebar.tsx                                  │
│  └─> SearchBar.tsx                                      │
└─────────────────────────────────────────────────────────┘
                    ↓                    ↓
        ┌───────────────────┐  ┌────────────────────┐
        │  Next.js Site     │  │  Keeper (React)    │
        │  (SSG/SSR)        │  │  (SPA)             │
        ├───────────────────┤  ├────────────────────┤
        │  marketplace.     │  │  localhost:5173    │
        │  rbee.dev         │  │                    │
        │                   │  │  Embedded in tabs  │
        │  SEO optimized    │  │  + SDK integration │
        │  Pre-rendered     │  │  + Protocol links  │
        └───────────────────┘  └────────────────────┘
```

---

## 🏗️ Architecture

### Monorepo Structure

```
frontend/
├─> packages/
│   ├─> marketplace-components/     ← SHARED COMPONENTS
│   │   ├─> src/
│   │   │   ├─> components/
│   │   │   │   ├─> ModelCard.tsx
│   │   │   │   ├─> WorkerCard.tsx
│   │   │   │   ├─> MarketplaceGrid.tsx
│   │   │   │   ├─> FilterSidebar.tsx
│   │   │   │   └─> SearchBar.tsx
│   │   │   ├─> hooks/
│   │   │   │   ├─> useModels.ts
│   │   │   │   ├─> useWorkers.ts
│   │   │   │   └─> useFilters.ts
│   │   │   ├─> types/
│   │   │   │   ├─> model.ts
│   │   │   │   └─> worker.ts
│   │   │   └─> utils/
│   │   │       ├─> formatters.ts
│   │   │       └─> validators.ts
│   │   ├─> package.json
│   │   └─> tsconfig.json
│   │
│   ├─> marketplace-sdk/            ← DATA LAYER (ABSTRACT)
│   │   ├─> src/
│   │   │   ├─> MarketplaceClient.ts    (interface)
│   │   │   ├─> HuggingFaceClient.ts
│   │   │   ├─> CivitAIClient.ts
│   │   │   └─> WorkerCatalogClient.ts
│   │   └─> package.json
│   │
│   └─> ui-components/              ← EXISTING (Button, Card, etc.)
│
├─> apps/
│   ├─> marketplace-site/           ← NEXT.JS (SSG/SSR)
│   │   ├─> app/
│   │   │   ├─> layout.tsx
│   │   │   ├─> page.tsx
│   │   │   ├─> models/
│   │   │   │   ├─> page.tsx
│   │   │   │   └─> [id]/
│   │   │   │       └─> page.tsx    ← SSG per model
│   │   │   ├─> workers/
│   │   │   │   ├─> page.tsx
│   │   │   │   └─> [id]/
│   │   │   │       └─> page.tsx    ← SSG per worker
│   │   │   └─> api/
│   │   │       └─> revalidate/
│   │   ├─> package.json
│   │   └─> next.config.js
│   │
│   └─> keeper/                     ← EXISTING KEEPER (VITE + REACT)
│       ├─> src/
│       │   ├─> pages/
│       │   │   └─> MarketplacePage.tsx  ← Uses shared components
│       │   └─> lib/
│       │       └─> marketplaceAdapter.ts ← Adapts SDK to Keeper
│       └─> package.json
```

---

## 🎨 Shared Components (Dumb/Presentational)

### Key Principle: **NO DATA FETCHING IN COMPONENTS**

Components receive data via props, don't care where it comes from.

### Example: ModelCard.tsx

```tsx
// packages/marketplace-components/src/components/ModelCard.tsx

export interface ModelCardProps {
  model: {
    id: string
    name: string
    description: string
    downloads: number
    likes: number
    size: string
    imageUrl?: string
    tags: string[]
  }
  onDownload?: (modelId: string) => void
  downloadButton?: React.ReactNode  // Customizable button
  mode?: 'ssg' | 'spa'  // Render mode
}

export function ModelCard({ 
  model, 
  onDownload, 
  downloadButton,
  mode = 'spa' 
}: ModelCardProps) {
  // PURE PRESENTATION - NO DATA FETCHING
  
  const handleDownload = () => {
    onDownload?.(model.id)
  }
  
  return (
    <Card className="model-card">
      {model.imageUrl && (
        <img 
          src={model.imageUrl} 
          alt={model.name}
          className="model-image"
        />
      )}
      
      <CardHeader>
        <h3>{model.name}</h3>
        <p className="text-muted">{model.description}</p>
      </CardHeader>
      
      <CardContent>
        <div className="model-stats">
          <span>⬇️ {formatNumber(model.downloads)}</span>
          <span>❤️ {formatNumber(model.likes)}</span>
          <span>📦 {model.size}</span>
        </div>
        
        <div className="model-tags">
          {model.tags.map(tag => (
            <Badge key={tag}>{tag}</Badge>
          ))}
        </div>
      </CardContent>
      
      <CardFooter>
        {/* Custom button or default */}
        {downloadButton || (
          <Button onClick={handleDownload}>
            Download
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
```

### Example: MarketplaceGrid.tsx

```tsx
// packages/marketplace-components/src/components/MarketplaceGrid.tsx

export interface MarketplaceGridProps<T> {
  items: T[]
  renderItem: (item: T) => React.ReactNode
  isLoading?: boolean
  error?: string
  emptyMessage?: string
}

export function MarketplaceGrid<T>({ 
  items, 
  renderItem,
  isLoading,
  error,
  emptyMessage = 'No items found'
}: MarketplaceGridProps<T>) {
  // PURE PRESENTATION
  
  if (isLoading) {
    return <LoadingSpinner />
  }
  
  if (error) {
    return <ErrorMessage message={error} />
  }
  
  if (items.length === 0) {
    return <EmptyState message={emptyMessage} />
  }
  
  return (
    <div className="marketplace-grid">
      {items.map(renderItem)}
    </div>
  )
}
```

---

## 📦 Marketplace SDK (Data Layer)

### Abstract Interface

```typescript
// packages/marketplace-sdk/src/MarketplaceClient.ts

export interface Model {
  id: string
  name: string
  description: string
  downloads: number
  likes: number
  size: string
  imageUrl?: string
  tags: string[]
  source: 'huggingface' | 'civitai'
}

export interface Worker {
  id: string
  name: string
  description: string
  version: string
  platform: string[]
  architecture: string[]
  workerType: 'cpu' | 'cuda' | 'metal'
}

export interface MarketplaceClient {
  // Models
  listModels(filters?: ModelFilters): Promise<Model[]>
  getModel(id: string): Promise<Model>
  searchModels(query: string): Promise<Model[]>
  
  // Workers
  listWorkers(filters?: WorkerFilters): Promise<Worker[]>
  getWorker(id: string): Promise<Worker>
  
  // Downloads (implementation-specific)
  downloadModel?(modelId: string): Promise<void>
  installWorker?(workerId: string): Promise<void>
}
```

### HuggingFace Implementation

```typescript
// packages/marketplace-sdk/src/HuggingFaceClient.ts

export class HuggingFaceClient implements MarketplaceClient {
  private baseUrl = 'https://huggingface.co/api'
  
  async listModels(filters?: ModelFilters): Promise<Model[]> {
    const response = await fetch(`${this.baseUrl}/models?...`)
    const data = await response.json()
    
    // Transform HuggingFace API response to our Model type
    return data.map(this.transformModel)
  }
  
  async getModel(id: string): Promise<Model> {
    const response = await fetch(`${this.baseUrl}/models/${id}`)
    const data = await response.json()
    return this.transformModel(data)
  }
  
  private transformModel(hfModel: any): Model {
    return {
      id: hfModel.id,
      name: hfModel.id,
      description: hfModel.description || '',
      downloads: hfModel.downloads || 0,
      likes: hfModel.likes || 0,
      size: hfModel.size || 'Unknown',
      imageUrl: hfModel.cardData?.thumbnail,
      tags: hfModel.tags || [],
      source: 'huggingface'
    }
  }
  
  // No download implementation (handled differently in SSG vs SPA)
}
```

### Worker Catalog Implementation

```typescript
// packages/marketplace-sdk/src/WorkerCatalogClient.ts

export class WorkerCatalogClient implements MarketplaceClient {
  private baseUrl: string
  
  constructor(baseUrl = 'http://localhost:8787') {
    this.baseUrl = baseUrl
  }
  
  async listWorkers(): Promise<Worker[]> {
    const response = await fetch(`${this.baseUrl}/workers`)
    const data = await response.json()
    return data.workers
  }
  
  async getWorker(id: string): Promise<Worker> {
    const response = await fetch(`${this.baseUrl}/workers/${id}`)
    return response.json()
  }
  
  // Models not supported
  async listModels(): Promise<Model[]> {
    throw new Error('Workers catalog does not support models')
  }
  
  async getModel(): Promise<Model> {
    throw new Error('Workers catalog does not support models')
  }
  
  async searchModels(): Promise<Model[]> {
    throw new Error('Workers catalog does not support models')
  }
}
```

---

## 🌐 Next.js Site (SSG for SEO)

### app/models/page.tsx (List Page)

```tsx
// apps/marketplace-site/app/models/page.tsx

import { HuggingFaceClient } from '@rbee/marketplace-sdk'
import { ModelCard, MarketplaceGrid } from '@rbee/marketplace-components'

export default async function ModelsPage() {
  // SERVER COMPONENT - Fetch at build time
  const client = new HuggingFaceClient()
  const models = await client.listModels({ limit: 100 })
  
  return (
    <div className="container">
      <h1>AI Models Marketplace</h1>
      <p>Download and run AI models locally with rbee</p>
      
      <MarketplaceGrid
        items={models}
        renderItem={(model) => (
          <ModelCard
            key={model.id}
            model={model}
            downloadButton={
              <a 
                href={`rbee://download/model/${model.source}/${model.id}`}
                className="btn-primary"
              >
                📦 Open in Keeper
              </a>
            }
            mode="ssg"
          />
        )}
      />
    </div>
  )
}

// Generate static page at build time
export const revalidate = 3600 // Revalidate every hour
```

### app/models/[id]/page.tsx (Detail Page - SSG)

```tsx
// apps/marketplace-site/app/models/[id]/page.tsx

import { HuggingFaceClient } from '@rbee/marketplace-sdk'
import { ModelCard } from '@rbee/marketplace-components'
import { Metadata } from 'next'

interface Props {
  params: { id: string }
}

// GENERATE STATIC PATHS FOR ALL MODELS
export async function generateStaticParams() {
  const client = new HuggingFaceClient()
  const models = await client.listModels({ limit: 1000 })
  
  return models.map(model => ({
    id: model.id
  }))
}

// GENERATE SEO METADATA
export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const client = new HuggingFaceClient()
  const model = await client.getModel(params.id)
  
  return {
    title: `${model.name} - Download with rbee`,
    description: `${model.description} | Run ${model.name} locally with rbee. Free, private, unlimited.`,
    keywords: [model.name, 'rbee', 'AI model', 'local AI', ...model.tags],
    openGraph: {
      title: model.name,
      description: model.description,
      images: model.imageUrl ? [model.imageUrl] : []
    }
  }
}

// PAGE COMPONENT
export default async function ModelDetailPage({ params }: Props) {
  const client = new HuggingFaceClient()
  const model = await client.getModel(params.id)
  
  return (
    <div className="container">
      <ModelCard
        model={model}
        downloadButton={
          <a 
            href={`rbee://download/model/huggingface/${model.id}`}
            className="btn-primary btn-lg"
          >
            📦 Open in Keeper
          </a>
        }
        mode="ssg"
      />
      
      {/* Additional SEO content */}
      <div className="model-details">
        <h2>About {model.name}</h2>
        <p>{model.description}</p>
        
        <h3>How to use with rbee</h3>
        <ol>
          <li>Install rbee Keeper</li>
          <li>Click "Open in Keeper"</li>
          <li>Download and run locally</li>
        </ol>
        
        <h3>Why use rbee?</h3>
        <ul>
          <li>✅ Free forever</li>
          <li>✅ 100% private</li>
          <li>✅ No API limits</li>
          <li>✅ Run on your hardware</li>
        </ul>
      </div>
    </div>
  )
}

export const revalidate = 3600 // Revalidate every hour
```

### next.config.js

```javascript
// apps/marketplace-site/next.config.js

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export', // Static export for Cloudflare Pages
  
  // OR for ISR (Incremental Static Regeneration)
  // output: 'standalone',
  
  images: {
    domains: ['huggingface.co', 'cdn-lfs.huggingface.co']
  },
  
  // Transpile shared packages
  transpilePackages: [
    '@rbee/marketplace-components',
    '@rbee/marketplace-sdk',
    '@rbee/ui-components'
  ]
}

module.exports = nextConfig
```

---

## 💻 Keeper Integration (SPA)

### MarketplacePage.tsx

```tsx
// apps/keeper/src/pages/MarketplacePage.tsx

import { useState, useEffect } from 'react'
import { ModelCard, MarketplaceGrid } from '@rbee/marketplace-components'
import { HuggingFaceClient } from '@rbee/marketplace-sdk'
import { useKeeperSDK } from '@/lib/keeperSDK'

export function MarketplacePage() {
  const [models, setModels] = useState<Model[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const keeperSDK = useKeeperSDK()
  
  useEffect(() => {
    // CLIENT-SIDE FETCH
    const client = new HuggingFaceClient()
    client.listModels().then(setModels).finally(() => setIsLoading(false))
  }, [])
  
  const handleDownload = async (modelId: string) => {
    // Use Keeper SDK to submit job
    await keeperSDK.downloadModel({
      hive_id: 'localhost',
      model_id: modelId,
      source: 'huggingface'
    })
    
    // Show notification
    toast.success(`Downloading ${modelId}`)
  }
  
  return (
    <div className="marketplace-page">
      <h1>AI Models Marketplace</h1>
      
      <MarketplaceGrid
        items={models}
        isLoading={isLoading}
        renderItem={(model) => (
          <ModelCard
            key={model.id}
            model={model}
            onDownload={handleDownload}
            mode="spa"
          />
        )}
      />
    </div>
  )
}
```

### Keeper SDK Adapter

```typescript
// apps/keeper/src/lib/keeperSDK.ts

export class KeeperSDK {
  private queenUrl = 'http://localhost:8500'
  
  async downloadModel(params: {
    hive_id: string
    model_id: string
    source: 'huggingface' | 'civitai'
  }) {
    const response = await fetch(`${this.queenUrl}/v1/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        operation: {
          ModelDownload: params
        }
      })
    })
    
    const { job_id } = await response.json()
    
    // Stream progress
    const eventSource = new EventSource(
      `${this.queenUrl}/v1/jobs/${job_id}/stream`
    )
    
    eventSource.onmessage = (event) => {
      console.log('Progress:', event.data)
      // Update UI
    }
    
    return job_id
  }
  
  async installWorker(params: {
    hive_id: string
    worker_id: string
  }) {
    // Similar to downloadModel
  }
}

export function useKeeperSDK() {
  return new KeeperSDK()
}
```

---

## 📊 Comparison: SSG vs SPA

| Feature | Next.js (SSG) | Keeper (SPA) |
|---------|---------------|--------------|
| Components | ✅ Same | ✅ Same |
| Data fetching | Server-side (build time) | Client-side (runtime) |
| SEO | ✅ Perfect | ❌ None |
| Download action | `rbee://` link | Keeper SDK |
| Deployment | Cloudflare Pages | Embedded in Keeper |
| URL | marketplace.rbee.dev | localhost:5173 |

---

## 🚀 Build Process

### Development

```bash
# Install dependencies
pnpm install

# Develop marketplace components (watch mode)
cd packages/marketplace-components
pnpm dev

# Develop Next.js site
cd apps/marketplace-site
pnpm dev  # localhost:3000

# Develop Keeper
cd apps/keeper
pnpm dev  # localhost:5173
```

### Production

```bash
# Build shared components
cd packages/marketplace-components
pnpm build

# Build Next.js site (SSG)
cd apps/marketplace-site
pnpm build
# Output: .next/ or out/ (static export)

# Build Keeper
cd apps/keeper
pnpm build
# Output: dist/
```

### Deployment

```bash
# Deploy Next.js to Cloudflare Pages
cd apps/marketplace-site
pnpm run deploy  # wrangler pages deploy

# Keeper is bundled with desktop app
cd apps/keeper
# dist/ is embedded in Tauri/Electron
```

---

## 🎯 SEO Strategy

### Pre-render Top 1000 Models

```typescript
// apps/marketplace-site/scripts/generate-static-paths.ts

import { HuggingFaceClient } from '@rbee/marketplace-sdk'

async function generateStaticPaths() {
  const client = new HuggingFaceClient()
  
  // Get top 1000 models by downloads
  const models = await client.listModels({ 
    sort: 'downloads',
    limit: 1000 
  })
  
  // Generate paths for Next.js
  return models.map(model => ({
    params: { id: model.id }
  }))
}
```

### Sitemap Generation

```typescript
// apps/marketplace-site/app/sitemap.ts

import { HuggingFaceClient, WorkerCatalogClient } from '@rbee/marketplace-sdk'

export default async function sitemap() {
  const hfClient = new HuggingFaceClient()
  const workerClient = new WorkerCatalogClient()
  
  const models = await hfClient.listModels({ limit: 1000 })
  const workers = await workerClient.listWorkers()
  
  return [
    {
      url: 'https://marketplace.rbee.dev',
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1
    },
    ...models.map(model => ({
      url: `https://marketplace.rbee.dev/models/${model.id}`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.8
    })),
    ...workers.map(worker => ({
      url: `https://marketplace.rbee.dev/workers/${worker.id}`,
      lastModified: new Date(),
      changeFrequency: 'monthly',
      priority: 0.6
    }))
  ]
}
```

### robots.txt

```txt
# apps/marketplace-site/public/robots.txt

User-agent: *
Allow: /

Sitemap: https://marketplace.rbee.dev/sitemap.xml
```

---

## ✅ Benefits

### 1. **ZERO Component Duplication**
- ✅ Write once, use everywhere
- ✅ Same UI in Next.js and Keeper
- ✅ Single source of truth

### 2. **SEO Goldmine**
- ✅ Every model gets its own page
- ✅ Pre-rendered with metadata
- ✅ Google indexes: "Llama 3.2 + rbee"
- ✅ Backlinks from model searches

### 3. **Flexible Data Layer**
- ✅ SDK abstracts data fetching
- ✅ Easy to add new sources (Ollama, LocalAI, etc.)
- ✅ Easy to mock for testing

### 4. **Clean Separation**
- ✅ Components = presentation only
- ✅ SDK = data fetching
- ✅ Apps = integration logic

### 5. **Easy Maintenance**
- ✅ Update component → both apps updated
- ✅ Fix bug once → fixed everywhere
- ✅ Add feature once → available everywhere

---

## 🔄 Migration Path

### Phase 1: Create Shared Packages (1 week)
1. Create `packages/marketplace-components/`
2. Create `packages/marketplace-sdk/`
3. Extract existing components to shared package
4. Make components dumb (remove data fetching)

### Phase 2: Build Next.js Site (1 week)
1. Create `apps/marketplace-site/`
2. Set up Next.js with App Router
3. Implement SSG for model pages
4. Deploy to Cloudflare Pages

### Phase 3: Integrate with Keeper (3 days)
1. Import shared components in Keeper
2. Create Keeper SDK adapter
3. Wire up download actions
4. Test end-to-end

### Phase 4: SEO Optimization (2 days)
1. Generate sitemap
2. Add metadata
3. Submit to Google Search Console
4. Monitor rankings

**Total: 2.5 weeks**

---

## 📦 Package.json Examples

### marketplace-components/package.json

```json
{
  "name": "@rbee/marketplace-components",
  "version": "1.0.0",
  "main": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": "./dist/index.js",
    "./components/*": "./dist/components/*.js"
  },
  "scripts": {
    "build": "tsup src/index.ts --format esm,cjs --dts",
    "dev": "tsup src/index.ts --format esm,cjs --dts --watch"
  },
  "peerDependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "tsup": "^8.0.0",
    "typescript": "^5.0.0"
  }
}
```

### marketplace-site/package.json

```json
{
  "name": "marketplace-site",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "deploy": "wrangler pages deploy .next"
  },
  "dependencies": {
    "next": "^14.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "@rbee/marketplace-components": "workspace:*",
    "@rbee/marketplace-sdk": "workspace:*",
    "@rbee/ui-components": "workspace:*"
  }
}
```

---

## 🎯 Summary

**Architecture:**
```
Shared Components (Dumb)
    ↓
    ├─> Next.js (SSG) → SEO goldmine
    └─> Keeper (SPA) → Embedded marketplace
```

**Key Principles:**
1. ✅ Components are DUMB (no data fetching)
2. ✅ SDK handles data (abstract interface)
3. ✅ Apps handle integration (Next.js vs Keeper)
4. ✅ ZERO duplication
5. ✅ SEO optimized (SSG)
6. ✅ Keeper integrated (SPA)

**Result:**
- 🎯 Every AI model page ranks for "model name + rbee"
- 🎯 Same components in website and Keeper
- 🎯 Easy to maintain and extend
- 🎯 Fast to build (2.5 weeks)

**THIS IS THE WAY!** 🚀
