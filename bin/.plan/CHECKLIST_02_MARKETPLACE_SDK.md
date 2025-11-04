# Checklist 02: Marketplace SDK Package

**Timeline:** 3 days  
**Status:** 📋 NOT STARTED  
**Dependencies:** None (can run parallel with Checklist 01)

---

## 🎯 Goal

Create `@rbee/marketplace-sdk` package with abstract data layer that works in both Next.js (server-side) and Tauri (client-side).

---

## 📦 Phase 1: Package Setup (Day 1, Morning)

### 1.1 Create Package Structure

- [ ] Create directory: `frontend/packages/marketplace-sdk/`
- [ ] Create `package.json`:
  ```json
  {
    "name": "@rbee/marketplace-sdk",
    "version": "1.0.0",
    "main": "./dist/index.js",
    "types": "./dist/index.d.ts",
    "scripts": {
      "build": "tsup src/index.ts --format esm,cjs --dts",
      "dev": "tsup src/index.ts --format esm,cjs --dts --watch",
      "test": "vitest"
    },
    "dependencies": {},
    "devDependencies": {
      "tsup": "^8.0.0",
      "typescript": "^5.0.0",
      "vitest": "^1.0.0"
    }
  }
  ```
- [ ] Create `tsconfig.json`
- [ ] Create `src/` directory
- [ ] Create `src/index.ts` (empty for now)
- [ ] Run `pnpm install`
- [ ] Verify build works: `pnpm build`

### 1.2 Define Interfaces

- [ ] Create `src/types.ts`:
  ```typescript
  export interface Model {
    id: string
    name: string
    description: string
    author?: string
    imageUrl?: string
    tags: string[]
    downloads: number
    likes: number
    size: string
    source: 'huggingface' | 'civitai'
    metadata?: Record<string, any>
  }
  
  export interface Worker {
    id: string
    name: string
    description: string
    version: string
    platform: string[]
    architecture: string[]
    workerType: 'cpu' | 'cuda' | 'metal'
    metadata?: Record<string, any>
  }
  
  export interface ModelFilters {
    search?: string
    category?: string
    sort?: 'popular' | 'recent' | 'trending'
    limit?: number
  }
  
  export interface WorkerFilters {
    workerType?: 'cpu' | 'cuda' | 'metal'
    platform?: string
  }
  ```
- [ ] Export from `src/index.ts`

### 1.3 Define Abstract Interface

- [ ] Create `src/MarketplaceClient.ts`:
  ```typescript
  import { Model, Worker, ModelFilters, WorkerFilters } from './types'
  
  export interface MarketplaceClient {
    // Models
    listModels(filters?: ModelFilters): Promise<Model[]>
    getModel(id: string): Promise<Model>
    searchModels(query: string): Promise<Model[]>
    
    // Workers
    listWorkers(filters?: WorkerFilters): Promise<Worker[]>
    getWorker(id: string): Promise<Worker>
    
    // Optional: Download/Install (implementation-specific)
    downloadModel?(modelId: string): Promise<void>
    installWorker?(workerId: string): Promise<void>
  }
  ```
- [ ] Export from `src/index.ts`

---

## 🤗 Phase 2: HuggingFace Client (Day 1, Afternoon)

### 2.1 Create HuggingFace Client

- [ ] Create `src/HuggingFaceClient.ts`
- [ ] Define class:
  ```typescript
  export class HuggingFaceClient implements MarketplaceClient {
    private baseUrl = 'https://huggingface.co/api'
    private apiToken?: string
    
    constructor(apiToken?: string) {
      this.apiToken = apiToken
    }
    
    async listModels(filters?: ModelFilters): Promise<Model[]> {
      // TODO: Implement
    }
    
    async getModel(id: string): Promise<Model> {
      // TODO: Implement
    }
    
    async searchModels(query: string): Promise<Model[]> {
      // TODO: Implement
    }
    
    async listWorkers(): Promise<Worker[]> {
      throw new Error('HuggingFace does not support workers')
    }
    
    async getWorker(): Promise<Worker> {
      throw new Error('HuggingFace does not support workers')
    }
  }
  ```

### 2.2 Implement listModels

- [ ] Research HuggingFace API endpoint for listing models
- [ ] Implement fetch with filters:
  ```typescript
  async listModels(filters?: ModelFilters): Promise<Model[]> {
    const params = new URLSearchParams()
    if (filters?.search) params.append('search', filters.search)
    if (filters?.sort) params.append('sort', filters.sort)
    if (filters?.limit) params.append('limit', filters.limit.toString())
    
    const url = `${this.baseUrl}/models?${params}`
    const response = await fetch(url, {
      headers: this.apiToken ? {
        'Authorization': `Bearer ${this.apiToken}`
      } : {}
    })
    
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.statusText}`)
    }
    
    const data = await response.json()
    return data.map(this.transformModel)
  }
  ```
- [ ] Implement transformModel helper:
  ```typescript
  private transformModel(hfModel: any): Model {
    return {
      id: hfModel.id || hfModel.modelId,
      name: hfModel.id || hfModel.modelId,
      description: hfModel.description || '',
      author: hfModel.author,
      imageUrl: hfModel.cardData?.thumbnail,
      tags: hfModel.tags || [],
      downloads: hfModel.downloads || 0,
      likes: hfModel.likes || 0,
      size: hfModel.size || 'Unknown',
      source: 'huggingface',
      metadata: {
        modelId: hfModel.modelId,
        pipeline_tag: hfModel.pipeline_tag,
        library_name: hfModel.library_name
      }
    }
  }
  ```
- [ ] Test with real API call
- [ ] Handle errors gracefully

### 2.3 Implement getModel

- [ ] Implement single model fetch:
  ```typescript
  async getModel(id: string): Promise<Model> {
    const url = `${this.baseUrl}/models/${id}`
    const response = await fetch(url, {
      headers: this.apiToken ? {
        'Authorization': `Bearer ${this.apiToken}`
      } : {}
    })
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Model not found: ${id}`)
      }
      throw new Error(`Failed to fetch model: ${response.statusText}`)
    }
    
    const data = await response.json()
    return this.transformModel(data)
  }
  ```
- [ ] Test with real model ID
- [ ] Handle 404 errors

### 2.4 Implement searchModels

- [ ] Implement search:
  ```typescript
  async searchModels(query: string): Promise<Model[]> {
    return this.listModels({ search: query, limit: 50 })
  }
  ```
- [ ] Test with various queries

### 2.5 Export Client

- [ ] Export from `src/index.ts`:
  ```typescript
  export { HuggingFaceClient } from './HuggingFaceClient'
  ```

---

## 🎨 Phase 3: CivitAI Client (Day 2, Morning)

### 3.1 Create CivitAI Client

- [ ] Create `src/CivitAIClient.ts`
- [ ] Define class:
  ```typescript
  export class CivitAIClient implements MarketplaceClient {
    private baseUrl = 'https://civitai.com/api/v1'
    private apiToken?: string
    
    constructor(apiToken?: string) {
      this.apiToken = apiToken
    }
    
    async listModels(filters?: ModelFilters): Promise<Model[]> {
      // TODO: Implement
    }
    
    async getModel(id: string): Promise<Model> {
      // TODO: Implement
    }
    
    async searchModels(query: string): Promise<Model[]> {
      // TODO: Implement
    }
    
    async listWorkers(): Promise<Worker[]> {
      throw new Error('CivitAI does not support workers')
    }
    
    async getWorker(): Promise<Worker> {
      throw new Error('CivitAI does not support workers')
    }
  }
  ```

### 3.2 Implement listModels

- [ ] Research CivitAI API endpoint
- [ ] Implement fetch with filters:
  ```typescript
  async listModels(filters?: ModelFilters): Promise<Model[]> {
    const params = new URLSearchParams()
    if (filters?.search) params.append('query', filters.search)
    if (filters?.sort) params.append('sort', filters.sort)
    if (filters?.limit) params.append('limit', filters.limit.toString())
    
    const url = `${this.baseUrl}/models?${params}`
    const response = await fetch(url, {
      headers: this.apiToken ? {
        'Authorization': `Bearer ${this.apiToken}`
      } : {}
    })
    
    if (!response.ok) {
      throw new Error(`Failed to fetch models: ${response.statusText}`)
    }
    
    const data = await response.json()
    return data.items.map(this.transformModel)
  }
  ```
- [ ] Implement transformModel helper:
  ```typescript
  private transformModel(civitModel: any): Model {
    const latestVersion = civitModel.modelVersions?.[0]
    
    return {
      id: `civitai:${civitModel.id}`,
      name: civitModel.name,
      description: civitModel.description || '',
      author: civitModel.creator?.username,
      imageUrl: latestVersion?.images?.[0]?.url,
      tags: civitModel.tags || [],
      downloads: civitModel.stats?.downloadCount || 0,
      likes: civitModel.stats?.favoriteCount || 0,
      size: latestVersion?.files?.[0]?.sizeKB 
        ? `${(latestVersion.files[0].sizeKB / 1024 / 1024).toFixed(1)} GB`
        : 'Unknown',
      source: 'civitai',
      metadata: {
        modelId: civitModel.id,
        type: civitModel.type,
        nsfw: civitModel.nsfw,
        versionId: latestVersion?.id
      }
    }
  }
  ```
- [ ] Test with real API call
- [ ] Handle NSFW filtering

### 3.3 Implement getModel

- [ ] Implement single model fetch:
  ```typescript
  async getModel(id: string): Promise<Model> {
    // Extract numeric ID from "civitai:123456" format
    const numericId = id.replace('civitai:', '')
    
    const url = `${this.baseUrl}/models/${numericId}`
    const response = await fetch(url, {
      headers: this.apiToken ? {
        'Authorization': `Bearer ${this.apiToken}`
      } : {}
    })
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Model not found: ${id}`)
      }
      throw new Error(`Failed to fetch model: ${response.statusText}`)
    }
    
    const data = await response.json()
    return this.transformModel(data)
  }
  ```
- [ ] Test with real model ID

### 3.4 Implement searchModels

- [ ] Implement search:
  ```typescript
  async searchModels(query: string): Promise<Model[]> {
    return this.listModels({ search: query, limit: 50 })
  }
  ```

### 3.5 Export Client

- [ ] Export from `src/index.ts`:
  ```typescript
  export { CivitAIClient } from './CivitAIClient'
  ```

---

## 👷 Phase 4: Worker Catalog Client (Day 2, Afternoon)

### 4.1 Create Worker Catalog Client

- [ ] Create `src/WorkerCatalogClient.ts`
- [ ] Define class:
  ```typescript
  export class WorkerCatalogClient implements MarketplaceClient {
    private baseUrl: string
    
    constructor(baseUrl = 'http://localhost:8787') {
      this.baseUrl = baseUrl
    }
    
    async listModels(): Promise<Model[]> {
      throw new Error('Worker catalog does not support models')
    }
    
    async getModel(): Promise<Model> {
      throw new Error('Worker catalog does not support models')
    }
    
    async searchModels(): Promise<Model[]> {
      throw new Error('Worker catalog does not support models')
    }
    
    async listWorkers(filters?: WorkerFilters): Promise<Worker[]> {
      // TODO: Implement
    }
    
    async getWorker(id: string): Promise<Worker> {
      // TODO: Implement
    }
  }
  ```

### 4.2 Implement listWorkers

- [ ] Implement fetch:
  ```typescript
  async listWorkers(filters?: WorkerFilters): Promise<Worker[]> {
    const url = `${this.baseUrl}/workers`
    const response = await fetch(url)
    
    if (!response.ok) {
      throw new Error(`Failed to fetch workers: ${response.statusText}`)
    }
    
    const data = await response.json()
    let workers = data.workers || []
    
    // Apply filters
    if (filters?.workerType) {
      workers = workers.filter((w: any) => 
        w.worker_type === filters.workerType
      )
    }
    
    if (filters?.platform) {
      workers = workers.filter((w: any) => 
        w.platform.includes(filters.platform)
      )
    }
    
    return workers.map(this.transformWorker)
  }
  ```
- [ ] Implement transformWorker helper:
  ```typescript
  private transformWorker(catalogWorker: any): Worker {
    return {
      id: catalogWorker.id,
      name: catalogWorker.name || catalogWorker.id,
      description: catalogWorker.description || '',
      version: catalogWorker.version || '0.1.0',
      platform: catalogWorker.platform || [],
      architecture: catalogWorker.architecture || [],
      workerType: catalogWorker.worker_type || 'cpu',
      metadata: {
        pkgbuildUrl: catalogWorker.pkgbuild_url,
        dependencies: catalogWorker.dependencies
      }
    }
  }
  ```
- [ ] Test with real catalog service

### 4.3 Implement getWorker

- [ ] Implement single worker fetch:
  ```typescript
  async getWorker(id: string): Promise<Worker> {
    const url = `${this.baseUrl}/workers/${id}`
    const response = await fetch(url)
    
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Worker not found: ${id}`)
      }
      throw new Error(`Failed to fetch worker: ${response.statusText}`)
    }
    
    const data = await response.json()
    return this.transformWorker(data)
  }
  ```
- [ ] Test with real worker ID

### 4.4 Export Client

- [ ] Export from `src/index.ts`:
  ```typescript
  export { WorkerCatalogClient } from './WorkerCatalogClient'
  ```

---

## ✅ Phase 5: Testing & Documentation (Day 3)

### 5.1 Unit Tests

- [ ] Install dependencies:
  ```bash
  pnpm add -D vitest @vitest/ui
  ```
- [ ] Create `vitest.config.ts`
- [ ] Create `src/__tests__/HuggingFaceClient.test.ts`:
  - [ ] Test listModels with filters
  - [ ] Test getModel with valid ID
  - [ ] Test getModel with invalid ID (404)
  - [ ] Test searchModels
  - [ ] Test error handling
- [ ] Create `src/__tests__/CivitAIClient.test.ts`:
  - [ ] Test listModels with filters
  - [ ] Test getModel with valid ID
  - [ ] Test NSFW filtering
  - [ ] Test error handling
- [ ] Create `src/__tests__/WorkerCatalogClient.test.ts`:
  - [ ] Test listWorkers with filters
  - [ ] Test getWorker with valid ID
  - [ ] Test error handling
- [ ] Run tests: `pnpm test`
- [ ] Verify all tests pass

### 5.2 Integration Tests

- [ ] Create `src/__tests__/integration.test.ts`
- [ ] Test HuggingFace API (real calls):
  - [ ] List popular models
  - [ ] Get specific model
  - [ ] Search for "llama"
- [ ] Test CivitAI API (real calls):
  - [ ] List popular models
  - [ ] Get specific model
  - [ ] Search for "sdxl"
- [ ] Test Worker Catalog (requires service running):
  - [ ] List all workers
  - [ ] Get specific worker
- [ ] Mark as integration tests (skip in CI)

### 5.3 Documentation

- [ ] Create `README.md`:
  ```markdown
  # @rbee/marketplace-sdk
  
  Data layer for marketplace integrations.
  
  ## Installation
  
  \`\`\`bash
  pnpm add @rbee/marketplace-sdk
  \`\`\`
  
  ## Usage
  
  ### HuggingFace
  
  \`\`\`typescript
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  
  const client = new HuggingFaceClient(apiToken)
  const models = await client.listModels({ limit: 10 })
  \`\`\`
  
  ### CivitAI
  
  \`\`\`typescript
  import { CivitAIClient } from '@rbee/marketplace-sdk'
  
  const client = new CivitAIClient(apiToken)
  const models = await client.listModels({ limit: 10 })
  \`\`\`
  
  ### Worker Catalog
  
  \`\`\`typescript
  import { WorkerCatalogClient } from '@rbee/marketplace-sdk'
  
  const client = new WorkerCatalogClient('http://localhost:8787')
  const workers = await client.listWorkers()
  \`\`\`
  
  ## API
  
  ### MarketplaceClient Interface
  
  All clients implement this interface:
  
  - `listModels(filters?)` - List models
  - `getModel(id)` - Get single model
  - `searchModels(query)` - Search models
  - `listWorkers(filters?)` - List workers
  - `getWorker(id)` - Get single worker
  
  ### Filters
  
  - `ModelFilters` - search, category, sort, limit
  - `WorkerFilters` - workerType, platform
  
  ## Error Handling
  
  All methods throw errors on failure:
  
  \`\`\`typescript
  try {
    const model = await client.getModel('invalid-id')
  } catch (error) {
    console.error('Failed to fetch model:', error.message)
  }
  \`\`\`
  ```
- [ ] Add API documentation for each client
- [ ] Add examples for common use cases

### 5.4 Build & Publish

- [ ] Build package: `pnpm build`
- [ ] Verify exports work
- [ ] Test in example Next.js app
- [ ] Test in example Tauri app
- [ ] Publish to workspace (not npm yet)

---

## 📊 Success Criteria

### Must Have

- [ ] All 3 clients implemented
- [ ] All clients implement MarketplaceClient interface
- [ ] HuggingFace client works with real API
- [ ] CivitAI client works with real API
- [ ] Worker Catalog client works with service
- [ ] Unit tests pass
- [ ] README with examples
- [ ] Package builds successfully

### Nice to Have

- [ ] Integration tests with real APIs
- [ ] Error retry logic
- [ ] Caching layer
- [ ] Rate limiting
- [ ] TypeScript strict mode

---

## 🚀 Deliverables

1. **Package:** `@rbee/marketplace-sdk` published to workspace
2. **Clients:** 3 marketplace clients
3. **Interface:** Abstract MarketplaceClient interface
4. **Tests:** Unit tests for all clients
5. **Documentation:** README with examples

---

## 📝 Notes

### Key Principles

1. **ABSTRACT INTERFACE** - All clients implement same interface
2. **ERROR HANDLING** - Throw descriptive errors
3. **TRANSFORMATION** - Transform API responses to common format
4. **OPTIONAL AUTH** - Support API tokens but don't require them
5. **FILTERS** - Support common filtering patterns

### API Tokens

- **HuggingFace:** Optional, increases rate limits
- **CivitAI:** Optional, required for NSFW content
- **Worker Catalog:** Not required (local service)

### Common Pitfalls

- ❌ Don't hardcode API URLs (make configurable)
- ❌ Don't swallow errors (throw with context)
- ❌ Don't assume API structure (validate responses)
- ✅ Transform API responses to common format
- ✅ Handle 404s gracefully
- ✅ Support optional authentication

---

**Complete each phase in order, test thoroughly!** ✅
