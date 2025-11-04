# Parts 3-10: Detailed Implementation Guides

**Note:** Due to size constraints, this document provides comprehensive outlines for Parts 3-10. Each part follows the same detailed structure as Parts 1-2.

---

## Part 3: CivitAI Client Implementation

**Time:** 3-4 days | **Complexity:** Medium

### Required Reading (2 hours)
- [CivitAI API Documentation](https://github.com/civitai/civitai/wiki/REST-API-Reference)
- [CivitAI Model Types](https://github.com/civitai/civitai/wiki/Model-Types)
- Understanding image generation models vs LLMs

### Module Structure
```
src/civitai/
├── mod.rs
├── client.rs          # CivitAIClient
├── types.rs           # CivitAI-specific types
├── endpoints.rs       # API URL builders
└── filters.rs         # Filtering logic
```

### Key Types
- `CivitAIModel` - Model response from API
- `CivitAIVersion` - Model version info
- `CivitAIFile` - Downloadable file info
- `ModelType` - Checkpoint, LoRA, Embedding, etc.
- `BaseModel` - SD 1.5, SDXL, Pony, etc.

### API Methods
- `list_models()` - List with pagination
- `search_models(query)` - Search by keyword
- `get_model(model_id)` - Get details
- `list_versions(model_id)` - Get all versions
- `get_download_url(version_id)` - Get download link

### Special Features
- NSFW filtering (important!)
- Creator filtering
- Base model filtering
- Sort by downloads/rating/newest
- Cursor-based pagination (different from HF)

### Testing
- Unit tests for all methods
- WASM tests in browser
- Handle NSFW content appropriately
- Test pagination cursor

### Acceptance Criteria
- [ ] All API methods work
- [ ] NSFW filtering works
- [ ] Pagination works (cursor-based)
- [ ] Type conversions to `Model`
- [ ] Tests pass
- [ ] WASM exports work

---

## Part 4: Worker Catalog Client Implementation

**Time:** 2-3 days | **Complexity:** Low-Medium

### Required Reading (1 hour)
- Worker Catalog API docs (internal)
- `bin/80-hono-worker-catalog/` codebase
- `artifacts-contract` types (already familiar)

### Module Structure
```
src/worker_catalog/
├── mod.rs
├── client.rs          # WorkerCatalogClient
└── endpoints.rs       # API URL builders
```

### Integration with artifacts-contract
```rust
// Use existing types!
use artifacts_contract::{WorkerBinary, WorkerType, Platform};

// No need for custom types - already defined
```

### API Methods
- `list_workers()` - List all workers
- `get_worker(id)` - Get worker details
- `list_by_type(WorkerType)` - Filter by CPU/CUDA/Metal
- `list_by_platform(Platform)` - Filter by OS
- `get_latest(type, platform)` - Get latest version
- `health_check()` - Check catalog availability

### Endpoints
- `GET /workers` - List all
- `GET /workers/:id` - Get one
- `GET /workers/type/:type` - Filter by type
- `GET /workers/platform/:platform` - Filter by platform
- `GET /health` - Health check

### Testing
- Mock catalog server for tests
- Test all filter combinations
- Test error handling (catalog offline)
- WASM integration tests

### Acceptance Criteria
- [ ] All API methods work
- [ ] Filtering works correctly
- [ ] Uses artifacts-contract types
- [ ] Error handling robust
- [ ] Tests pass
- [ ] WASM exports work

---

## Part 5: Unified Marketplace API

**Time:** 2-3 days | **Complexity:** Medium

### Required Reading (1 hour)
- Review Parts 2-4 implementations
- Async aggregation patterns
- Result deduplication strategies

### Structure
```
src/marketplace.rs     # Unified Marketplace struct
```

### Marketplace Struct
```rust
#[wasm_bindgen]
pub struct Marketplace {
    huggingface: HuggingFaceClient,
    civitai: CivitAIClient,
    worker_catalog: WorkerCatalogClient,
}
```

### Unified Methods
- `search_all_models(query)` - Search HF + CivitAI
- `list_all_models()` - Aggregate from all sources
- `list_all_workers()` - From worker catalog
- `get_model(id, source)` - Get from specific source
- `get_worker(id)` - Get worker details

### Aggregation Logic
```rust
async fn search_all_models(&self, query: String) -> Result<Vec<Model>> {
    // Parallel requests
    let (hf_results, civitai_results) = tokio::join!(
        self.huggingface.search_models(query.clone(), None),
        self.civitai.search_models(query.clone(), None),
    );
    
    // Merge results
    let mut all_models = Vec::new();
    if let Ok(hf) = hf_results {
        all_models.extend(hf);
    }
    if let Ok(civitai) = civitai_results {
        all_models.extend(civitai);
    }
    
    // Deduplicate by ID
    // Sort by relevance
    // Apply filters
    
    Ok(all_models)
}
```

### Features
- Parallel API calls (faster)
- Result deduplication
- Unified sorting
- Unified filtering
- Error handling (partial failures OK)

### Testing
- Test with all clients
- Test partial failures
- Test deduplication
- Test sorting across sources
- WASM integration

### Acceptance Criteria
- [ ] Unified search works
- [ ] Parallel requests work
- [ ] Deduplication works
- [ ] Partial failures handled
- [ ] Tests pass
- [ ] WASM exports work

---

## Part 6: TypeScript Integration

**Time:** 1-2 days | **Complexity:** Low

### Tasks
1. **Generate TypeScript Types**
   - Run `wasm-pack build`
   - Verify `.d.ts` files
   - Check all exports

2. **Create NPM Package**
   - Use template from Part 1
   - Set version, description
   - Configure exports

3. **Write Usage Examples**
   ```typescript
   // Example 1: Search HuggingFace
   import { HuggingFaceClient } from '@rbee/marketplace-sdk';
   
   const client = new HuggingFaceClient();
   const models = await client.searchModels('llama', 10);
   
   // Example 2: Unified search
   import { Marketplace } from '@rbee/marketplace-sdk';
   
   const marketplace = new Marketplace();
   const allModels = await marketplace.searchAllModels('llama');
   ```

4. **Create API Documentation**
   - JSDoc comments
   - Usage examples
   - Error handling guide

5. **Publish to NPM**
   - Test locally first
   - Publish to NPM registry
   - Verify installation works

### Acceptance Criteria
- [ ] Types generated correctly
- [ ] Package published to NPM
- [ ] Examples work
- [ ] Documentation complete

---

## Part 7: Next.js Marketplace Integration

**Time:** 4-5 days | **Complexity:** High

### Structure
```
frontend/apps/marketplace/
├── app/
│   ├── page.tsx              # Home/search
│   ├── models/
│   │   ├── page.tsx          # Browse models
│   │   └── [id]/page.tsx     # Model details
│   └── workers/
│       ├── page.tsx          # Browse workers
│       └── [id]/page.tsx     # Worker details
├── components/
│   ├── SearchBar.tsx
│   ├── ModelCard.tsx
│   ├── WorkerCard.tsx
│   └── FilterSidebar.tsx
└── lib/
    └── marketplace.ts         # WASM loader
```

### Key Features
1. **WASM Loading**
   ```typescript
   // lib/marketplace.ts
   import init, { Marketplace } from '@rbee/marketplace-sdk';
   
   let marketplace: Marketplace | null = null;
   
   export async function getMarketplace() {
     if (!marketplace) {
       await init();
       marketplace = new Marketplace();
     }
     return marketplace;
   }
   ```

2. **Search Page**
   - Search bar with autocomplete
   - Filter sidebar (tags, size, source)
   - Sort dropdown
   - Infinite scroll pagination
   - Model cards with images

3. **Model Details Page**
   - Full model information
   - Download button
   - File list (GGUF variants)
   - Related models
   - Comments/reviews (future)

4. **Worker Browse Page**
   - List all workers
   - Filter by type/platform
   - Install button
   - Compatibility badges

### State Management
```typescript
// Use Zustand or React Context
interface MarketplaceState {
  searchQuery: string;
  filters: ModelFilters;
  models: Model[];
  loading: boolean;
  error: string | null;
}
```

### Deployment
- Build for Cloudflare Workers
- Configure environment variables
- Set up CDN for WASM files
- Test production build

### Acceptance Criteria
- [ ] Search works
- [ ] Filtering works
- [ ] Pagination works
- [ ] Model details load
- [ ] Worker listing works
- [ ] Deployed to Cloudflare
- [ ] Performance is good (<3s load)

---

## Part 8: Tauri Desktop Integration

**Time:** 3-4 days | **Complexity:** Medium-High

### Structure
```
bin/00_rbee_keeper/ui/
├── src/
│   ├── pages/
│   │   └── Marketplace.tsx    # New marketplace tab
│   ├── components/
│   │   └── marketplace/
│   │       ├── ModelBrowser.tsx
│   │       ├── WorkerBrowser.tsx
│   │       └── DownloadProgress.tsx
│   └── lib/
│       └── marketplace.ts      # WASM loader
```

### Integration Points
1. **Add Marketplace Tab**
   ```typescript
   // Add to navigation
   <Tab icon={ShoppingCart} label="Marketplace" />
   ```

2. **WASM Loading in Tauri**
   ```typescript
   // Vite config for WASM
   export default defineConfig({
     plugins: [
       react(),
       wasm(),
     ],
   });
   ```

3. **Connect to Tauri Commands**
   ```typescript
   // Download model
   async function downloadModel(modelId: string) {
     const marketplace = await getMarketplace();
     const model = await marketplace.getModel(modelId, 'HuggingFace');
     
     // Call Tauri command
     await invoke('model_download', {
       modelId: model.id,
       url: model.downloadUrl,
     });
   }
   ```

4. **Show Local Catalog**
   ```typescript
   // Show installed models
   const installed = await invoke('model_list');
   
   // Mark as installed in UI
   <ModelCard 
     model={model}
     installed={installed.includes(model.id)}
   />
   ```

### Features
- Browse HuggingFace/CivitAI models
- Browse worker catalog
- Download models (with progress)
- Install workers
- View installed artifacts
- "Already installed" badges

### Testing
- Test on Linux
- Test on macOS (if available)
- Test on Windows (if available)
- Test download flows
- Test install flows

### Acceptance Criteria
- [ ] Marketplace tab works
- [ ] Can browse models
- [ ] Can browse workers
- [ ] Download works
- [ ] Install works
- [ ] Shows installed items
- [ ] Cross-platform compatible

---

## Part 9: Testing & Quality

**Time:** 3-4 days | **Complexity:** Medium

### Test Categories

1. **Unit Tests (Rust)**
   - All client methods
   - Type conversions
   - Filtering logic
   - Sorting logic
   - URL builders
   - Error handling

2. **Integration Tests (Rust)**
   - Real API calls (with rate limiting)
   - Mock servers for offline tests
   - Error scenarios
   - Timeout handling

3. **WASM Tests**
   - Browser environment
   - Async operations
   - Memory management
   - Type conversions

4. **Frontend Tests**
   - Component tests (React Testing Library)
   - Integration tests (Playwright)
   - E2E tests (full user flows)

5. **Performance Tests**
   - WASM bundle size
   - Load time
   - API response time
   - Memory usage

### Test Coverage Goals
- Unit tests: >80%
- Integration tests: All critical paths
- E2E tests: All user flows

### Acceptance Criteria
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All WASM tests pass
- [ ] All E2E tests pass
- [ ] Coverage >80%
- [ ] No memory leaks
- [ ] Performance acceptable

---

## Part 10: Deployment & Publishing

**Time:** 2-3 days | **Complexity:** Low-Medium

### Tasks

1. **NPM Publishing**
   - Final version bump
   - Generate changelog
   - Publish to NPM
   - Create GitHub release
   - Tag version

2. **Next.js Deployment**
   - Build for production
   - Deploy to Cloudflare
   - Configure domain
   - Set up analytics
   - Monitor errors

3. **Tauri Releases**
   - Build for Linux
   - Build for macOS (if available)
   - Build for Windows (if available)
   - Create installers
   - Sign binaries
   - Upload to GitHub releases

4. **Documentation**
   - API reference (rustdoc)
   - Usage guide
   - Integration examples
   - Troubleshooting
   - FAQ

5. **CI/CD**
   - Automated builds
   - Automated tests
   - Automated publishing
   - Version bumping
   - Changelog generation

### Acceptance Criteria
- [ ] NPM package published
- [ ] Next.js app deployed
- [ ] Tauri releases created
- [ ] Documentation complete
- [ ] CI/CD working
- [ ] Monitoring set up

---

## Summary

**Total Implementation Time:** 26-35 days (5-7 weeks)

**Phases:**
1. ✅ Infrastructure (2-3 days)
2. ⏳ HuggingFace (3-4 days)
3. ⏳ CivitAI (3-4 days)
4. ⏳ Worker Catalog (2-3 days)
5. ⏳ Unified API (2-3 days)
6. ⏳ TypeScript (1-2 days)
7. ⏳ Next.js (4-5 days)
8. ⏳ Tauri (3-4 days)
9. ⏳ Testing (3-4 days)
10. ⏳ Deployment (2-3 days)

**Next Steps:**
1. Complete Part 1 (Infrastructure)
2. Start Part 2 (HuggingFace)
3. Follow each part's detailed guide
4. Test thoroughly at each step
5. Document as you go

---

**For detailed implementation of each part, refer to:**
- PART_01_INFRASTRUCTURE.md ✅
- PART_02_HUGGINGFACE.md ✅
- PART_03_CIVITAI.md (create when ready)
- PART_04_WORKER_CATALOG.md (create when ready)
- PART_05_UNIFIED_API.md (create when ready)
- PART_06_TYPESCRIPT.md (create when ready)
- PART_07_NEXTJS.md (create when ready)
- PART_08_TAURI.md (create when ready)
- PART_09_TESTING.md (create when ready)
- PART_10_DEPLOYMENT.md (create when ready)

Each detailed part document will follow the same structure as Parts 1-2 with:
- Required reading
- Detailed tasks with code examples
- Acceptance criteria
- Testing checklist
- Common issues & solutions
