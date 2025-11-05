# TEAM-408: Phase 2 - Worker Catalog in marketplace-sdk

**Created:** 2025-11-05  
**Team:** TEAM-408  
**Duration:** 2-3 days  
**Status:** ⏳ WAITING (blocked by TEAM-407)  
**Dependencies:** TEAM-407 complete (clean types)

---

## 🎯 Mission

Implement worker catalog client in marketplace-sdk (Rust + WASM) and expose it to marketplace-node for Next.js/Tauri consumption.

---

## ✅ Checklist

### Task 2.1: Create Worker Catalog Client (Rust)
- [ ] Create `bin/99_shared_crates/marketplace-sdk/src/worker_catalog.rs`
- [ ] Implement `WorkerCatalogClient` struct
- [ ] Add method: `list_workers() -> Vec<WorkerBinary>`
- [ ] Add method: `get_worker(id: &str) -> Option<WorkerBinary>`
- [ ] Add method: `filter_workers(filter: WorkerFilter) -> Vec<WorkerBinary>`
- [ ] Fetch data from Hono catalog API (http://localhost:3000/workers)
- [ ] Add error handling
- [ ] Add TEAM-408 signatures
- [ ] Commit: "TEAM-408: Add WorkerCatalogClient"

**WorkerFilter struct:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerFilter {
    pub worker_type: Option<WorkerType>,
    pub platform: Option<Platform>,
    pub architecture: Option<String>,
    pub min_context_length: Option<u32>,
}
```

**Acceptance:**
- ✅ WorkerCatalogClient compiles
- ✅ Can fetch workers from Hono API
- ✅ Can filter by type/platform
- ✅ Error handling for network failures

---

### Task 2.2: Add WASM Bindings
- [ ] Create `bin/99_shared_crates/marketplace-sdk/src/wasm_worker.rs`
- [ ] Add `#[wasm_bindgen]` functions:
  - [ ] `list_workers() -> JsValue`
  - [ ] `get_worker(id: String) -> JsValue`
  - [ ] `filter_workers(filter: JsValue) -> JsValue`
- [ ] Convert Rust types to JsValue
- [ ] Add error handling for WASM
- [ ] Export from lib.rs
- [ ] Commit: "TEAM-408: Add WASM worker bindings"

**Example:**
```rust
#[wasm_bindgen]
pub async fn list_workers() -> Result<JsValue, JsValue> {
    let client = WorkerCatalogClient::new();
    let workers = client.list_workers().await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    serde_wasm_bindgen::to_value(&workers)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

**Acceptance:**
- ✅ WASM functions compile
- ✅ Can call from JavaScript
- ✅ Types serialize correctly

---

### Task 2.3: Build WASM Package
- [ ] Run `wasm-pack build --target bundler` in marketplace-sdk
- [ ] Verify `pkg/` directory created
- [ ] Check TypeScript types generated
- [ ] Verify worker functions in .d.ts file
- [ ] Test WASM module loads
- [ ] Commit: "TEAM-408: Build WASM package"

**Verification:**
```bash
cd bin/99_shared_crates/marketplace-sdk
wasm-pack build --target bundler
ls pkg/
cat pkg/marketplace_sdk.d.ts | grep -i worker
```

**Acceptance:**
- ✅ WASM builds without errors
- ✅ TypeScript types include worker functions
- ✅ Package size reasonable (<500KB)

---

### Task 2.4: Update marketplace-node Wrapper
- [ ] Open `frontend/packages/marketplace-node/src/index.ts`
- [ ] Remove TODO from `listWorkerBinaries()`
- [ ] Call WASM `list_workers()` function
- [ ] Add error handling
- [ ] Add TypeScript types
- [ ] Test function returns data
- [ ] Commit: "TEAM-408: Implement listWorkerBinaries()"

**Implementation:**
```typescript
export async function listWorkerBinaries(
  options: SearchOptions = {}
): Promise<Worker[]> {
  const sdk = await getSDK()
  
  try {
    const workers = await sdk.list_workers()
    return workers as Worker[]
  } catch (error) {
    console.error('Failed to list workers:', error)
    return []
  }
}
```

**Acceptance:**
- ✅ Function calls WASM successfully
- ✅ Returns array of workers
- ✅ Error handling works
- ✅ TypeScript types correct

---

### Task 2.5: Add Worker Filtering Functions
- [ ] Add `filterWorkersByType(type: WorkerType)` to marketplace-node
- [ ] Add `filterWorkersByPlatform(platform: Platform)` to marketplace-node
- [ ] Add `getCompatibleWorkers(model: ModelMetadata)` to marketplace-node
- [ ] Call WASM filter functions
- [ ] Add tests
- [ ] Commit: "TEAM-408: Add worker filtering functions"

**Example:**
```typescript
export async function filterWorkersByType(
  type: WorkerType
): Promise<Worker[]> {
  const sdk = await getSDK()
  const filter = { worker_type: type }
  return await sdk.filter_workers(filter) as Worker[]
}

export async function getCompatibleWorkers(
  model: ModelMetadata
): Promise<Worker[]> {
  const workers = await listWorkerBinaries()
  
  return workers.filter(worker => {
    // Check architecture compatibility
    if (!worker.supported_architectures.includes(model.architecture)) {
      return false
    }
    
    // Check format compatibility
    if (!worker.supported_formats.includes(model.format)) {
      return false
    }
    
    // Check context length
    if (model.max_context_length > worker.max_context_length) {
      return false
    }
    
    return true
  })
}
```

**Acceptance:**
- ✅ Can filter by type
- ✅ Can filter by platform
- ✅ Can get compatible workers for model
- ✅ All functions tested

---

### Task 2.6: Write Unit Tests
- [ ] Create `bin/99_shared_crates/marketplace-sdk/tests/worker_catalog_tests.rs`
- [ ] Test `WorkerCatalogClient::list_workers()`
- [ ] Test `WorkerCatalogClient::get_worker()`
- [ ] Test `WorkerCatalogClient::filter_workers()`
- [ ] Test error handling
- [ ] Mock Hono API responses
- [ ] Run `cargo test -p marketplace-sdk`
- [ ] Commit: "TEAM-408: Add worker catalog tests"

**Test Cases:**
```rust
#[tokio::test]
async fn test_list_workers() {
    let client = WorkerCatalogClient::new();
    let workers = client.list_workers().await.unwrap();
    assert!(!workers.is_empty());
}

#[tokio::test]
async fn test_filter_by_type() {
    let client = WorkerCatalogClient::new();
    let filter = WorkerFilter {
        worker_type: Some(WorkerType::Cuda),
        ..Default::default()
    };
    let workers = client.filter_workers(filter).await.unwrap();
    assert!(workers.iter().all(|w| w.worker_type == WorkerType::Cuda));
}
```

**Acceptance:**
- ✅ All tests pass
- ✅ Edge cases covered
- ✅ Error cases tested

---

### Task 2.7: Write Integration Tests
- [ ] Create `frontend/packages/marketplace-node/tests/workers.test.ts`
- [ ] Test `listWorkerBinaries()`
- [ ] Test `filterWorkersByType()`
- [ ] Test `filterWorkersByPlatform()`
- [ ] Test `getCompatibleWorkers()`
- [ ] Run tests with `pnpm test`
- [ ] Commit: "TEAM-408: Add marketplace-node worker tests"

**Test Setup:**
```typescript
import { describe, it, expect } from 'vitest'
import { listWorkerBinaries, filterWorkersByType } from '../src/index'

describe('Worker Catalog', () => {
  it('should list workers', async () => {
    const workers = await listWorkerBinaries()
    expect(workers).toBeInstanceOf(Array)
    expect(workers.length).toBeGreaterThan(0)
  })
  
  it('should filter by type', async () => {
    const workers = await filterWorkersByType('cuda')
    expect(workers.every(w => w.worker_type === 'cuda')).toBe(true)
  })
})
```

**Acceptance:**
- ✅ All tests pass
- ✅ WASM loads correctly in tests
- ✅ Functions return expected data

---

### Task 2.8: Update Documentation
- [ ] Update `bin/99_shared_crates/marketplace-sdk/README.md`
- [ ] Add worker catalog usage examples
- [ ] Document WASM functions
- [ ] Add API reference
- [ ] Update `frontend/packages/marketplace-node/README.md`
- [ ] Add worker functions to examples
- [ ] Commit: "TEAM-408: Update documentation"

**README Example:**
```markdown
## Worker Catalog

### List Workers
\`\`\`typescript
import { listWorkerBinaries } from '@rbee/marketplace-node'

const workers = await listWorkerBinaries()
console.log(workers)
\`\`\`

### Filter Workers
\`\`\`typescript
import { filterWorkersByType } from '@rbee/marketplace-node'

const cudaWorkers = await filterWorkersByType('cuda')
\`\`\`

### Get Compatible Workers
\`\`\`typescript
import { getCompatibleWorkers } from '@rbee/marketplace-node'

const model = { architecture: 'llama', format: 'safetensors', ... }
const workers = await getCompatibleWorkers(model)
\`\`\`
```

**Acceptance:**
- ✅ README updated with examples
- ✅ API documented
- ✅ Usage clear

---

### Task 2.9: Verification
- [ ] Run `cargo check --workspace` - ZERO errors
- [ ] Run `cargo test -p marketplace-sdk` - ALL PASS
- [ ] Run `pnpm test` in marketplace-node - ALL PASS
- [ ] Build WASM: `wasm-pack build --target bundler`
- [ ] Verify TypeScript types generated
- [ ] Test in Next.js (import and call function)
- [ ] Review all changes for TEAM-408 signatures
- [ ] Create handoff document (max 2 pages)

**Handoff Document Contents:**
- What was implemented (worker catalog client)
- WASM functions exposed
- marketplace-node functions added
- Test coverage
- Next team ready: TEAM-409

---

## 📁 Files Created/Modified

### New Files
- `bin/99_shared_crates/marketplace-sdk/src/worker_catalog.rs`
- `bin/99_shared_crates/marketplace-sdk/src/wasm_worker.rs`
- `bin/99_shared_crates/marketplace-sdk/tests/worker_catalog_tests.rs`
- `frontend/packages/marketplace-node/tests/workers.test.ts`
- `TEAM_408_HANDOFF.md`

### Modified Files
- `bin/99_shared_crates/marketplace-sdk/src/lib.rs` - Re-exports
- `bin/99_shared_crates/marketplace-sdk/Cargo.toml` - Dependencies
- `frontend/packages/marketplace-node/src/index.ts` - Worker functions
- `bin/99_shared_crates/marketplace-sdk/README.md` - Documentation
- `frontend/packages/marketplace-node/README.md` - Documentation

---

## ⚠️ Blockers & Dependencies

### Blocked By
- TEAM-407 (needs clean types)

### Blocks
- TEAM-409 (needs worker catalog to implement compatibility checks)

---

## 🎯 Success Criteria

- [ ] WorkerCatalogClient implemented in Rust
- [ ] WASM bindings working
- [ ] marketplace-node functions implemented
- [ ] All tests passing (Rust + TypeScript)
- [ ] Documentation complete
- [ ] Can list/filter workers from Next.js
- [ ] Handoff document complete (≤2 pages)

---

## 📚 References

- Engineering Rules: `.windsurf/rules/engineering-rules.md`
- Hono catalog: `bin/80-hono-worker-catalog/src/data.ts`
- marketplace-sdk: `bin/99_shared_crates/marketplace-sdk/`
- marketplace-node: `frontend/packages/marketplace-node/`

---

**TEAM-408 - Phase 2 Checklist v1.0**  
**Next Phase:** TEAM-409 (Compatibility Matrix Data Layer)
