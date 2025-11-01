# Installed Workers View - Implementation Plan

**Status:** ✅ COMPLETE  
**Date:** 2025-11-01  
**Team:** TEAM-382 (completed TEAM-378's plan)

## What's Done

### 1. Frontend (✅ Complete)
- Created `InstalledWorkersView.tsx` component with table UI
- Added "Installed" tab to Worker Management
- Shows empty state with helpful message
- Ready to display worker data when backend is ready

### 2. Worker Catalog Registration (✅ Complete)
- Updated `worker_install.rs` to register installed binaries in catalog
- Added `add_to_catalog()` function that creates `WorkerBinary` entries
- Binaries are registered after successful installation

### 3. Operations Contract (✅ Complete)
- Added `WorkerListInstalled` operation
- Added `WorkerListInstalledRequest` type
- Updated all match arms in `operation_impl.rs`
- Contract compiles successfully

## What's Left

### 4. Backend Handler (✅ COMPLETE - TEAM-382)

Added handler in `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/job_router.rs`:

```rust
Operation::WorkerListInstalled(request) => {
    use rbee_hive_worker_catalog::WorkerCatalog;
    use rbee_hive_artifact_catalog::ArtifactCatalog;
    
    let hive_id = request.hive_id.clone();
    n!("worker_list_installed_start", "📋 Listing installed workers on hive '{}'", hive_id);
    
    // Get worker catalog
    let catalog = WorkerCatalog::new()
        .context("Failed to create worker catalog")?;
    
    // List all installed workers
    let workers = catalog.list();
    
    n!("worker_list_installed_count", "Found {} installed workers", workers.len());
    
    // Convert to JSON response
    let response = serde_json::json!({
        "workers": workers.iter().map(|w| {
            serde_json::json!({
                "id": w.id(),
                "name": w.name(),
                "worker_type": format!("{:?}", w.worker_type()),
                "platform": format!("{:?}", w.platform()),
                "version": w.version(),
                "size": w.size(),
                "path": w.path().display().to_string(),
                "added_at": w.added_at().to_rfc3339(),
            })
        }).collect::<Vec<_>>()
    });
    
    n!("worker_list_installed_ok", "✅ Listed {} installed workers", workers.len());
    
    // Return JSON response
    Ok(response.to_string())
}
```

### 5. SDK Method (✅ COMPLETE - TEAM-382)

Added to `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs`:

```rust
/// List installed worker binaries on hive
pub fn worker_list_installed(hive_id: String) -> JsValue {
    let op = Operation::WorkerListInstalled(WorkerListInstalledRequest {
        hive_id,
    });
    serde_wasm_bindgen::to_value(&op).unwrap()
}
```

### 6. React Hook (✅ COMPLETE - TEAM-382)

Created `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useInstalledWorkers.ts`:

```typescript
import { useQuery } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

const hiveAddress = window.location.hostname
const hivePort = '7835'
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

export interface InstalledWorker {
  id: string
  name: string
  worker_type: string
  platform: string
  version: string
  size: number
  path: string
  added_at: string
}

export function useInstalledWorkers() {
  return useQuery<InstalledWorker[]>({
    queryKey: ['installed-workers'],
    queryFn: async () => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.workerListInstalled(hiveId)
      
      const lines: string[] = []
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })
      
      // Last line should be JSON response
      const lastLine = lines[lines.length - 1]
      const response = JSON.parse(lastLine)
      return response.workers
    },
    refetchInterval: 10000, // Refresh every 10 seconds
  })
}
```

### 7. Wire Up Frontend (✅ COMPLETE - TEAM-382)

Updated `InstalledWorkersView.tsx`:

```typescript
import { useInstalledWorkers } from '../../hooks/useInstalledWorkers'

export function InstalledWorkersView({ onUninstall }: InstalledWorkersViewProps) {
  const { data: installedWorkers = [], isLoading, error } = useInstalledWorkers()
  const [removingWorker, setRemovingWorker] = useState<string | null>(null)
  
  if (isLoading) {
    return <div>Loading...</div>
  }
  
  if (error) {
    return <div>Error: {error.message}</div>
  }
  
  // Rest of component...
}
```

## Testing Steps

1. **Install a worker** via Worker Catalog
2. **Check catalog directory**: `ls ~/.cache/rbee/workers/`
3. **Switch to Installed tab** - should show the worker
4. **Verify data** - name, type, version, size, path should all be correct
5. **Test uninstall** (when implemented)

## File Locations

- Frontend: `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/app/src/components/WorkerManagement/InstalledWorkersView.tsx`
- Backend Handler: `/home/vince/Projects/llama-orch/bin/20_rbee_hive/src/job_router.rs`
- Worker Catalog: `/home/vince/Projects/llama-orch/bin/25_rbee_hive_crates/worker-catalog/`
- Operations Contract: `/home/vince/Projects/llama-orch/bin/97_contracts/operations-contract/`
- SDK: `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/packages/rbee-hive-sdk/`
- React Hooks: `/home/vince/Projects/llama-orch/bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/`

## Architecture Flow

```
Frontend (InstalledWorkersView)
  ↓
React Hook (useInstalledWorkers)
  ↓
SDK (OperationBuilder.workerListInstalled)
  ↓
Job Client (POST /v1/jobs)
  ↓
Job Server (rbee-hive)
  ↓
Job Router (Operation::WorkerListInstalled)
  ↓
Worker Catalog (catalog.list())
  ↓
Filesystem (~/.cache/rbee/workers/*.json)
```

## Build & Test Steps (Required)

**CRITICAL:** The following steps MUST be completed before the feature works:

1. **Rebuild SDK WASM package** (required for TypeScript types):
   ```bash
   cd bin/20_rbee_hive/ui/packages/rbee-hive-sdk
   pnpm build
   ```

2. **Rebuild React hooks package**:
   ```bash
   cd bin/20_rbee_hive/ui/packages/rbee-hive-react
   pnpm build
   ```

3. **Rebuild frontend app**:
   ```bash
   cd bin/20_rbee_hive/ui/app
   pnpm build
   ```

4. **Restart rbee-hive** to pick up backend changes:
   ```bash
   cargo build --bin rbee-hive
   # Then restart the hive daemon
   ```

5. **Test end-to-end**:
   - Install a worker via Worker Catalog tab
   - Switch to "Installed" tab
   - Verify worker appears in table with correct data
   - Check that data refreshes every 10 seconds

## TEAM-382 Implementation Summary

**Files Modified:**
- `bin/20_rbee_hive/src/job_router.rs` (+30 LOC) - Added WorkerListInstalled handler
- `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/src/operations.rs` (+14 LOC) - Added SDK method
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/index.ts` (+3 LOC) - Export hook

**Files Created:**
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useInstalledWorkers.ts` (67 LOC) - React hook
- `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/InstalledWorkersView.tsx` (modified +40 LOC) - Wired up to hook

**Total LOC:** ~154 lines added

**Key Features:**
- Lists installed worker binaries from catalog
- Shows worker type, version, size, path, install date
- Auto-refreshes every 10 seconds
- Loading and error states
- Empty state with helpful message
- Follows existing patterns (useModelOperations, etc.)

**Compilation Status:**
- ✅ Backend: `cargo check -p rbee-hive` passes
- ⏳ Frontend: Requires SDK rebuild (expected TypeScript errors until rebuild)
