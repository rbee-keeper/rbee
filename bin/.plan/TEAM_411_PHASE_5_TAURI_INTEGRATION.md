# TEAM-411: Phase 5 - Tauri Integration

**Created:** 2025-11-05  
**Team:** TEAM-411  
**Duration:** 2-3 days  
**Status:** ⏳ WAITING (blocked by TEAM-410)  
**Dependencies:** TEAM-410 complete (Next.js integration)

---

## 🎯 Mission

Integrate compatibility matrix into Tauri Keeper app: show compatibility in marketplace page, add worker selection based on compatibility, and implement install flow with compatibility checks.

---

## 🏗️ Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│ TAURI INTEGRATION: SDK → Tauri Commands → SPA GUI          │
└─────────────────────────────────────────────────────────────┘

1. marketplace-sdk (Rust Crate)
   ├─ compatibility.rs (core logic)
   ├─ NO WASM (native Rust in Tauri backend)
   └─ Used by: Tauri commands directly

2. Tauri Commands (Rust)
   ├─ File: bin/00_rbee_keeper/src/commands/compatibility.rs
   ├─ Functions:
   │  ├─ check_model_compatibility(model_id, worker_id) -> CompatibilityResult
   │  ├─ list_compatible_workers(model_id) -> Vec<Worker>
   │  └─ list_compatible_models(worker_id) -> Vec<Model>
   └─ Exposed to: Frontend via #[tauri::command]

3. SPA Frontend (React + TypeScript)
   ├─ File: bin/00_rbee_keeper/ui/src/api/compatibility.ts
   ├─ Import: import { invoke } from '@tauri-apps/api/tauri'
   ├─ Call: await invoke('check_model_compatibility', { modelId, workerId })
   └─ Components: WorkerSelector, CompatibilityBadge, etc.

4. User Flow
   ├─ User browses marketplace in Keeper
   ├─ Frontend calls Tauri commands
   ├─ Tauri backend uses marketplace-sdk (native Rust)
   ├─ Results returned to frontend
   └─ UI updates with compatibility info
```

**Key Differences from Next.js:**
- ❌ NO WASM (native Rust instead)
- ❌ NO marketplace-node wrapper (direct Rust crate usage)
- ✅ Tauri commands bridge Rust ↔ TypeScript
- ✅ Runtime compatibility checks (not pre-computed)
- ✅ Local-first (no network calls for compatibility)

---

## ✅ Checklist

### Task 5.1: Create Tauri Compatibility Commands
- [ ] Create `bin/00_rbee_keeper/src/commands/compatibility.rs`
- [ ] Add marketplace-sdk dependency to Keeper's Cargo.toml
- [ ] Implement Tauri commands:
  - [ ] `check_model_compatibility(model_id, worker_id)`
  - [ ] `list_compatible_workers(model_id)`
  - [ ] `list_compatible_models(worker_id)`
- [ ] Register commands in main.rs
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add Tauri compatibility commands"

**Implementation:**
```rust
// TEAM-411: Tauri compatibility commands
// bin/00_rbee_keeper/src/commands/compatibility.rs

use marketplace_sdk::{
    compatibility::check_compatibility,
    types::{ModelMetadata, Worker, CompatibilityResult},
};
use tauri::State;

#[tauri::command]
pub async fn check_model_compatibility(
    model_id: String,
    worker_id: String,
    state: State<'_, AppState>,
) -> Result<CompatibilityResult, String> {
    // Extract model metadata from HuggingFace or local cache
    let model_metadata = extract_model_metadata(&model_id)
        .await
        .map_err(|e| e.to_string())?;
    
    // Get worker from catalog
    let worker = get_worker_by_id(&worker_id, &state)
        .await
        .map_err(|e| e.to_string())?;
    
    // Check compatibility using marketplace-sdk
    let result = check_compatibility(&model_metadata, &worker);
    
    Ok(result)
}

#[tauri::command]
pub async fn list_compatible_workers(
    model_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<Worker>, String> {
    let model_metadata = extract_model_metadata(&model_id)
        .await
        .map_err(|e| e.to_string())?;
    
    let all_workers = list_all_workers(&state)
        .await
        .map_err(|e| e.to_string())?;
    
    let compatible = all_workers
        .into_iter()
        .filter(|worker| {
            check_compatibility(&model_metadata, worker).compatible
        })
        .collect();
    
    Ok(compatible)
}

#[tauri::command]
pub async fn list_compatible_models(
    worker_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<String>, String> {
    let worker = get_worker_by_id(&worker_id, &state)
        .await
        .map_err(|e| e.to_string())?;
    
    // Get models from cache or HuggingFace
    let models = fetch_top_models(100)
        .await
        .map_err(|e| e.to_string())?;
    
    let compatible = models
        .into_iter()
        .filter(|model| {
            check_compatibility(model, &worker).compatible
        })
        .map(|m| m.id)
        .collect();
    
    Ok(compatible)
}
```

**Register in main.rs:**
```rust
// bin/00_rbee_keeper/src/main.rs

mod commands;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // ... existing commands
            commands::compatibility::check_model_compatibility,
            commands::compatibility::list_compatible_workers,
            commands::compatibility::list_compatible_models,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

**Acceptance:**
- ✅ Tauri commands compile
- ✅ marketplace-sdk integrated
- ✅ Commands callable from frontend
- ✅ No WASM (native Rust)

---

### Task 5.2: Add Frontend API Wrapper
- [ ] Create `bin/00_rbee_keeper/ui/src/api/compatibility.ts`
- [ ] Wrap Tauri commands in TypeScript functions
- [ ] Add proper TypeScript types
- [ ] Export for use in components
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add frontend compatibility API wrapper"

**Implementation:**
```typescript
// TEAM-411: Frontend compatibility API
// bin/00_rbee_keeper/ui/src/api/compatibility.ts

import { invoke } from '@tauri-apps/api/tauri'

export interface CompatibilityResult {
  compatible: boolean
  confidence: 'high' | 'medium' | 'low' | 'none'
  reasons: string[]
  warnings: string[]
  recommendations: string[]
}

export interface Worker {
  id: string
  name: string
  worker_type: 'cpu' | 'cuda' | 'metal'
  platform: 'linux' | 'macos' | 'windows'
}

/**
 * Check if a model is compatible with a worker
 */
export async function checkModelCompatibility(
  modelId: string,
  workerId: string
): Promise<CompatibilityResult> {
  return invoke('check_model_compatibility', { modelId, workerId })
}

/**
 * List all workers compatible with a model
 */
export async function listCompatibleWorkers(
  modelId: string
): Promise<Worker[]> {
  return invoke('list_compatible_workers', { modelId })
}

/**
 * List all models compatible with a worker
 */
export async function listCompatibleModels(
  workerId: string
): Promise<string[]> {
  return invoke('list_compatible_models', { workerId })
}
```

**Acceptance:**
- ✅ TypeScript types match Rust structs
- ✅ Functions wrap Tauri invoke calls
- ✅ Exported for component use

---

### Task 5.3: Add Compatibility to Marketplace Page
- [ ] Open `bin/00_rbee_keeper/ui/src/pages/MarketplacePage.tsx`
- [ ] Import compatibility API functions
- [ ] Fetch compatible workers for each model
- [ ] Display compatibility badges
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add compatibility to Keeper marketplace page"

**Implementation:**
```tsx
// TEAM-411: Marketplace page with compatibility

import { useQuery } from '@tanstack/react-query'
import { listWorkerBinaries, getCompatibleWorkersForModel } from '@rbee/marketplace-node'
import { CompatibilityBadge } from '@rbee/ui/marketplace'

export function MarketplacePage() {
  const { data: workers } = useQuery({
    queryKey: ['workers'],
    queryFn: listWorkerBinaries,
  })
  
  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
  })
  
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Model Marketplace</h1>
      
      <div className="grid gap-4">
        {models?.map(model => (
          <ModelCard 
            key={model.id} 
            model={model}
            workers={workers}
          />
        ))}
      </div>
    </div>
  )
}

function ModelCard({ model, workers }) {
  const { data: compatible } = useQuery({
    queryKey: ['compatibility', model.id],
    queryFn: () => getCompatibleWorkersForModel(model.id),
    enabled: !!workers,
  })
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>{model.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          {model.description}
        </p>
        
        {/* TEAM-411: Compatibility badges */}
        {compatible && (
          <div className="flex gap-2 flex-wrap">
            {compatible.map(worker => (
              <CompatibilityBadge
                key={worker.id}
                result={{ compatible: true, ... }}
                workerName={worker.name}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
```

**Acceptance:**
- ✅ Compatibility badges show in Keeper
- ✅ Data fetched via TanStack Query
- ✅ Loading states handled

---

### Task 5.2: Add Worker Selection with Compatibility
- [ ] Create `bin/00_rbee_keeper/ui/src/components/WorkerSelector.tsx`
- [ ] Show only compatible workers for selected model
- [ ] Highlight recommended worker (best compatibility)
- [ ] Show incompatible workers (grayed out with reasons)
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add worker selector with compatibility"

**Implementation:**
```tsx
// TEAM-411: Worker selector component

import { useState } from 'react'
import { checkCompatibility } from '@rbee/marketplace-node'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'

interface WorkerSelectorProps {
  modelId: string
  workers: Worker[]
  onSelect: (workerId: string) => void
}

export function WorkerSelector({ modelId, workers, onSelect }: WorkerSelectorProps) {
  const [selected, setSelected] = useState<string>()
  
  const { data: compatibilityResults } = useQuery({
    queryKey: ['worker-compatibility', modelId],
    queryFn: async () => {
      const model = await extractModelMetadata(modelId)
      return Promise.all(
        workers.map(async worker => ({
          worker,
          compatibility: await checkCompatibility(model, worker),
        }))
      )
    },
  })
  
  if (!compatibilityResults) return <div>Loading...</div>
  
  // Sort: compatible first, then by confidence
  const sorted = [...compatibilityResults].sort((a, b) => {
    if (a.compatibility.compatible && !b.compatibility.compatible) return -1
    if (!a.compatibility.compatible && b.compatibility.compatible) return 1
    return 0
  })
  
  return (
    <div className="space-y-4">
      <h3 className="font-semibold">Select Worker</h3>
      
      <RadioGroup value={selected} onValueChange={setSelected}>
        {sorted.map(({ worker, compatibility }) => (
          <div
            key={worker.id}
            className={`flex items-center space-x-2 p-3 rounded border ${
              compatibility.compatible ? '' : 'opacity-50'
            }`}
          >
            <RadioGroupItem 
              value={worker.id} 
              disabled={!compatibility.compatible}
            />
            <Label className="flex-1 cursor-pointer">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">{worker.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {worker.worker_type} • {worker.platform}
                  </p>
                </div>
                <div className="flex gap-2">
                  {compatibility.compatible ? (
                    <Badge variant="success">Compatible</Badge>
                  ) : (
                    <Badge variant="destructive">Incompatible</Badge>
                  )}
                  {compatibility.confidence === 'high' && (
                    <Badge variant="outline">Recommended</Badge>
                  )}
                </div>
              </div>
              
              {/* Show reasons/warnings */}
              {compatibility.reasons.length > 0 && (
                <p className="text-xs text-muted-foreground mt-1">
                  {compatibility.reasons[0]}
                </p>
              )}
            </Label>
          </div>
        ))}
      </RadioGroup>
      
      <Button 
        onClick={() => selected && onSelect(selected)}
        disabled={!selected}
      >
        Continue
      </Button>
    </div>
  )
}
```

**Acceptance:**
- ✅ Shows only compatible workers first
- ✅ Incompatible workers grayed out
- ✅ Recommended worker highlighted
- ✅ Selection works

---

### Task 5.3: Add Compatibility Check to Install Flow
- [ ] Open `bin/00_rbee_keeper/src/handlers/protocol.rs`
- [ ] Add compatibility check before install
- [ ] Show error if incompatible
- [ ] Suggest compatible workers
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add compatibility check to install flow"

**Implementation:**
```rust
// TEAM-411: Protocol handler with compatibility check

use marketplace_sdk::{check_compatibility, extract_metadata_from_hf};

pub async fn handle_rbee_protocol(url: &str) -> Result<(), ProtocolError> {
    // Parse rbee://install/model/TinyLlama/TinyLlama-1.1B-Chat-v1.0?worker=cpu
    let (model_id, worker_id) = parse_protocol_url(url)?;
    
    // TEAM-411: Check compatibility before install
    let model_metadata = extract_metadata_from_hf(&model_id).await?;
    let worker = get_worker_by_id(&worker_id).await?;
    
    let compatibility = check_compatibility(&model_metadata, &worker);
    
    if !compatibility.compatible {
        return Err(ProtocolError::Incompatible {
            model: model_id,
            worker: worker_id,
            reasons: compatibility.reasons,
            suggestions: get_compatible_workers(&model_metadata).await?,
        });
    }
    
    // Proceed with install
    install_model_with_worker(model_id, worker_id).await?;
    
    Ok(())
}
```

**Acceptance:**
- ✅ Compatibility checked before install
- ✅ Error shown if incompatible
- ✅ Suggestions provided
- ✅ Install proceeds if compatible

---

### Task 5.4: Add Compatibility Warning Dialog
- [ ] Create `bin/00_rbee_keeper/ui/src/components/CompatibilityWarningDialog.tsx`
- [ ] Show when user tries to install incompatible model
- [ ] Display reasons for incompatibility
- [ ] Suggest compatible alternatives
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add compatibility warning dialog"

**Implementation:**
```tsx
// TEAM-411: Compatibility warning dialog

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog'

interface CompatibilityWarningDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  modelName: string
  workerName: string
  reasons: string[]
  suggestions: Worker[]
  onSelectAlternative: (workerId: string) => void
}

export function CompatibilityWarningDialog({
  open,
  onOpenChange,
  modelName,
  workerName,
  reasons,
  suggestions,
  onSelectAlternative,
}: CompatibilityWarningDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Incompatible Worker</AlertDialogTitle>
          <AlertDialogDescription>
            {modelName} is not compatible with {workerName}.
          </AlertDialogDescription>
        </AlertDialogHeader>
        
        <div className="space-y-4">
          <div>
            <p className="font-semibold text-sm mb-2">Reasons:</p>
            <ul className="list-disc list-inside text-sm space-y-1">
              {reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          </div>
          
          {suggestions.length > 0 && (
            <div>
              <p className="font-semibold text-sm mb-2">
                Try these compatible workers instead:
              </p>
              <div className="space-y-2">
                {suggestions.map(worker => (
                  <Button
                    key={worker.id}
                    variant="outline"
                    className="w-full justify-start"
                    onClick={() => {
                      onSelectAlternative(worker.id)
                      onOpenChange(false)
                    }}
                  >
                    {worker.name}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </div>
        
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
```

**Acceptance:**
- ✅ Dialog shows on incompatibility
- ✅ Reasons displayed clearly
- ✅ Suggestions actionable
- ✅ User can select alternative

---

### Task 5.5: Add Compatibility Indicator to Model Cards
- [ ] Update model card component
- [ ] Show compatibility status badge
- [ ] Show number of compatible workers
- [ ] Add quick filter by compatibility
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add compatibility indicators to model cards"

**Implementation:**
```tsx
// TEAM-411: Model card with compatibility indicator

function ModelCard({ model, installedWorkers }) {
  const { data: compatible } = useQuery({
    queryKey: ['model-compatibility', model.id],
    queryFn: async () => {
      const metadata = await extractModelMetadata(model.id)
      const results = await Promise.all(
        installedWorkers.map(w => checkCompatibility(metadata, w))
      )
      return results.filter(r => r.compatible).length
    },
  })
  
  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <CardTitle>{model.name}</CardTitle>
          
          {/* TEAM-411: Compatibility indicator */}
          {compatible !== undefined && (
            <Badge variant={compatible > 0 ? 'success' : 'secondary'}>
              {compatible > 0 
                ? `${compatible} compatible worker${compatible > 1 ? 's' : ''}`
                : 'No compatible workers'
              }
            </Badge>
          )}
        </div>
      </CardHeader>
      {/* ... rest of card */}
    </Card>
  )
}
```

**Acceptance:**
- ✅ Badge shows compatibility count
- ✅ Color-coded (green if compatible)
- ✅ Updates when workers change

---

### Task 5.6: Add Compatibility to Worker Management
- [ ] Open worker management page
- [ ] Show compatible models for each worker
- [ ] Add "Install Compatible Model" button
- [ ] Filter models by worker compatibility
- [ ] Add TEAM-411 signatures
- [ ] Commit: "TEAM-411: Add compatibility to worker management"

**Implementation:**
```tsx
// TEAM-411: Worker management with compatible models

function WorkerManagementPage() {
  const { data: workers } = useQuery({
    queryKey: ['installed-workers'],
    queryFn: listInstalledWorkers,
  })
  
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Installed Workers</h1>
      
      <div className="grid gap-4">
        {workers?.map(worker => (
          <WorkerCard key={worker.id} worker={worker} />
        ))}
      </div>
    </div>
  )
}

function WorkerCard({ worker }) {
  const { data: compatibleModels } = useQuery({
    queryKey: ['worker-compatible-models', worker.id],
    queryFn: async () => {
      const models = await fetchTopModels(50)
      return getCompatibleModelsForWorker(worker.id, models)
    },
  })
  
  return (
    <Card>
      <CardHeader>
        <CardTitle>{worker.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          {worker.worker_type} • {worker.platform}
        </p>
        
        {/* TEAM-411: Compatible models count */}
        <div className="flex items-center justify-between">
          <p className="text-sm">
            {compatibleModels?.length || 0} compatible models
          </p>
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => navigateToMarketplace({ worker: worker.id })}
          >
            Browse Compatible Models
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
```

**Acceptance:**
- ✅ Shows compatible model count
- ✅ Can browse compatible models
- ✅ Filter works

---

### Task 5.7: Write Integration Tests
- [ ] Create `bin/00_rbee_keeper/ui/tests/compatibility.test.tsx`
- [ ] Test WorkerSelector component
- [ ] Test CompatibilityWarningDialog
- [ ] Test install flow with compatibility check
- [ ] Run `pnpm test`
- [ ] Commit: "TEAM-411: Add Keeper compatibility tests"

**Test Setup:**
```tsx
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { WorkerSelector } from '../components/WorkerSelector'

describe('WorkerSelector', () => {
  it('shows compatible workers first', async () => {
    const workers = [
      { id: 'cpu', name: 'CPU Worker', compatible: true },
      { id: 'cuda', name: 'CUDA Worker', compatible: false },
    ]
    
    render(<WorkerSelector modelId="test" workers={workers} onSelect={vi.fn()} />)
    
    const items = await screen.findAllByRole('radio')
    expect(items[0]).not.toBeDisabled()
    expect(items[1]).toBeDisabled()
  })
  
  it('shows warning for incompatible selection', async () => {
    // Test warning dialog appears
  })
})
```

**Acceptance:**
- ✅ All tests pass
- ✅ Edge cases covered
- ✅ User flows tested

---

### Task 5.8: Update Documentation
- [ ] Update `bin/00_rbee_keeper/README.md`
- [ ] Document compatibility features
- [ ] Add screenshots
- [ ] Document protocol handler changes
- [ ] Commit: "TEAM-411: Update Keeper documentation"

**README Example:**
```markdown
## Compatibility Features

### Marketplace
- Model cards show number of compatible workers
- Filter models by installed workers
- Compatibility badges on each model

### Worker Selection
- Only compatible workers shown during install
- Incompatible workers grayed out with reasons
- Recommended worker highlighted

### Install Flow
- Automatic compatibility check before install
- Warning dialog if incompatible
- Suggestions for compatible alternatives
```

**Acceptance:**
- ✅ Documentation updated
- ✅ Screenshots added
- ✅ Usage clear

---

### Task 5.9: Verification
- [ ] Run `pnpm build` in Keeper UI - SUCCESS
- [ ] Run `pnpm test` - ALL PASS
- [ ] Run `cargo build --bin rbee-keeper` - SUCCESS
- [ ] Test in Keeper app (compatibility features work)
- [ ] Test protocol handler (rbee:// URLs)
- [ ] Review all changes for TEAM-411 signatures
- [ ] Create handoff document (max 2 pages)

**Handoff Document Contents:**
- Components created
- Protocol handler updated
- Test coverage
- Next team ready: TEAM-412

---

## 📁 Files Created/Modified

### New Files
- `bin/00_rbee_keeper/ui/src/components/WorkerSelector.tsx`
- `bin/00_rbee_keeper/ui/src/components/CompatibilityWarningDialog.tsx`
- `bin/00_rbee_keeper/ui/tests/compatibility.test.tsx`
- `TEAM_411_HANDOFF.md`

### Modified Files
- `bin/00_rbee_keeper/ui/src/pages/MarketplacePage.tsx` - Compatibility badges
- `bin/00_rbee_keeper/ui/src/pages/WorkerManagementPage.tsx` - Compatible models
- `bin/00_rbee_keeper/ui/src/components/ModelCard.tsx` - Compatibility indicator
- `bin/00_rbee_keeper/src/handlers/protocol.rs` - Compatibility check
- `bin/00_rbee_keeper/README.md` - Documentation

---

## ⚠️ Blockers & Dependencies

### Blocked By
- TEAM-410 (needs Next.js patterns and components)

### Blocks
- TEAM-412 (documentation and launch)

---

## 🎯 Success Criteria

- [ ] Compatibility badges in Keeper marketplace
- [ ] Worker selector with compatibility
- [ ] Install flow checks compatibility
- [ ] Warning dialog for incompatible installs
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Handoff document complete (≤2 pages)

---

## 📚 References

- Engineering Rules: `.windsurf/rules/engineering-rules.md`
- Keeper app: `bin/00_rbee_keeper/`
- marketplace-node: `frontend/packages/marketplace-node/`
- rbee-ui components: `frontend/packages/rbee-ui/src/marketplace/`

---

**TEAM-411 - Phase 5 Checklist v1.0**  
**Next Phase:** TEAM-412 (Documentation & Launch)
