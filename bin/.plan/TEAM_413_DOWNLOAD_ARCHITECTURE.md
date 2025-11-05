# TEAM-413: Download Tracking Architecture Analysis

**Date:** 2025-11-05  
**Status:** 📋 ARCHITECTURE DOCUMENTED  
**Next:** Implement proper wiring

---

## 🏗️ Existing Architecture (Discovered)

### 1. Job Submission Flow

**CLI → Backend:**
```
CLI Command
  ↓
handle_model() in handlers/model.rs
  ↓
submit_and_stream_job_to_hive()
  ↓
JobClient::submit_and_stream()
  ↓
POST /v1/jobs → Hive
  ↓
GET /jobs/{job_id}/stream (SSE)
  ↓
Print to stdout
```

**GUI → Backend:**
```
Tauri Command (e.g., marketplace_list_models)
  ↓
invoke() from @tauri-apps/api/core
  ↓
Tauri command handler
  ↓
Returns Result<T, String>
```

### 2. Progress Tracking (Narration)

**Backend → GUI:**
```
Backend emits narration event
  ↓
Tauri event system
  ↓
setupNarrationListener() in narrationListener.ts
  ↓
useNarrationStore.addEntry()
  ↓
NarrationPanel displays
```

**Key Files:**
- `ui/src/utils/narrationListener.ts` - Listens for narration events
- `ui/src/store/narrationStore.ts` - Zustand store (already exists!)
- `ui/src/components/NarrationPanel.tsx` - Displays narration

### 3. State Management

**Zustand Pattern:**
```typescript
// store/narrationStore.ts
export const useNarrationStore = create<NarrationState>()(
  persist(
    immer((set) => ({
      entries: [],
      addEntry: (event) => set((state) => {
        state.entries.unshift({ ...event, id: state.idCounter++ })
      }),
      // ...
    })),
    { name: 'narration-store' }
  )
)
```

---

## 🎯 Correct Download Tracking Architecture

### Option 1: Extend Narration System (RECOMMENDED)

**Why:** Already works, proven pattern, no new infrastructure

**Flow:**
```
User clicks "Install" in GUI
  ↓
Tauri command: model_download(model_id)
  ↓
Backend: submit_and_stream_job_to_hive()
  ↓
Backend emits narration events:
  - "📥 Downloading model..."
  - "📊 Progress: 45.2% (2.5 MB / 5.5 MB)"
  - "✅ Download complete"
  ↓
setupNarrationListener() receives events
  ↓
Parse progress from narration message
  ↓
Update downloadStore
  ↓
DownloadPanel displays
```

**Advantages:**
- ✅ Uses existing narration infrastructure
- ✅ No new Tauri events needed
- ✅ Works with existing job system
- ✅ Progress already emitted by backend
- ✅ Minimal changes required

**Implementation:**
1. Add Tauri command: `model_download(model_id)`
2. Parse progress from narration events
3. Update downloadStore from narration
4. Display in DownloadPanel

### Option 2: Dedicated Download Events (COMPLEX)

**Why:** More structured, but requires new infrastructure

**Flow:**
```
User clicks "Install"
  ↓
Tauri command: model_download(model_id)
  ↓
Backend emits dedicated events:
  - "download:start" { id, name, type }
  - "download:progress" { id, bytes, percentage, speed }
  - "download:complete" { id }
  ↓
useDownloadProgress() hook listens
  ↓
Update downloadStore
  ↓
DownloadPanel displays
```

**Disadvantages:**
- ❌ Requires new event types
- ❌ Requires new backend code
- ❌ Duplicates existing narration
- ❌ More complex to maintain

---

## 📋 Recommended Implementation (Option 1)

### Step 1: Add Tauri Command for Model Download

**File:** `bin/00_rbee_keeper/src/tauri_commands.rs`

```rust
/// Download a model from marketplace
/// TEAM-413: GUI model download with progress tracking
#[tauri::command]
#[specta::specta]
pub async fn model_download(
    hive_id: String,
    model_id: String,
) -> Result<String, String> {
    use crate::cli::ModelAction;
    use crate::handlers::model::handle_model;
    
    // Submit job and stream progress (via narration)
    handle_model(hive_id, ModelAction::Download { 
        model: Some(model_id) 
    })
    .await
    .map(|_| "Download started".to_string())
    .map_err(|e| e.to_string())
}
```

### Step 2: Enhance Narration Listener

**File:** `ui/src/utils/narrationListener.ts`

```typescript
// TEAM-413: Parse download progress from narration
const parseDownloadProgress = (message: string): DownloadProgress | null => {
  // Match: "📊 Progress: 45.2% (2.5 MB / 5.5 MB)"
  const progressMatch = message.match(/Progress: ([\d.]+)%.*\(([\d.]+) ([A-Z]+) \/ ([\d.]+) ([A-Z]+)\)/)
  if (progressMatch) {
    return {
      percentage: parseFloat(progressMatch[1]),
      bytesDownloaded: parseSize(progressMatch[2], progressMatch[3]),
      totalSize: parseSize(progressMatch[4], progressMatch[5]),
    }
  }
  return null
}

// In setupNarrationListener():
if (message.type === "NARRATION_EVENT") {
  const narrationEvent = message.payload as BackendNarrationEvent
  
  // TEAM-413: Check if this is a download progress event
  const progress = parseDownloadProgress(narrationEvent.human)
  if (progress && narrationEvent.job_id) {
    useDownloadStore.getState().updateDownload(narrationEvent.job_id, {
      bytesDownloaded: progress.bytesDownloaded,
      totalSize: progress.totalSize,
      percentage: progress.percentage,
    })
  }
  
  // ... rest of narration handling
}
```

### Step 3: Update Download Store

**File:** `ui/src/store/downloadStore.ts`

```typescript
// TEAM-413: Add job_id tracking
export interface Download {
  id: string // This is the job_id from backend
  name: string
  type: 'model' | 'worker'
  status: 'downloading' | 'complete' | 'failed' | 'cancelled'
  bytesDownloaded: number
  totalSize: number | null
  percentage: number | null
  speed: string | null
  eta: string | null
  error?: string
}

// Add helper to start download
startDownload: (jobId: string, name: string, type: 'model' | 'worker') => void
```

### Step 4: Wire Up Install Button

**File:** `ui/src/components/InstallButton.tsx` (NEW for Tauri context)

```typescript
// TEAM-413: Install button for Tauri (actually downloads)
import { invoke } from '@tauri-apps/api/core'
import { useDownloadStore } from '@/store/downloadStore'

export function TauriInstallButton({ modelId }: { modelId: string }) {
  const { startDownload } = useDownloadStore()
  
  const handleInstall = async () => {
    try {
      // Start download via Tauri command
      const jobId = await invoke<string>('model_download', {
        hiveId: 'localhost',
        modelId,
      })
      
      // Add to download tracker
      startDownload(jobId, modelId, 'model')
      
      // Navigate to show downloads
      // navigate('/') or show notification
    } catch (error) {
      console.error('Download failed:', error)
    }
  }
  
  return (
    <button onClick={handleInstall}>
      Download Model
    </button>
  )
}
```

---

## 🔄 Complete Flow

### User Journey

1. **User clicks "Install" in Keeper GUI**
   ```typescript
   <TauriInstallButton modelId="meta-llama/Llama-3.2-1B" />
   ```

2. **Frontend calls Tauri command**
   ```typescript
   const jobId = await invoke('model_download', { 
     hiveId: 'localhost', 
     modelId 
   })
   ```

3. **Backend starts download**
   ```rust
   handle_model(hive_id, ModelAction::Download { model })
     → submit_and_stream_job_to_hive()
     → JobClient streams SSE
   ```

4. **Backend emits narration events**
   ```
   📥 Downloading model: meta-llama/Llama-3.2-1B
   📊 Progress: 10.5% (500 MB / 4.7 GB)
   📊 Progress: 25.3% (1.2 GB / 4.7 GB)
   📊 Progress: 50.0% (2.4 GB / 4.7 GB)
   ✅ Download complete
   ```

5. **Frontend receives narration events**
   ```typescript
   setupNarrationListener() → parseDownloadProgress()
   ```

6. **Download store updates**
   ```typescript
   updateDownload(jobId, { percentage: 50.0, bytesDownloaded: 2.4GB })
   ```

7. **DownloadPanel shows progress**
   ```
   ┌─────────────────────────────────────────┐
   │ [🔄] meta-llama/Llama-3.2-1B   [Cancel] │
   │ model                                    │
   │ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ 50.0%             │
   │ 2.4 GB / 4.7 GB                         │
   │ 25 MB/s                    ETA: 1m 32s  │
   └─────────────────────────────────────────┘
   ```

---

## 📊 Comparison: Web vs Tauri

### Next.js (Marketplace Website)
```typescript
// Triggers rbee:// protocol
<button onClick={() => window.location.href = `rbee://model/${modelId}`}>
  Run with rbee
</button>
```

### Tauri (Keeper GUI)
```typescript
// Actually downloads the model
<button onClick={() => invoke('model_download', { hiveId, modelId })}>
  Download Model
</button>
```

**Key Difference:**
- **Web:** Opens external app (Keeper)
- **Tauri:** Downloads directly in app

---

## ✅ Implementation Checklist

### Phase 1: Backend (30 minutes)
- [ ] Add `model_download` Tauri command
- [ ] Add `worker_download` Tauri command
- [ ] Register commands in main.rs
- [ ] Test commands return job_id

### Phase 2: Frontend Store (30 minutes)
- [ ] Update Download interface (add job_id)
- [ ] Add `startDownload()` action
- [ ] Add `updateFromNarration()` helper
- [ ] Test store updates

### Phase 3: Narration Parser (45 minutes)
- [ ] Add `parseDownloadProgress()` function
- [ ] Integrate with setupNarrationListener()
- [ ] Update downloadStore from narration
- [ ] Test progress parsing

### Phase 4: UI Integration (45 minutes)
- [ ] Create TauriInstallButton component
- [ ] Use in marketplace pages
- [ ] Test download flow
- [ ] Test cancel/retry

**Total:** ~2.5 hours

---

## 🎯 Success Criteria

- [ ] User clicks "Install" in Keeper GUI
- [ ] Download appears in DownloadPanel
- [ ] Progress bar updates in real-time
- [ ] Speed and ETA calculated
- [ ] Cancel button works
- [ ] Retry on failure works
- [ ] Complete state shows checkmark
- [ ] No duplicate infrastructure

---

**TEAM-413 - Architecture Documented** ✅  
**Recommendation:** Use Option 1 (Extend Narration System)  
**Estimated Time:** 2.5 hours

**This is the correct way! 🐝📥**
