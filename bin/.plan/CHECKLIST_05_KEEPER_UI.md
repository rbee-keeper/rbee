# Checklist 05: Keeper UI - Add Marketplace Tab

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** Checklist 04 (Protocol Handler)  
**TEAM-400:** ✅ RULE ZERO - Keeper UI EXISTS, just add marketplace page

---

## 🎯 Goal

Add marketplace page to EXISTING Keeper UI. Users can browse models/workers and install them directly from Keeper.

**TEAM-400:** Keeper UI exists at `bin/00_rbee_keeper/ui/` with React Router + Zustand. Just add `/marketplace` route.

---

## 📦 Phase 1: Setup (Day 1, Morning)

**TEAM-400:** Keeper UI already has routing, Zustand, and rbee-ui components.

### 1.1 Verify Current Structure

- [ ] Check existing structure:
  ```
  bin/00_rbee_keeper/ui/src/
  ├── App.tsx (React Router with routes)
  ├── pages/
  │   ├── ServicesPage.tsx (home)
  │   ├── QueenPage.tsx
  │   ├── HivePage.tsx
  │   ├── SettingsPage.tsx
  │   └── HelpPage.tsx
  ├── store/
  │   ├── commandStore.ts
  │   ├── narrationStore.ts
  │   ├── queenQueries.ts
  │   └── hiveQueries.ts
  └── components/
      └── Shell.tsx (sidebar + content)
  ```
- [ ] Verify dependencies in `package.json`:
  - `@rbee/ui` (already has this!)
  - `react-router-dom` (already has this!)
  - `zustand` (already has this!)

### 1.2 Add Marketplace SDK Dependency

- [ ] Update `bin/00_rbee_keeper/ui/package.json`:
  ```json
  {
    "dependencies": {
      "@rbee/ui": "workspace:*",
      "@rbee/marketplace-sdk": "workspace:*",
      "@rbee/iframe-bridge": "workspace:*",
      "@rbee/narration-client": "workspace:*",
      // ... existing deps
    }
  }
  ```
- [ ] Install: `cd bin/00_rbee_keeper/ui && pnpm install`
- [ ] Verify: `pnpm build` succeeds

---

## 📄 Phase 2: Marketplace Page (Days 1-2)

**TEAM-400:** Create new page, use marketplace components from rbee-ui.

### 2.1 Create Marketplace Page

- [ ] Create: `bin/00_rbee_keeper/ui/src/pages/MarketplacePage.tsx`
  ```tsx
  // TEAM-400: Marketplace page for Keeper
  'use client'
  
  import { useState, useEffect } from 'react'
  import { ModelsPage } from '@rbee/ui/marketplace/pages/ModelsPage'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  
  export default function MarketplacePage() {
    const [models, setModels] = useState([])
    const [loading, setLoading] = useState(true)
    
    useEffect(() => {
      // TEAM-400: Fetch models from HuggingFace
      async function fetchModels() {
        try {
          const client = new HuggingFaceClient()
          const data = await client.list_models({ limit: 50 })
          setModels(data)
        } catch (error) {
          console.error('Failed to fetch models:', error)
        } finally {
          setLoading(false)
        }
      }
      
      fetchModels()
    }, [])
    
    if (loading) {
      return <div className="p-8">Loading marketplace...</div>
    }
    
    // TEAM-400: Use ModelsPage from rbee-ui
    return (
      <ModelsPage
        seo={{ title: 'Marketplace', description: 'Browse models' }}
        template={{
          title: 'Marketplace',
          description: 'Browse and install AI models',
          models: models.map(m => ({
            model: m,
            onAction: (modelId) => handleInstall(modelId),
          })),
          filters: { search: '', sort: 'popular' }
        }}
      />
    )
  }
  
  // TEAM-400: Handle model installation
  async function handleInstall(modelId: string) {
    console.log('Installing model:', modelId)
    // TODO: Trigger download via Tauri command
    // await invoke('download_model', { modelId })
  }
  ```

### 2.2 Add Route to App

- [ ] Update `bin/00_rbee_keeper/ui/src/App.tsx`:
  ```tsx
  import MarketplacePage from './pages/MarketplacePage'
  
  function App() {
    return (
      <BrowserRouter>
        <SidebarProvider>
          <Shell>
            <Routes>
              <Route path="/" element={<KeeperPage />} />
              <Route path="/queen" element={<QueenPage />} />
              <Route path="/hive/:hiveId" element={<HivePage />} />
              <Route path="/marketplace" element={<MarketplacePage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/help" element={<HelpPage />} />
            </Routes>
          </Shell>
        </SidebarProvider>
      </BrowserRouter>
    )
  }
  ```

### 2.3 Add Sidebar Link

- [ ] Update `bin/00_rbee_keeper/ui/src/components/Shell.tsx`:
  ```tsx
  // TEAM-400: Add marketplace link to sidebar
  const navItems = [
    { path: '/', label: 'Services', icon: <HomeIcon /> },
    { path: '/queen', label: 'Queen', icon: <CrownIcon /> },
    { path: '/marketplace', label: 'Marketplace', icon: <ShoppingCartIcon /> },
    { path: '/settings', label: 'Settings', icon: <SettingsIcon /> },
    { path: '/help', label: 'Help', icon: <HelpIcon /> },
  ]
  ```

---

## 🔧 Phase 3: Install Functionality (Day 3)

**TEAM-400:** Connect marketplace to Tauri commands for downloading models.

### 3.1 Create Tauri Command for Model Download

- [ ] Create: `bin/00_rbee_keeper/src/handlers/marketplace.rs`
  ```rust
  // TEAM-400: Marketplace commands for Keeper
  
  use anyhow::Result;
  use job_client::JobClient;
  use operations_contract::Operation;
  use tauri::{AppHandle, Emitter};
  
  /// Download model from marketplace
  #[tauri::command]
  pub async fn download_model(
      app: AppHandle,
      model_id: String,
  ) -> Result<String, String> {
      println!("📥 Downloading model: {}", model_id);
      
      // TEAM-400: Submit download job to hive
      let client = JobClient::new("http://localhost:9200");
      
      let operation = Operation::ModelDownload {
          model_id: model_id.clone(),
          source: "huggingface".to_string(),
      };
      
      // TEAM-400: Stream progress to frontend
      client.submit_and_stream(operation, |line| {
          // Emit progress event to frontend
          let _ = app.emit("download-progress", line);
          Ok(())
      }).await
      .map_err(|e| e.to_string())?;
      
      Ok(format!("Model {} downloaded successfully", model_id))
  }
  
  /// Spawn worker with model
  #[tauri::command]
  pub async fn spawn_worker_with_model(
      app: AppHandle,
      model_id: String,
      worker_type: String,
  ) -> Result<String, String> {
      println!("🐝 Spawning worker for model: {}", model_id);
      
      let client = JobClient::new("http://localhost:9200");
      
      let operation = Operation::WorkerSpawn {
          worker_type,
          model: Some(model_id.clone()),
          port: None,
      };
      
      client.submit_and_stream(operation, |line| {
          let _ = app.emit("spawn-progress", line);
          Ok(())
      }).await
      .map_err(|e| e.to_string())?;
      
      Ok(format!("Worker spawned for model {}", model_id))
  }
  ```

### 3.2 Register Commands

- [ ] Update `bin/00_rbee_keeper/src/main.rs`:
  ```rust
  mod handlers;
  use handlers::marketplace::{download_model, spawn_worker_with_model};
  
  fn main() {
      tauri::Builder::default()
          // ... existing setup ...
          .invoke_handler(tauri::generate_handler![
              // ... existing commands ...
              download_model,
              spawn_worker_with_model,
          ])
          .run(tauri::generate_context!())
          .expect("error while running tauri application");
  }
  ```

### 3.3 Use Commands in Frontend

- [ ] Update `MarketplacePage.tsx`:
  ```tsx
  import { invoke } from '@tauri-apps/api/core'
  import { listen } from '@tauri-apps/api/event'
  
  async function handleInstall(modelId: string) {
    try {
      // TEAM-400: Listen for progress
      const unlisten = await listen<string>('download-progress', (event) => {
        console.log('Download progress:', event.payload)
        // TODO: Show progress in UI
      })
      
      // TEAM-400: Download model
      await invoke('download_model', { modelId })
      
      // TEAM-400: Spawn worker
      await invoke('spawn_worker_with_model', {
        modelId,
        workerType: 'cpu-llm'
      })
      
      unlisten()
      
      alert(`Model ${modelId} is ready!`)
    } catch (error) {
      console.error('Installation failed:', error)
      alert(`Failed to install: ${error}`)
    }
  }
  ```

---

## 🎨 Phase 4: UI Enhancements (Day 4)

### 4.1 Add Progress Indicator

- [ ] Create: `bin/00_rbee_keeper/ui/src/components/DownloadProgress.tsx`
  ```tsx
  // TEAM-400: Show download progress
  import { useEffect, useState } from 'react'
  import { listen } from '@tauri-apps/api/event'
  import { Progress } from '@rbee/ui/atoms/Progress'
  
  export function DownloadProgress({ modelId }: { modelId: string }) {
    const [progress, setProgress] = useState(0)
    const [status, setStatus] = useState('')
    
    useEffect(() => {
      const unlisten = listen<string>('download-progress', (event) => {
        setStatus(event.payload)
        // Parse progress from line (if available)
        const match = event.payload.match(/(\d+)%/)
        if (match) {
          setProgress(parseInt(match[1]))
        }
      })
      
      return () => {
        unlisten.then(fn => fn())
      }
    }, [])
    
    return (
      <div className="space-y-2">
        <Progress value={progress} />
        <p className="text-sm text-muted-foreground">{status}</p>
      </div>
    )
  }
  ```

### 4.2 Add Model Detail View

- [ ] Create: `bin/00_rbee_keeper/ui/src/pages/ModelDetailPage.tsx`
  ```tsx
  // TEAM-400: Model detail page in Keeper
  import { useParams } from 'react-router-dom'
  import { ModelDetailPage } from '@rbee/ui/marketplace/pages/ModelDetailPage'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  
  export default function KeeperModelDetailPage() {
    const { modelId } = useParams()
    const [model, setModel] = useState(null)
    
    useEffect(() => {
      async function fetchModel() {
        const client = new HuggingFaceClient()
        const data = await client.get_model(modelId!)
        setModel(data)
      }
      fetchModel()
    }, [modelId])
    
    if (!model) return <div>Loading...</div>
    
    return (
      <ModelDetailPage
        seo={{ title: model.name, description: model.description }}
        template={{
          model,
          installButton: <InstallButton modelId={model.id} />,
        }}
      />
    )
  }
  ```

- [ ] Add route: `/marketplace/models/:modelId`

### 4.3 Add Search and Filters

- [ ] Add search state to `MarketplacePage`:
  ```tsx
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState<'popular' | 'recent'>('popular')
  
  const filteredModels = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase())
  )
  ```

---

## 🔗 Phase 5: Protocol Integration (Day 5)

**TEAM-400:** Connect protocol handler (from CHECKLIST_04) to marketplace page.

### 5.1 Listen for Protocol Events

- [ ] Update `MarketplacePage.tsx`:
  ```tsx
  import { useEffect } from 'react'
  import { listen } from '@tauri-apps/api/event'
  import { useNavigate } from 'react-router-dom'
  
  export default function MarketplacePage() {
    const navigate = useNavigate()
    
    useEffect(() => {
      // TEAM-400: Listen for protocol events
      const unlistenModel = listen<string>('protocol:install-model', (event) => {
        const modelId = event.payload
        console.log('Protocol triggered for model:', modelId)
        
        // Navigate to model detail or start install
        handleInstall(modelId)
      })
      
      return () => {
        unlistenModel.then(fn => fn())
      }
    }, [])
    
    // ... rest of component
  }
  ```

### 5.2 Test Protocol Flow

- [ ] Test: `rbee://model/llama-3.2-1b`
- [ ] Verify:
  - [ ] Keeper opens
  - [ ] Navigates to marketplace
  - [ ] Shows model detail or starts download
  - [ ] Progress indicator appears
  - [ ] Model downloads successfully
  - [ ] Worker spawns automatically

---

## ✅ Phase 6: Testing (Days 6-7)

### 6.1 Unit Tests

- [ ] Test MarketplacePage renders
- [ ] Test model fetching
- [ ] Test install function
- [ ] Test progress indicator
- [ ] Test search/filter

### 6.2 Integration Tests

- [ ] Test full flow:
  1. Open Keeper
  2. Click Marketplace in sidebar
  3. Browse models
  4. Click "Install" on a model
  5. See progress
  6. Model downloads
  7. Worker spawns
  8. Model ready to use

### 6.3 Protocol Tests

- [ ] Test protocol from browser:
  1. Open marketplace.rbee.dev
  2. Click "Run with rbee"
  3. Keeper opens to marketplace
  4. Model installs automatically

---

## 📊 Success Criteria

### Must Have

- [ ] Marketplace page accessible at `/marketplace`
- [ ] Sidebar link to marketplace
- [ ] Models list displays from HuggingFace
- [ ] Install button works
- [ ] Download progress shows
- [ ] Worker spawns after download
- [ ] Protocol integration works
- [ ] Search and filter work

### Nice to Have

- [ ] Model detail pages
- [ ] Workers list page
- [ ] Download queue
- [ ] Cancel download
- [ ] Retry failed downloads
- [ ] Recently installed models

---

## 🚀 Deliverables

1. **Marketplace Page:** `pages/MarketplacePage.tsx`
2. **Model Detail Page:** `pages/ModelDetailPage.tsx`
3. **Tauri Commands:** `handlers/marketplace.rs`
4. **Progress Component:** `components/DownloadProgress.tsx`
5. **Routes:** `/marketplace`, `/marketplace/models/:id`
6. **Protocol Integration:** Listen for `protocol:install-model` events

---

## 📝 Notes

### Key Principles

1. **UI EXISTS** - Don't create from scratch, add to existing
2. **USE rbee-ui** - Import marketplace components
3. **USE marketplace-sdk** - WASM SDK for data fetching
4. **TAURI COMMANDS** - Bridge frontend to Rust backend
5. **PROTOCOL INTEGRATION** - Connect to CHECKLIST_04

### Common Pitfalls

- ❌ Don't create new UI app (Keeper exists!)
- ❌ Don't recreate components (use rbee-ui)
- ❌ Don't fetch data without SDK (use marketplace-sdk)
- ❌ Don't forget protocol integration
- ✅ Add page to existing Keeper
- ✅ Use existing routing
- ✅ Use existing Zustand store (if needed)
- ✅ Connect to protocol handler

### Keeper UI Structure

**Current:**
```
bin/00_rbee_keeper/ui/
├── src/
│   ├── App.tsx (React Router)
│   ├── pages/
│   │   ├── ServicesPage.tsx
│   │   ├── QueenPage.tsx
│   │   ├── HivePage.tsx
│   │   ├── SettingsPage.tsx
│   │   └── HelpPage.tsx
│   ├── store/ (Zustand)
│   └── components/
│       └── Shell.tsx (sidebar)
```

**After:**
```
bin/00_rbee_keeper/ui/
├── src/
│   ├── App.tsx (+ marketplace route)
│   ├── pages/
│   │   ├── MarketplacePage.tsx (NEW)
│   │   ├── ModelDetailPage.tsx (NEW)
│   │   └── ... existing pages
│   └── components/
│       ├── DownloadProgress.tsx (NEW)
│       └── ... existing components
```

---

**Start with Phase 1, Keeper UI exists!** ✅

**TEAM-400 🐝🎊**
