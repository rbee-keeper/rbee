# Complete Onboarding Flow - SEO to Running Model

**Date:** 2025-11-04  
**Status:** 🎯 FINAL ARCHITECTURE  
**Purpose:** User searches Google → Runs model locally in 5 minutes

---

## 🎯 The Complete Journey

### Step-by-Step User Flow

```
1. User searches Google: "llama model locally"
   ↓
2. User clicks: "rbee - run your llama model locally"
   ↓
3. Browser opens: marketplace.rbee.dev/models/llama-3.2-1b (Next.js on Cloudflare)
   ↓
4. User sees model page with button: "Run Llama Locally with rbee"
   ↓
5. User clicks button
   ↓
6. Browser tries: rbee://download/model/huggingface/llama-3.2-1b
   ↓
   ├─> rbee installed? → Opens Keeper (Tauri app)
   │                      ↓
   │                      Shows same marketplace UI (Tauri edition)
   │                      ↓
   │                      Button says: "Download Llama Model"
   │                      ↓
   │                      User clicks → Auto-downloads model
   │                      ↓
   │                      Auto-installs worker
   │                      ↓
   │                      Auto-starts hive
   │                      ↓
   │                      Auto-spawns worker
   │                      ↓
   │                      🎉 MODEL RUNNING!
   │
   └─> rbee NOT installed? → Shows install instructions
                              ↓
                              "rbee is available for Arch Linux and Debian"
                              ↓
                              Install via AUR or apt-get
                              ↓
                              Button: "Back to Llama Model"
                              ↓
                              User installs rbee
                              ↓
                              User clicks button again
                              ↓
                              Now rbee opens! ✅
```

---

## 🏗️ Architecture (CORRECTED)

### NOT SPA - It's TAURI!

```
┌─────────────────────────────────────────────────────────┐
│  @rbee/marketplace-components (Shared Package)          │
│  ├─> ModelCard.tsx                                      │
│  ├─> WorkerCard.tsx                                     │
│  ├─> MarketplaceGrid.tsx                                │
│  └─> FilterSidebar.tsx                                  │
│                                                          │
│  DUMB COMPONENTS - NO DATA FETCHING                     │
└─────────────────────────────────────────────────────────┘
                    ↓                    ↓
        ┌───────────────────┐  ┌────────────────────┐
        │  Next.js Site     │  │  Keeper (TAURI)    │
        │  (SSG/SSR)        │  │  (NOT SPA!)        │
        ├───────────────────┤  ├────────────────────┤
        │  marketplace.     │  │  Tauri app         │
        │  rbee.dev         │  │                    │
        │                   │  │  Same components   │
        │  SEO optimized    │  │  + Tauri commands  │
        │  Pre-rendered     │  │  + Native features │
        │                   │  │                    │
        │  Button:          │  │  Button:           │
        │  "Run with rbee"  │  │  "Download Model"  │
        │  → rbee:// link   │  │  → Tauri command   │
        └───────────────────┘  └────────────────────┘
```

---

## 📦 Monorepo Structure (CORRECTED)

```
frontend/
├─> packages/
│   ├─> marketplace-components/     ← SHARED COMPONENTS
│   │   ├─> src/
│   │   │   ├─> components/
│   │   │   │   ├─> ModelCard.tsx
│   │   │   │   ├─> WorkerCard.tsx
│   │   │   │   ├─> MarketplaceGrid.tsx
│   │   │   │   └─> FilterSidebar.tsx
│   │   │   └─> types/
│   │   │       ├─> model.ts
│   │   │       └─> worker.ts
│   │   └─> package.json
│   │
│   ├─> marketplace-sdk/            ← DATA LAYER
│   │   ├─> src/
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
│   │   │   ├─> models/
│   │   │   │   └─> [id]/
│   │   │   │       └─> page.tsx    ← SSG per model
│   │   │   └─> install/
│   │   │       └─> page.tsx        ← Installation instructions
│   │   └─> package.json
│   │
│   └─> keeper/                     ← TAURI APP (NOT SPA!)
│       ├─> src/                    ← React frontend
│       │   ├─> pages/
│       │   │   └─> MarketplacePage.tsx
│       │   └─> lib/
│       │       └─> tauriCommands.ts
│       ├─> src-tauri/              ← Rust backend
│       │   ├─> src/
│       │   │   ├─> main.rs
│       │   │   ├─> commands.rs     ← Tauri commands
│       │   │   └─> protocol.rs     ← rbee:// handler
│       │   └─> Cargo.toml
│       └─> package.json
```

---

## 🌐 Next.js Site (marketplace.rbee.dev)

### Model Detail Page

```tsx
// apps/marketplace-site/app/models/[id]/page.tsx

import { HuggingFaceClient } from '@rbee/marketplace-sdk'
import { ModelCard } from '@rbee/marketplace-components'
import { InstallationAwareButton } from '@/components/InstallationAwareButton'

export default async function ModelDetailPage({ params }: { params: { id: string } }) {
  const client = new HuggingFaceClient()
  const model = await client.getModel(params.id)
  
  return (
    <div className="container">
      <ModelCard
        model={model}
        downloadButton={
          <InstallationAwareButton
            modelId={model.id}
            modelName={model.name}
          />
        }
        mode="nextjs"
      />
      
      {/* SEO content */}
      <div className="model-details">
        <h2>Run {model.name} Locally with rbee</h2>
        <p>
          Download and run {model.name} on your own hardware. 
          Free, private, unlimited.
        </p>
        
        <h3>Why rbee?</h3>
        <ul>
          <li>✅ Free forever - no API costs</li>
          <li>✅ 100% private - your data never leaves your machine</li>
          <li>✅ No limits - run as much as you want</li>
          <li>✅ Use your own GPU - maximize performance</li>
        </ul>
      </div>
    </div>
  )
}
```

### Installation-Aware Button

```tsx
// apps/marketplace-site/components/InstallationAwareButton.tsx

'use client'

import { useState } from 'react'
import { openInKeeperWithIframe } from '@/lib/protocolDetection'
import { InstallModal } from './InstallModal'

interface Props {
  modelId: string
  modelName: string
}

export function InstallationAwareButton({ modelId, modelName }: Props) {
  const [showInstallModal, setShowInstallModal] = useState(false)
  const [isChecking, setIsChecking] = useState(false)
  
  const handleClick = async () => {
    setIsChecking(true)
    
    const rbeeUrl = `rbee://download/model/huggingface/${modelId}`
    const opened = await openInKeeperWithIframe(rbeeUrl)
    
    setIsChecking(false)
    
    if (!opened) {
      // rbee not installed - show install modal
      setShowInstallModal(true)
    }
    // If opened, user is now in Keeper app!
  }
  
  return (
    <>
      <button 
        onClick={handleClick}
        disabled={isChecking}
        className="btn-primary btn-lg"
      >
        {isChecking ? '⏳ Opening rbee...' : `🚀 Run ${modelName} Locally with rbee`}
      </button>
      
      {showInstallModal && (
        <InstallModal
          onClose={() => setShowInstallModal(false)}
          modelId={modelId}
          modelName={modelName}
        />
      )}
    </>
  )
}
```

### Install Modal

```tsx
// apps/marketplace-site/components/InstallModal.tsx

'use client'

interface Props {
  onClose: () => void
  modelId: string
  modelName: string
}

export function InstallModal({ onClose, modelId, modelName }: Props) {
  const rbeeUrl = `rbee://download/model/huggingface/${modelId}`
  
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Install rbee Keeper</h2>
        <p>
          To run <strong>{modelName}</strong> locally, you need to install rbee first.
        </p>
        
        <div className="install-options">
          <h3>Available for:</h3>
          
          {/* Arch Linux */}
          <div className="install-option">
            <h4>🐧 Arch Linux</h4>
            <pre><code>yay -S rbee-keeper</code></pre>
          </div>
          
          {/* Debian/Ubuntu */}
          <div className="install-option">
            <h4>🐧 Debian/Ubuntu</h4>
            <pre><code>curl -fsSL https://rbee.dev/install.sh | sh</code></pre>
          </div>
          
          {/* Coming soon */}
          <div className="install-option disabled">
            <h4>🍎 macOS (Coming Soon)</h4>
            <p>Sign up to get notified when macOS support is ready</p>
            <input type="email" placeholder="your@email.com" />
            <button className="btn-secondary">Notify Me</button>
          </div>
          
          <div className="install-option disabled">
            <h4>🪟 Windows (Coming Soon)</h4>
            <p>Sign up to get notified when Windows support is ready</p>
            <input type="email" placeholder="your@email.com" />
            <button className="btn-secondary">Notify Me</button>
          </div>
        </div>
        
        <div className="modal-footer">
          <p className="text-sm text-muted">
            After installing, come back and click the button again!
          </p>
          
          <a 
            href={rbeeUrl}
            className="btn-primary"
            onClick={() => {
              // Try again after user installs
              setTimeout(() => {
                window.location.reload()
              }, 1000)
            }}
          >
            ← Back to {modelName}
          </a>
        </div>
      </div>
    </div>
  )
}
```

---

## 🖥️ Keeper (Tauri App)

### Tauri Configuration

```json
// apps/keeper/src-tauri/tauri.conf.json

{
  "build": {
    "beforeDevCommand": "pnpm dev",
    "beforeBuildCommand": "pnpm build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "package": {
    "productName": "rbee Keeper",
    "version": "0.1.0"
  },
  "tauri": {
    "bundle": {
      "identifier": "dev.rbee.keeper",
      "protocols": [
        {
          "name": "rbee",
          "schemes": ["rbee"]
        }
      ]
    },
    "allowlist": {
      "all": false,
      "shell": {
        "all": false,
        "open": true
      },
      "protocol": {
        "asset": true,
        "assetScope": ["**"]
      }
    },
    "windows": [
      {
        "title": "rbee Keeper",
        "width": 1200,
        "height": 800
      }
    ]
  }
}
```

### Protocol Handler (Rust)

```rust
// apps/keeper/src-tauri/src/protocol.rs

use tauri::{AppHandle, Manager};

pub fn handle_protocol_url(app: &AppHandle, url: String) {
    println!("Received protocol URL: {}", url);
    
    // Parse: rbee://download/model/huggingface/llama-3.2-1b
    if let Some(path) = url.strip_prefix("rbee://") {
        let parts: Vec<&str> = path.split('/').collect();
        
        match parts.as_slice() {
            ["download", "model", source, model_id] => {
                // Emit event to frontend
                app.emit_all("protocol-download-model", ProtocolDownloadEvent {
                    source: source.to_string(),
                    model_id: model_id.to_string(),
                }).unwrap();
            }
            ["install", "worker", worker_id] => {
                app.emit_all("protocol-install-worker", ProtocolInstallEvent {
                    worker_id: worker_id.to_string(),
                }).unwrap();
            }
            _ => {
                eprintln!("Unknown protocol path: {}", path);
            }
        }
    }
}

#[derive(Clone, serde::Serialize)]
struct ProtocolDownloadEvent {
    source: String,
    model_id: String,
}

#[derive(Clone, serde::Serialize)]
struct ProtocolInstallEvent {
    worker_id: String,
}
```

### Main (Rust)

```rust
// apps/keeper/src-tauri/src/main.rs

mod protocol;
mod commands;

use tauri::Manager;

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // Listen for protocol URLs
            let app_handle = app.handle();
            
            #[cfg(target_os = "macos")]
            {
                app.listen_global("open-url", move |event| {
                    if let Some(url) = event.payload() {
                        protocol::handle_protocol_url(&app_handle, url.to_string());
                    }
                });
            }
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::download_model,
            commands::install_worker,
            commands::auto_run_model,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### Tauri Commands (Rust)

```rust
// apps/keeper/src-tauri/src/commands.rs

use tauri::State;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadModelRequest {
    pub hive_id: String,
    pub model_id: String,
    pub source: String,
}

#[tauri::command]
pub async fn download_model(request: DownloadModelRequest) -> Result<String, String> {
    println!("Downloading model: {:?}", request);
    
    // Call Queen API
    let client = reqwest::Client::new();
    let response = client
        .post("http://localhost:8500/v1/jobs")
        .json(&serde_json::json!({
            "operation": {
                "ModelDownload": {
                    "hive_id": request.hive_id,
                    "model_id": request.model_id,
                    "source": request.source
                }
            }
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let job: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    let job_id = job["job_id"].as_str().ok_or("No job_id")?;
    
    Ok(job_id.to_string())
}

#[tauri::command]
pub async fn install_worker(hive_id: String, worker_id: String) -> Result<String, String> {
    // Similar to download_model
    // Call Queen API with WorkerInstall operation
    todo!()
}

#[tauri::command]
pub async fn auto_run_model(
    hive_id: String,
    model_id: String,
    source: String
) -> Result<String, String> {
    println!("Auto-running model: {}", model_id);
    
    // Step 1: Check if hive is running
    // If not, start hive
    
    // Step 2: Check if model is downloaded
    // If not, download model
    
    // Step 3: Check if worker is installed
    // If not, install worker (auto-detect best: CUDA > Metal > CPU)
    
    // Step 4: Spawn worker with model
    
    // Step 5: Return worker ID
    
    Ok("worker-123".to_string())
}
```

### Frontend (React + Tauri)

```tsx
// apps/keeper/src/pages/MarketplacePage.tsx

import { useState, useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'
import { invoke } from '@tauri-apps/api/tauri'
import { ModelCard, MarketplaceGrid } from '@rbee/marketplace-components'
import { HuggingFaceClient } from '@rbee/marketplace-sdk'

export function MarketplacePage() {
  const [models, setModels] = useState<Model[]>([])
  const [isLoading, setIsLoading] = useState(true)
  
  useEffect(() => {
    // Fetch models (client-side)
    const client = new HuggingFaceClient()
    client.listModels().then(setModels).finally(() => setIsLoading(false))
    
    // Listen for protocol events
    const unlisten = listen<ProtocolDownloadEvent>('protocol-download-model', (event) => {
      console.log('Protocol download:', event.payload)
      handleAutoDownload(event.payload)
    })
    
    return () => {
      unlisten.then(fn => fn())
    }
  }, [])
  
  const handleAutoDownload = async (payload: ProtocolDownloadEvent) => {
    // User came from marketplace.rbee.dev
    // Auto-download and run model
    
    try {
      const jobId = await invoke<string>('auto_run_model', {
        hiveId: 'localhost',
        modelId: payload.model_id,
        source: payload.source
      })
      
      // Show notification
      toast.success(`Running ${payload.model_id}`)
      
      // Navigate to worker view
      navigate(`/workers/${jobId}`)
    } catch (error) {
      toast.error(`Failed to run model: ${error}`)
    }
  }
  
  const handleDownload = async (modelId: string) => {
    // User clicked download button in Keeper
    
    try {
      const jobId = await invoke<string>('download_model', {
        request: {
          hiveId: 'localhost',
          modelId: modelId,
          source: 'huggingface'
        }
      })
      
      toast.success(`Downloading ${modelId}`)
    } catch (error) {
      toast.error(`Failed to download: ${error}`)
    }
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
            downloadButton={
              <button onClick={() => handleDownload(model.id)} className="btn-primary">
                📦 Download Model
              </button>
            }
            mode="tauri"
          />
        )}
      />
    </div>
  )
}

interface ProtocolDownloadEvent {
  source: string
  model_id: string
}
```

---

## 🚀 Auto-Run Flow (The Magic!)

### When User Clicks "Run Llama Locally with rbee"

```rust
// apps/keeper/src-tauri/src/commands.rs

#[tauri::command]
pub async fn auto_run_model(
    hive_id: String,
    model_id: String,
    source: String
) -> Result<AutoRunResult, String> {
    let client = reqwest::Client::new();
    let queen_url = "http://localhost:8500";
    
    // Step 1: Ensure hive is running
    let hive_status = check_hive_status(&client, &queen_url, &hive_id).await?;
    if !hive_status.is_running {
        start_hive(&client, &queen_url, &hive_id).await?;
    }
    
    // Step 2: Check if model exists
    let model_exists = check_model_exists(&client, &queen_url, &hive_id, &model_id).await?;
    if !model_exists {
        // Download model
        let job_id = download_model_internal(&client, &queen_url, &hive_id, &model_id, &source).await?;
        
        // Wait for download to complete (stream progress)
        wait_for_job(&client, &queen_url, &job_id).await?;
    }
    
    // Step 3: Auto-detect best worker
    let worker_type = detect_best_worker(&client, &queen_url, &hive_id).await?;
    
    // Step 4: Check if worker is installed
    let worker_installed = check_worker_installed(&client, &queen_url, &hive_id, &worker_type).await?;
    if !worker_installed {
        // Install worker
        let job_id = install_worker_internal(&client, &queen_url, &hive_id, &worker_type).await?;
        wait_for_job(&client, &queen_url, &job_id).await?;
    }
    
    // Step 5: Spawn worker with model
    let worker_id = spawn_worker(&client, &queen_url, &hive_id, &model_id, &worker_type).await?;
    
    Ok(AutoRunResult {
        worker_id,
        model_id,
        hive_id,
        message: format!("Model {} is now running!", model_id)
    })
}

#[derive(Serialize)]
struct AutoRunResult {
    worker_id: String,
    model_id: String,
    hive_id: String,
    message: String,
}

async fn detect_best_worker(
    client: &reqwest::Client,
    queen_url: &str,
    hive_id: &str
) -> Result<String, String> {
    // Get hive capabilities
    let response = client
        .get(format!("{}/v1/hives/{}", queen_url, hive_id))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let hive: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
    let capabilities = &hive["capabilities"];
    
    // Priority: CUDA > Metal > CPU
    if capabilities["gpu"]["cuda"].as_bool().unwrap_or(false) {
        Ok("llm-worker-rbee-cuda".to_string())
    } else if capabilities["gpu"]["metal"].as_bool().unwrap_or(false) {
        Ok("llm-worker-rbee-metal".to_string())
    } else {
        Ok("llm-worker-rbee-cpu".to_string())
    }
}
```

---

## 🎯 Complete User Experience

### Scenario 1: First-Time User (Arch Linux)

```
1. Google search: "llama model locally"
2. Click: marketplace.rbee.dev/models/llama-3.2-1b
3. See: "Run Llama Locally with rbee" button
4. Click button
5. Wait 2 seconds...
6. Modal appears: "Install rbee Keeper"
7. See: "Arch Linux: yay -S rbee-keeper"
8. Copy command, run in terminal
9. rbee installs
10. Click: "Back to Llama Model"
11. Click: "Run Llama Locally with rbee" again
12. rbee Keeper opens! ✅
13. Keeper shows: "Downloading Llama 3.2 1B..."
14. Progress bar: 45%... 78%... 100%
15. Keeper shows: "Installing CUDA worker..."
16. Progress bar: 100%
17. Keeper shows: "Starting worker..."
18. 🎉 Model is running!
19. Chat interface appears
20. User types: "Hello!"
21. Model responds: "Hello! How can I help you?"
```

**Time from Google search to running model: 5 minutes** ⚡

### Scenario 2: Returning User

```
1. Google search: "mistral 7b locally"
2. Click: marketplace.rbee.dev/models/mistral-7b
3. Click: "Run Mistral Locally with rbee"
4. rbee Keeper opens immediately ✅
5. Auto-downloads model (30 seconds)
6. Auto-spawns worker (already installed)
7. 🎉 Model is running!
8. Chat interface appears
```

**Time: 30 seconds** 🚀

### Scenario 3: Multi-Hive User

```
1. User is in Keeper marketplace
2. User sees: "Download Model" button
3. User sees dropdown: "Run on..."
   - localhost (this PC)
   - workstation (192.168.1.100)
   - server (192.168.1.200)
4. User selects: "workstation"
5. Keeper downloads model to workstation
6. Keeper spawns worker on workstation
7. 🎉 Model running on remote hive!
```

---

## 📊 Architecture Summary

### Components

```
@rbee/marketplace-components
├─> ModelCard.tsx
├─> WorkerCard.tsx
├─> MarketplaceGrid.tsx
└─> FilterSidebar.tsx

DUMB COMPONENTS - NO DATA FETCHING
WORK IN NEXT.JS AND TAURI
```

### Next.js Site (marketplace.rbee.dev)

```
- SSG for SEO
- Pre-render top 1000 models
- Button: "Run with rbee" → rbee:// link
- Fallback: Install modal
```

### Keeper (Tauri App)

```
- Same components
- Button: "Download Model" → Tauri command
- Auto-run flow:
  1. Start hive (if needed)
  2. Download model (if needed)
  3. Install worker (if needed)
  4. Spawn worker
  5. 🎉 Running!
```

---

## ✅ Key Features

### 1. **Installation-Aware Button**
- ✅ Detects if rbee is installed
- ✅ Shows install instructions if not
- ✅ Returns user to model page after install

### 2. **Auto-Run Flow**
- ✅ One click from marketplace → running model
- ✅ Auto-downloads model
- ✅ Auto-installs worker
- ✅ Auto-starts hive
- ✅ Auto-spawns worker

### 3. **Multi-Hive Support**
- ✅ Dropdown to select hive
- ✅ Based on SSH config (NOT hives.conf!)
- ✅ Run models on remote machines

### 4. **SEO Goldmine**
- ✅ Every model gets own page
- ✅ Google indexes: "model name + rbee"
- ✅ Massive backlinks

### 5. **Zero Duplication**
- ✅ Same components in Next.js and Tauri
- ✅ Maintain once, works everywhere

---

## 🚀 Implementation Timeline

### Phase 1: Shared Components (1 week)
- Create `@rbee/marketplace-components`
- Make components dumb (props only)
- Test in both Next.js and Tauri

### Phase 2: Next.js Site (1 week)
- Build marketplace.rbee.dev
- SSG for top 1000 models
- Installation-aware button
- Install modal

### Phase 3: Tauri Integration (1 week)
- Protocol handler (rbee://)
- Tauri commands (download, install, auto-run)
- Frontend integration
- Test end-to-end

### Phase 4: Auto-Run Flow (3 days)
- Implement auto-run logic
- Detect best worker
- Progress tracking
- Error handling

### Phase 5: Multi-Hive (2 days)
- SSH config parsing
- Hive dropdown
- Remote execution

**Total: 3.5 weeks**

---

## 🎯 Success Metrics

**User Journey:**
- Google search → Running model: **5 minutes**
- Returning user → Running model: **30 seconds**

**SEO:**
- 1000+ model pages indexed
- "model name + rbee" rankings
- Backlinks from model searches

**Conversion:**
- Click "Run with rbee" → Install rate: **>50%**
- Install → First model running: **>80%**

---

**THIS IS THE COMPLETE FLOW!** 🚀
