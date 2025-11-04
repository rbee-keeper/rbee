# Checklist 04: Tauri Integration & Auto-Run

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** Checklist 01 (Shared Components), Checklist 02 (SDK)

---

## 🎯 Goal

Convert Keeper to Tauri app, register `rbee://` protocol, implement auto-run flow (one-click from browser to running model).

---

## 📦 Phase 1: Tauri Setup (Day 1)

### 1.1 Install Tauri CLI

- [ ] Install Tauri CLI globally:
  ```bash
  cargo install tauri-cli
  ```
- [ ] Verify installation: `cargo tauri --version`

### 1.2 Initialize Tauri in Keeper

- [ ] Navigate to Keeper: `cd bin/00_rbee_keeper/ui`
- [ ] Initialize Tauri:
  ```bash
  cargo tauri init
  ```
- [ ] Answer prompts:
  - App name: `rbee Keeper`
  - Window title: `rbee Keeper`
  - Web assets: `../dist`
  - Dev server: `http://localhost:5173`
  - Before dev command: `pnpm dev`
  - Before build command: `pnpm build`
- [ ] Verify `src-tauri/` directory created

### 1.3 Configure Tauri

- [ ] Update `src-tauri/tauri.conf.json`:
  ```json
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
        "icon": [
          "icons/32x32.png",
          "icons/128x128.png",
          "icons/128x128@2x.png",
          "icons/icon.icns",
          "icons/icon.ico"
        ],
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
        "http": {
          "all": true,
          "request": true,
          "scope": ["http://localhost:*"]
        }
      },
      "windows": [
        {
          "title": "rbee Keeper",
          "width": 1400,
          "height": 900,
          "resizable": true,
          "fullscreen": false
        }
      ],
      "security": {
        "csp": null
      }
    }
  }
  ```
- [ ] Test Tauri dev: `cargo tauri dev`
- [ ] Verify app opens

### 1.4 Create App Icons

- [ ] Create icon.png (1024x1024)
- [ ] Generate icons:
  ```bash
  cargo tauri icon icon.png
  ```
- [ ] Verify icons created in `src-tauri/icons/`

---

## 🔗 Phase 2: Protocol Registration (Day 2)

### 2.1 Configure Protocol in Tauri

- [ ] Verify `protocols` in `tauri.conf.json` (already done in 1.3)
- [ ] Test protocol registration:
  ```bash
  cargo tauri build
  ./target/release/rbee-keeper
  ```
- [ ] Test protocol: `xdg-open "rbee://test"` (Linux)
- [ ] Verify app opens

### 2.2 Create Protocol Handler (Rust)

- [ ] Create `src-tauri/src/protocol.rs`:
  ```rust
  use tauri::{AppHandle, Manager};
  use serde::{Serialize, Deserialize};
  
  #[derive(Clone, Serialize, Deserialize)]
  pub struct ProtocolDownloadEvent {
      pub source: String,
      pub model_id: String,
  }
  
  #[derive(Clone, Serialize, Deserialize)]
  pub struct ProtocolInstallEvent {
      pub worker_id: String,
  }
  
  pub fn handle_protocol_url(app: &AppHandle, url: String) {
      println!("Received protocol URL: {}", url);
      
      // Parse: rbee://download/model/huggingface/llama-3.2-1b
      if let Some(path) = url.strip_prefix("rbee://") {
          let parts: Vec<&str> = path.split('/').collect();
          
          match parts.as_slice() {
              ["download", "model", source, model_id] => {
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
  ```
- [ ] Add to `src-tauri/src/main.rs`:
  ```rust
  mod protocol;
  
  // In main():
  #[cfg(target_os = "macos")]
  {
      use tauri::Manager;
      let app_handle = app.handle();
      app.listen_global("open-url", move |event| {
          if let Some(url) = event.payload() {
              protocol::handle_protocol_url(&app_handle, url.to_string());
          }
      });
  }
  ```
- [ ] Test protocol handling

### 2.3 Handle Protocol on Linux

- [ ] Update `src-tauri/src/main.rs`:
  ```rust
  use std::env;
  
  fn main() {
      // Get command line arguments
      let args: Vec<String> = env::args().collect();
      
      tauri::Builder::default()
          .setup(move |app| {
              // Check if launched with URL
              if args.len() > 1 && args[1].starts_with("rbee://") {
                  let url = args[1].clone();
                  let app_handle = app.handle();
                  protocol::handle_protocol_url(&app_handle, url);
              }
              Ok(())
          })
          .run(tauri::generate_context!())
          .expect("error while running tauri application");
  }
  ```
- [ ] Test on Linux: `./rbee-keeper "rbee://download/model/huggingface/test"`

---

## 🎯 Phase 3: Tauri Commands (Day 3)

### 3.1 Create Commands Module

- [ ] Create `src-tauri/src/commands.rs`:
  ```rust
  use serde::{Deserialize, Serialize};
  use tauri::State;
  
  #[derive(Debug, Serialize, Deserialize)]
  pub struct DownloadModelRequest {
      pub hive_id: String,
      pub model_id: String,
      pub source: String,
  }
  
  #[derive(Debug, Serialize, Deserialize)]
  pub struct InstallWorkerRequest {
      pub hive_id: String,
      pub worker_id: String,
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
  pub async fn install_worker(request: InstallWorkerRequest) -> Result<String, String> {
      println!("Installing worker: {:?}", request);
      
      let client = reqwest::Client::new();
      let response = client
          .post("http://localhost:8500/v1/jobs")
          .json(&serde_json::json!({
              "operation": {
                  "WorkerInstall": {
                      "hive_id": request.hive_id,
                      "worker_id": request.worker_id
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
  ```
- [ ] Add to `src-tauri/Cargo.toml`:
  ```toml
  [dependencies]
  reqwest = { version = "0.11", features = ["json"] }
  serde = { version = "1.0", features = ["derive"] }
  serde_json = "1.0"
  ```
- [ ] Add to `src-tauri/src/main.rs`:
  ```rust
  mod commands;
  
  // In Builder:
  .invoke_handler(tauri::generate_handler![
      commands::download_model,
      commands::install_worker,
      commands::auto_run_model,
  ])
  ```

### 3.2 Create Auto-Run Command

- [ ] Add to `src-tauri/src/commands.rs`:
  ```rust
  #[tauri::command]
  pub async fn auto_run_model(
      hive_id: String,
      model_id: String,
      source: String
  ) -> Result<String, String> {
      println!("Auto-running model: {}", model_id);
      
      let client = reqwest::Client::new();
      let queen_url = "http://localhost:8500";
      
      // Step 1: Check if hive is running
      let hive_status = check_hive_status(&client, queen_url, &hive_id).await?;
      if !hive_status {
          start_hive(&client, queen_url, &hive_id).await?;
      }
      
      // Step 2: Check if model exists
      let model_exists = check_model_exists(&client, queen_url, &hive_id, &model_id).await?;
      if !model_exists {
          // Download model
          let job_id = download_model_internal(&client, queen_url, &hive_id, &model_id, &source).await?;
          wait_for_job(&client, queen_url, &job_id).await?;
      }
      
      // Step 3: Auto-detect best worker
      let worker_type = detect_best_worker(&client, queen_url, &hive_id).await?;
      
      // Step 4: Check if worker is installed
      let worker_installed = check_worker_installed(&client, queen_url, &hive_id, &worker_type).await?;
      if !worker_installed {
          let job_id = install_worker_internal(&client, queen_url, &hive_id, &worker_type).await?;
          wait_for_job(&client, queen_url, &job_id).await?;
      }
      
      // Step 5: Spawn worker with model
      let worker_id = spawn_worker(&client, queen_url, &hive_id, &model_id, &worker_type).await?;
      
      Ok(worker_id)
  }
  
  async fn check_hive_status(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str
  ) -> Result<bool, String> {
      // TODO: Implement
      Ok(true)
  }
  
  async fn start_hive(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str
  ) -> Result<(), String> {
      // TODO: Implement
      Ok(())
  }
  
  async fn check_model_exists(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str,
      model_id: &str
  ) -> Result<bool, String> {
      // TODO: Implement
      Ok(false)
  }
  
  async fn download_model_internal(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str,
      model_id: &str,
      source: &str
  ) -> Result<String, String> {
      // TODO: Implement
      Ok("job-123".to_string())
  }
  
  async fn wait_for_job(
      client: &reqwest::Client,
      queen_url: &str,
      job_id: &str
  ) -> Result<(), String> {
      // TODO: Implement
      Ok(())
  }
  
  async fn detect_best_worker(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str
  ) -> Result<String, String> {
      // TODO: Implement - Priority: CUDA > Metal > CPU
      Ok("llm-worker-rbee-cuda".to_string())
  }
  
  async fn check_worker_installed(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str,
      worker_type: &str
  ) -> Result<bool, String> {
      // TODO: Implement
      Ok(false)
  }
  
  async fn install_worker_internal(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str,
      worker_type: &str
  ) -> Result<String, String> {
      // TODO: Implement
      Ok("job-456".to_string())
  }
  
  async fn spawn_worker(
      client: &reqwest::Client,
      queen_url: &str,
      hive_id: &str,
      model_id: &str,
      worker_type: &str
  ) -> Result<String, String> {
      // TODO: Implement
      Ok("worker-789".to_string())
  }
  ```
- [ ] Implement all helper functions
- [ ] Test auto-run flow

---

## ⚛️ Phase 4: Frontend Integration (Day 4)

### 4.1 Install Tauri API

- [ ] Install in Keeper UI:
  ```bash
  cd bin/00_rbee_keeper/ui
  pnpm add @tauri-apps/api
  ```

### 4.2 Create Tauri Hooks

- [ ] Create `src/lib/tauri.ts`:
  ```typescript
  import { invoke } from '@tauri-apps/api/tauri'
  import { listen } from '@tauri-apps/api/event'
  
  export interface DownloadModelRequest {
    hive_id: string
    model_id: string
    source: string
  }
  
  export interface InstallWorkerRequest {
    hive_id: string
    worker_id: string
  }
  
  export async function downloadModel(request: DownloadModelRequest): Promise<string> {
    return invoke('download_model', { request })
  }
  
  export async function installWorker(request: InstallWorkerRequest): Promise<string> {
    return invoke('install_worker', { request })
  }
  
  export async function autoRunModel(
    hiveId: string,
    modelId: string,
    source: string
  ): Promise<string> {
    return invoke('auto_run_model', {
      hiveId,
      modelId,
      source
    })
  }
  
  export function listenProtocolDownload(
    callback: (event: { source: string; model_id: string }) => void
  ) {
    return listen('protocol-download-model', (event) => {
      callback(event.payload as any)
    })
  }
  
  export function listenProtocolInstall(
    callback: (event: { worker_id: string }) => void
  ) {
    return listen('protocol-install-worker', (event) => {
      callback(event.payload as any)
    })
  }
  ```

### 4.3 Update Marketplace Page

- [ ] Update `src/pages/MarketplacePage.tsx`:
  ```tsx
  import { useState, useEffect } from 'react'
  import { ModelCard, MarketplaceGrid } from '@rbee/marketplace-components'
  import { HuggingFaceClient } from '@rbee/marketplace-sdk'
  import { downloadModel, listenProtocolDownload, autoRunModel } from '@/lib/tauri'
  import { toast } from 'sonner'
  
  export function MarketplacePage() {
    const [models, setModels] = useState<Model[]>([])
    const [isLoading, setIsLoading] = useState(true)
    
    useEffect(() => {
      // Fetch models
      const client = new HuggingFaceClient()
      client.listModels({ limit: 100 })
        .then(setModels)
        .finally(() => setIsLoading(false))
      
      // Listen for protocol events
      const unlisten = listenProtocolDownload(async (event) => {
        console.log('Protocol download:', event)
        await handleAutoDownload(event)
      })
      
      return () => {
        unlisten.then(fn => fn())
      }
    }, [])
    
    const handleAutoDownload = async (payload: { source: string; model_id: string }) => {
      try {
        toast.loading(`Running ${payload.model_id}...`)
        
        const workerId = await autoRunModel(
          'localhost',
          payload.model_id,
          payload.source
        )
        
        toast.success(`Model running! Worker ID: ${workerId}`)
        
        // Navigate to worker view
        // navigate(`/workers/${workerId}`)
      } catch (error) {
        toast.error(`Failed to run model: ${error}`)
      }
    }
    
    const handleDownload = async (modelId: string) => {
      try {
        toast.loading(`Downloading ${modelId}...`)
        
        const jobId = await downloadModel({
          hive_id: 'localhost',
          model_id: modelId,
          source: 'huggingface'
        })
        
        toast.success(`Download started! Job ID: ${jobId}`)
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
              mode="tauri"
            />
          )}
        />
      </div>
    )
  }
  ```
- [ ] Test marketplace page
- [ ] Test download button
- [ ] Test protocol handling

---

## 🧪 Phase 5: Testing (Day 5)

### 5.1 Test Protocol Registration

- [ ] Build Tauri app: `cargo tauri build`
- [ ] Install app
- [ ] Test protocol from terminal:
  ```bash
  xdg-open "rbee://download/model/huggingface/test"
  ```
- [ ] Verify app opens
- [ ] Verify event received in frontend
- [ ] Test on macOS (if available)
- [ ] Test on Windows (if available)

### 5.2 Test Auto-Run Flow

- [ ] Open marketplace.rbee.dev in browser
- [ ] Click "Run with rbee" on a model
- [ ] Verify Keeper opens
- [ ] Verify auto-run starts
- [ ] Verify model downloads
- [ ] Verify worker installs
- [ ] Verify worker spawns
- [ ] Verify model runs

### 5.3 Test Error Handling

- [ ] Test with Queen not running
- [ ] Test with Hive not running
- [ ] Test with invalid model ID
- [ ] Test with network error
- [ ] Verify error messages are clear
- [ ] Verify user can retry

### 5.4 Test Multi-Hive

- [ ] Add SSH config for remote hive
- [ ] Test auto-run on remote hive
- [ ] Verify model downloads to correct hive
- [ ] Verify worker spawns on correct hive

---

## 📦 Phase 6: Build & Package (Day 6)

### 6.1 Configure Build

- [ ] Update `src-tauri/Cargo.toml`:
  ```toml
  [package]
  name = "rbee-keeper"
  version = "0.1.0"
  description = "rbee Keeper - AI Infrastructure Manager"
  authors = ["rbee team"]
  license = "MIT"
  repository = "https://github.com/rbee/rbee"
  edition = "2021"
  ```
- [ ] Update version in `tauri.conf.json`
- [ ] Update version in `package.json`

### 6.2 Build for Linux

- [ ] Build: `cargo tauri build`
- [ ] Test AppImage: `./target/release/bundle/appimage/rbee-keeper_0.1.0_amd64.AppImage`
- [ ] Test .deb: `sudo dpkg -i ./target/release/bundle/deb/rbee-keeper_0.1.0_amd64.deb`
- [ ] Verify protocol registration
- [ ] Test auto-run flow

### 6.3 Build for macOS (if available)

- [ ] Build: `cargo tauri build`
- [ ] Test .app bundle
- [ ] Test .dmg installer
- [ ] Verify protocol registration
- [ ] Test auto-run flow

### 6.4 Build for Windows (if available)

- [ ] Build: `cargo tauri build`
- [ ] Test .msi installer
- [ ] Verify protocol registration
- [ ] Test auto-run flow

---

## 🚀 Phase 7: Distribution (Day 7)

### 7.1 Create GitHub Release

- [ ] Tag version: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Create GitHub release
- [ ] Upload Linux AppImage
- [ ] Upload Linux .deb
- [ ] Upload macOS .dmg (if available)
- [ ] Upload Windows .msi (if available)
- [ ] Write release notes

### 7.2 Create AUR Package (Arch Linux)

- [ ] Create PKGBUILD for rbee-keeper
- [ ] Test installation: `makepkg -si`
- [ ] Submit to AUR
- [ ] Verify package works

### 7.3 Create APT Repository (Debian/Ubuntu)

- [ ] Set up APT repository
- [ ] Add .deb package
- [ ] Create install script:
  ```bash
  #!/bin/bash
  curl -fsSL https://rbee.dev/install.sh | sh
  ```
- [ ] Test installation
- [ ] Update website with instructions

### 7.4 Update Documentation

- [ ] Update README with installation instructions
- [ ] Create user guide
- [ ] Create troubleshooting guide
- [ ] Update marketplace.rbee.dev with download links

---

## ✅ Success Criteria

### Must Have

- [ ] Tauri app builds successfully
- [ ] `rbee://` protocol registered
- [ ] Protocol handler works
- [ ] Auto-run flow works end-to-end
- [ ] Download command works
- [ ] Install command works
- [ ] Error handling works
- [ ] Linux build works
- [ ] Packaged for distribution

### Nice to Have

- [ ] macOS build works
- [ ] Windows build works
- [ ] AUR package available
- [ ] APT repository available
- [ ] Auto-update mechanism
- [ ] Crash reporting

---

## 🚀 Deliverables

1. **Tauri App:** rbee Keeper as native desktop app
2. **Protocol:** `rbee://` registered and working
3. **Auto-Run:** One-click from browser to running model
4. **Commands:** Download, install, auto-run
5. **Packages:** AppImage, .deb, .dmg, .msi
6. **Distribution:** GitHub releases, AUR, APT

---

## 📝 Notes

### Key Principles

1. **NATIVE APP** - Use Tauri for native performance
2. **PROTOCOL HANDLER** - Register `rbee://` on all platforms
3. **AUTO-RUN** - One-click to running model
4. **ERROR HANDLING** - Clear messages, easy retry
5. **MULTI-PLATFORM** - Linux, macOS, Windows

### Common Pitfalls

- ❌ Don't forget to register protocol in tauri.conf.json
- ❌ Don't hardcode localhost URLs (use env vars)
- ❌ Don't swallow errors (show to user)
- ✅ Test protocol registration on each platform
- ✅ Handle Queen/Hive not running
- ✅ Show progress during auto-run

---

**Complete each phase, test on real hardware, ship it!** ✅
