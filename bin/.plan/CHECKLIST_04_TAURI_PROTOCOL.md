# Checklist 04: Tauri Protocol & Auto-Run

**Timeline:** 1 week  
**Status:** 📋 NOT STARTED  
**Dependencies:** Checklist 01 (Shared Components), Checklist 02 (SDK)

---

## 🎯 Goal

**Keeper is already a Tauri app!** Add `rbee://` protocol registration and implement auto-run flow (one-click from browser to running model).

---

## ✅ Phase 0: Verify Existing Tauri Setup (Day 1, Morning)

### 0.1 Verify Tauri is Working

- [ ] Navigate to Keeper: `cd bin/00_rbee_keeper`
- [ ] Check `src-tauri/` directory exists
- [ ] Check `src-tauri/tauri.conf.json` exists
- [ ] Check `src/tauri_commands.rs` exists (already has commands!)
- [ ] Run Tauri dev: `cargo tauri dev`
- [ ] Verify app opens successfully

### 0.2 Review Existing Commands

- [ ] Read `src/tauri_commands.rs`
- [ ] Note existing commands:
  - `test_narration`
  - `ssh_list`, `ssh_open_config`
  - `get_installed_hives`
  - `queen_status`, `queen_start`, `queen_stop`, `queen_install`, `queen_rebuild`, `queen_uninstall`
  - `hive_start`, `hive_stop`, `hive_status`, `hive_install`, `hive_uninstall`, `hive_rebuild`
- [ ] Verify TypeScript bindings are generated
- [ ] Check `ui/src/lib/tauri/` for existing hooks

### 0.3 Review UI Structure

- [ ] Navigate to UI: `cd ui`
- [ ] Check `package.json` - verify `@tauri-apps/api` is installed
- [ ] Check if `@rbee/ui` is already used
- [ ] Run dev server: `pnpm dev`
- [ ] Verify UI works

---

## 🔗 Phase 1: Protocol Registration (Day 1, Afternoon)

### 1.1 Update Tauri Config for Protocol

- [ ] Open `src-tauri/tauri.conf.json`
- [ ] Add protocol to `bundle.protocols`:
  ```json
  {
    "bundle": {
      "identifier": "dev.rbee.keeper",
      "protocols": [
        {
          "name": "rbee",
          "schemes": ["rbee"]
        }
      ]
    }
  }
  ```
- [ ] Verify config is valid: `cargo tauri build --debug`

### 1.2 Create Protocol Handler Module

- [ ] Create `src/protocol_handler.rs`:
  ```rust
  //! Protocol handler for rbee:// URLs
  //! TEAM-XXX: Created protocol handler for marketplace integration
  
  use tauri::{AppHandle, Manager};
  use serde::{Serialize, Deserialize};
  
  #[derive(Clone, Serialize, Deserialize, Debug)]
  #[serde(tag = "type", rename_all = "snake_case")]
  pub enum ProtocolEvent {
      DownloadModel {
          source: String,
          model_id: String,
      },
      InstallWorker {
          worker_id: String,
      },
      AutoRun {
          source: String,
          model_id: String,
          hive_id: Option<String>,
      },
  }
  
  /// Parse rbee:// URL and emit event to frontend
  /// 
  /// Supported formats:
  /// - rbee://download/model/huggingface/llama-3.2-1b
  /// - rbee://install/worker/llm-worker-rbee-cuda
  /// - rbee://auto-run/model/huggingface/llama-3.2-1b
  /// - rbee://auto-run/model/huggingface/llama-3.2-1b?hive=my-remote-hive
  pub fn handle_protocol_url(app: &AppHandle, url: String) {
      tracing::info!("Received protocol URL: {}", url);
      
      // Parse: rbee://action/...
      let Some(path) = url.strip_prefix("rbee://") else {
          tracing::error!("Invalid protocol URL: {}", url);
          return;
      };
      
      // Split path and query
      let (path, query) = if let Some(idx) = path.find('?') {
          (&path[..idx], Some(&path[idx + 1..]))
      } else {
          (path, None)
      };
      
      let parts: Vec<&str> = path.split('/').collect();
      
      let event = match parts.as_slice() {
          ["download", "model", source, model_id] => {
              ProtocolEvent::DownloadModel {
                  source: source.to_string(),
                  model_id: model_id.to_string(),
              }
          }
          ["install", "worker", worker_id] => {
              ProtocolEvent::InstallWorker {
                  worker_id: worker_id.to_string(),
              }
          }
          ["auto-run", "model", source, model_id] => {
              let hive_id = query
                  .and_then(|q| q.split('&')
                      .find(|p| p.starts_with("hive="))
                      .map(|p| p.strip_prefix("hive=").unwrap().to_string()));
              
              ProtocolEvent::AutoRun {
                  source: source.to_string(),
                  model_id: model_id.to_string(),
                  hive_id,
              }
          }
          _ => {
              tracing::error!("Unknown protocol path: {}", path);
              return;
          }
      };
      
      tracing::info!("Emitting protocol event: {:?}", event);
      
      if let Err(e) = app.emit_all("protocol-event", &event) {
          tracing::error!("Failed to emit protocol event: {}", e);
      }
  }
  ```
- [ ] Add to `src/main.rs`:
  ```rust
  mod protocol_handler;
  ```

### 1.3 Wire Up Protocol Handler

- [ ] Update `src/main.rs` to handle protocol URLs:
  ```rust
  use std::env;
  
  fn main() {
      // Get command line arguments
      let args: Vec<String> = env::args().collect();
      
      tauri::Builder::default()
          .setup(move |app| {
              // Check if launched with URL (Linux/Windows)
              if args.len() > 1 && args[1].starts_with("rbee://") {
                  let url = args[1].clone();
                  let app_handle = app.handle();
                  protocol_handler::handle_protocol_url(&app_handle, url);
              }
              
              // macOS protocol handling
              #[cfg(target_os = "macos")]
              {
                  use tauri::Manager;
                  let app_handle = app.handle();
                  app.listen_global("open-url", move |event| {
                      if let Some(url) = event.payload() {
                          protocol_handler::handle_protocol_url(&app_handle, url.to_string());
                      }
                  });
              }
              
              Ok(())
          })
          // ... rest of builder
  }
  ```
- [ ] Test protocol handling: `./target/debug/rbee-keeper "rbee://download/model/huggingface/test"`

---

## 🎯 Phase 2: Auto-Run Commands (Day 2)

### 2.1 Create Auto-Run Module

- [ ] Create `src/auto_run.rs`:
  ```rust
  //! Auto-run flow: Download model + Install worker + Spawn worker
  //! TEAM-XXX: Created auto-run flow for one-click marketplace integration
  
  use anyhow::{Context, Result};
  use serde::{Deserialize, Serialize};
  
  #[derive(Debug, Serialize, Deserialize)]
  pub struct AutoRunRequest {
      pub hive_id: String,
      pub model_id: String,
      pub source: String,
  }
  
  #[derive(Debug, Serialize, Deserialize)]
  pub struct AutoRunProgress {
      pub step: String,
      pub status: String,
      pub message: String,
  }
  
  /// Execute auto-run flow
  /// 
  /// Steps:
  /// 1. Check if hive is running (start if needed)
  /// 2. Check if model exists (download if needed)
  /// 3. Detect best worker type (CUDA > Metal > CPU)
  /// 4. Check if worker is installed (install if needed)
  /// 5. Spawn worker with model
  pub async fn execute_auto_run(
      request: AutoRunRequest,
      progress_callback: impl Fn(AutoRunProgress),
  ) -> Result<String> {
      let queen_url = "http://localhost:8500";
      let client = reqwest::Client::new();
      
      // Step 1: Check hive status
      progress_callback(AutoRunProgress {
          step: "hive_check".to_string(),
          status: "in_progress".to_string(),
          message: format!("Checking hive: {}", request.hive_id),
      });
      
      let hive_running = check_hive_status(&client, queen_url, &request.hive_id).await?;
      
      if !hive_running {
          progress_callback(AutoRunProgress {
              step: "hive_start".to_string(),
              status: "in_progress".to_string(),
              message: format!("Starting hive: {}", request.hive_id),
          });
          
          start_hive(&client, queen_url, &request.hive_id).await?;
      }
      
      // Step 2: Check if model exists
      progress_callback(AutoRunProgress {
          step: "model_check".to_string(),
          status: "in_progress".to_string(),
          message: format!("Checking model: {}", request.model_id),
      });
      
      let model_exists = check_model_exists(&client, queen_url, &request.hive_id, &request.model_id).await?;
      
      if !model_exists {
          progress_callback(AutoRunProgress {
              step: "model_download".to_string(),
              status: "in_progress".to_string(),
              message: format!("Downloading model: {}", request.model_id),
          });
          
          let job_id = download_model(&client, queen_url, &request).await?;
          wait_for_job(&client, queen_url, &job_id, |status| {
              progress_callback(AutoRunProgress {
                  step: "model_download".to_string(),
                  status: "in_progress".to_string(),
                  message: format!("Downloading: {}", status),
              });
          }).await?;
      }
      
      // Step 3: Detect best worker
      progress_callback(AutoRunProgress {
          step: "worker_detect".to_string(),
          status: "in_progress".to_string(),
          message: "Detecting best worker type".to_string(),
      });
      
      let worker_type = detect_best_worker(&client, queen_url, &request.hive_id).await?;
      
      // Step 4: Check if worker is installed
      progress_callback(AutoRunProgress {
          step: "worker_check".to_string(),
          status: "in_progress".to_string(),
          message: format!("Checking worker: {}", worker_type),
      });
      
      let worker_installed = check_worker_installed(&client, queen_url, &request.hive_id, &worker_type).await?;
      
      if !worker_installed {
          progress_callback(AutoRunProgress {
              step: "worker_install".to_string(),
              status: "in_progress".to_string(),
              message: format!("Installing worker: {}", worker_type),
          });
          
          let job_id = install_worker(&client, queen_url, &request.hive_id, &worker_type).await?;
          wait_for_job(&client, queen_url, &job_id, |status| {
              progress_callback(AutoRunProgress {
                  step: "worker_install".to_string(),
                  status: "in_progress".to_string(),
                  message: format!("Installing: {}", status),
              });
          }).await?;
      }
      
      // Step 5: Spawn worker
      progress_callback(AutoRunProgress {
          step: "worker_spawn".to_string(),
          status: "in_progress".to_string(),
          message: "Spawning worker".to_string(),
      });
      
      let worker_id = spawn_worker(&client, queen_url, &request.hive_id, &request.model_id, &worker_type).await?;
      
      progress_callback(AutoRunProgress {
          step: "complete".to_string(),
          status: "success".to_string(),
          message: format!("Worker running: {}", worker_id),
      });
      
      Ok(worker_id)
  }
  
  // Helper functions (implement these)
  async fn check_hive_status(client: &reqwest::Client, queen_url: &str, hive_id: &str) -> Result<bool> {
      // TODO: Call Queen API to check hive status
      Ok(true)
  }
  
  async fn start_hive(client: &reqwest::Client, queen_url: &str, hive_id: &str) -> Result<()> {
      // TODO: Call Queen API to start hive
      Ok(())
  }
  
  async fn check_model_exists(client: &reqwest::Client, queen_url: &str, hive_id: &str, model_id: &str) -> Result<bool> {
      // TODO: Call Queen API to check if model exists
      Ok(false)
  }
  
  async fn download_model(client: &reqwest::Client, queen_url: &str, request: &AutoRunRequest) -> Result<String> {
      // TODO: Call Queen API to download model
      Ok("job-123".to_string())
  }
  
  async fn wait_for_job<F>(client: &reqwest::Client, queen_url: &str, job_id: &str, progress: F) -> Result<()>
  where
      F: Fn(&str),
  {
      // TODO: Poll Queen API for job status
      Ok(())
  }
  
  async fn detect_best_worker(client: &reqwest::Client, queen_url: &str, hive_id: &str) -> Result<String> {
      // TODO: Detect GPU capabilities
      // Priority: CUDA > Metal > CPU
      Ok("llm-worker-rbee-cuda".to_string())
  }
  
  async fn check_worker_installed(client: &reqwest::Client, queen_url: &str, hive_id: &str, worker_type: &str) -> Result<bool> {
      // TODO: Check if worker binary exists
      Ok(false)
  }
  
  async fn install_worker(client: &reqwest::Client, queen_url: &str, hive_id: &str, worker_type: &str) -> Result<String> {
      // TODO: Call Queen API to install worker
      Ok("job-456".to_string())
  }
  
  async fn spawn_worker(client: &reqwest::Client, queen_url: &str, hive_id: &str, model_id: &str, worker_type: &str) -> Result<String> {
      // TODO: Call Queen API to spawn worker
      Ok("worker-789".to_string())
  }
  ```
- [ ] Add to `src/main.rs`:
  ```rust
  mod auto_run;
  ```

### 2.2 Create Tauri Command for Auto-Run

- [ ] Add to `src/tauri_commands.rs`:
  ```rust
  use crate::auto_run::{execute_auto_run, AutoRunRequest, AutoRunProgress};
  use tauri::{AppHandle, Manager};
  
  #[tauri::command]
  #[specta::specta]
  pub async fn auto_run_model(
      app: AppHandle,
      hive_id: String,
      model_id: String,
      source: String,
  ) -> Result<String, String> {
      let request = AutoRunRequest {
          hive_id,
          model_id,
          source,
      };
      
      execute_auto_run(request, |progress| {
          // Emit progress events to frontend
          let _ = app.emit_all("auto-run-progress", &progress);
      })
      .await
      .map_err(|e| e.to_string())
  }
  ```
- [ ] Add `auto_run_model` to command list in test
- [ ] Regenerate TypeScript bindings: `cargo test`

---

## ⚛️ Phase 3: Frontend Integration (Day 3)

### 3.1 Create Protocol Event Listener

- [ ] Create `ui/src/hooks/useProtocolHandler.ts`:
  ```typescript
  import { useEffect } from 'react'
  import { listen } from '@tauri-apps/api/event'
  import { invoke } from '@tauri-apps/api/tauri'
  import { toast } from 'sonner'
  
  interface ProtocolEvent {
    type: 'download_model' | 'install_worker' | 'auto_run'
    source?: string
    model_id?: string
    worker_id?: string
    hive_id?: string
  }
  
  export function useProtocolHandler() {
    useEffect(() => {
      const unlisten = listen<ProtocolEvent>('protocol-event', async (event) => {
        console.log('Protocol event received:', event.payload)
        
        const { type, source, model_id, worker_id, hive_id } = event.payload
        
        try {
          switch (type) {
            case 'auto_run':
              if (source && model_id) {
                toast.loading(`Running ${model_id}...`)
                await handleAutoRun(hive_id || 'localhost', model_id, source)
              }
              break
            
            case 'download_model':
              if (source && model_id) {
                toast.loading(`Downloading ${model_id}...`)
                // TODO: Call download command
              }
              break
            
            case 'install_worker':
              if (worker_id) {
                toast.loading(`Installing ${worker_id}...`)
                // TODO: Call install command
              }
              break
          }
        } catch (error) {
          toast.error(`Failed: ${error}`)
        }
      })
      
      return () => {
        unlisten.then(fn => fn())
      }
    }, [])
  }
  
  async function handleAutoRun(hiveId: string, modelId: string, source: string) {
    try {
      const workerId = await invoke<string>('auto_run_model', {
        hiveId,
        modelId,
        source,
      })
      
      toast.success(`Model running! Worker ID: ${workerId}`)
      
      // Navigate to worker view
      // window.location.href = `/workers/${workerId}`
    } catch (error) {
      toast.error(`Failed to run model: ${error}`)
    }
  }
  ```
- [ ] Test hook in App component

### 3.2 Create Progress Listener

- [ ] Create `ui/src/hooks/useAutoRunProgress.ts`:
  ```typescript
  import { useEffect, useState } from 'react'
  import { listen } from '@tauri-apps/api/event'
  
  interface AutoRunProgress {
    step: string
    status: string
    message: string
  }
  
  export function useAutoRunProgress() {
    const [progress, setProgress] = useState<AutoRunProgress | null>(null)
    
    useEffect(() => {
      const unlisten = listen<AutoRunProgress>('auto-run-progress', (event) => {
        setProgress(event.payload)
      })
      
      return () => {
        unlisten.then(fn => fn())
      }
    }, [])
    
    return progress
  }
  ```
- [ ] Create progress UI component
- [ ] Test progress updates

### 3.3 Update App Component

- [ ] Update `ui/src/App.tsx`:
  ```tsx
  import { useProtocolHandler } from './hooks/useProtocolHandler'
  import { useAutoRunProgress } from './hooks/useAutoRunProgress'
  import { Toaster } from 'sonner'
  
  export function App() {
    useProtocolHandler()
    const progress = useAutoRunProgress()
    
    return (
      <>
        {/* Your existing app UI */}
        
        {/* Progress indicator */}
        {progress && (
          <div className="fixed bottom-4 right-4 bg-white shadow-lg rounded-lg p-4">
            <div className="font-bold">{progress.step}</div>
            <div className="text-sm text-gray-600">{progress.message}</div>
          </div>
        )}
        
        <Toaster />
      </>
    )
  }
  ```
- [ ] Test protocol handling end-to-end

---

## 🧪 Phase 4: Testing (Days 4-5)

### 4.1 Test Protocol Registration

- [ ] Build Tauri app: `cargo tauri build`
- [ ] Install app
- [ ] Test protocol from terminal:
  ```bash
  xdg-open "rbee://auto-run/model/huggingface/test"
  ```
- [ ] Verify app opens
- [ ] Verify event received in frontend
- [ ] Test on macOS (if available)
- [ ] Test on Windows (if available)

### 4.2 Test Auto-Run Flow

- [ ] Ensure Queen is running: `cargo run --bin queen-rbee`
- [ ] Ensure Hive is running: `cargo run --bin rbee-hive`
- [ ] Open marketplace.rbee.dev in browser
- [ ] Click "Run with rbee" on a model
- [ ] Verify Keeper opens
- [ ] Verify auto-run starts
- [ ] Verify progress updates appear
- [ ] Verify model downloads
- [ ] Verify worker installs
- [ ] Verify worker spawns
- [ ] Verify success message

### 4.3 Test Error Handling

- [ ] Test with Queen not running
- [ ] Test with Hive not running
- [ ] Test with invalid model ID
- [ ] Test with network error
- [ ] Verify error messages are clear
- [ ] Verify user can retry

### 4.4 Test Multi-Hive

- [ ] Add SSH config for remote hive
- [ ] Test auto-run with `?hive=remote-hive`
- [ ] Verify model downloads to correct hive
- [ ] Verify worker spawns on correct hive

---

## 📦 Phase 5: Build & Package (Day 6)

### 5.1 Build for Linux

- [ ] Build: `cargo tauri build`
- [ ] Test AppImage: `./target/release/bundle/appimage/rbee-keeper_*.AppImage`
- [ ] Test .deb: `sudo dpkg -i ./target/release/bundle/deb/rbee-keeper_*.deb`
- [ ] Verify protocol registration
- [ ] Test auto-run flow

### 5.2 Build for macOS (if available)

- [ ] Build: `cargo tauri build`
- [ ] Test .app bundle
- [ ] Test .dmg installer
- [ ] Verify protocol registration
- [ ] Test auto-run flow

### 5.3 Build for Windows (if available)

- [ ] Build: `cargo tauri build`
- [ ] Test .msi installer
- [ ] Verify protocol registration
- [ ] Test auto-run flow

---

## 🚀 Phase 6: Distribution (Day 7)

### 6.1 Create GitHub Release

- [ ] Tag version: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Create GitHub release
- [ ] Upload Linux AppImage
- [ ] Upload Linux .deb
- [ ] Upload macOS .dmg (if available)
- [ ] Upload Windows .msi (if available)
- [ ] Write release notes

### 6.2 Update Documentation

- [ ] Update README with protocol usage
- [ ] Document auto-run flow
- [ ] Add troubleshooting guide
- [ ] Update marketplace.rbee.dev with download links

---

## ✅ Success Criteria

### Must Have

- [ ] `rbee://` protocol registered
- [ ] Protocol handler works
- [ ] Auto-run flow works end-to-end
- [ ] Progress updates work
- [ ] Error handling works
- [ ] Linux build works
- [ ] Packaged for distribution

### Nice to Have

- [ ] macOS build works
- [ ] Windows build works
- [ ] Multi-hive support
- [ ] Auto-update mechanism
- [ ] Crash reporting

---

## 🚀 Deliverables

1. **Protocol Handler:** `rbee://` registered and working
2. **Auto-Run:** One-click from browser to running model
3. **Commands:** Auto-run with progress tracking
4. **Packages:** AppImage, .deb, .dmg, .msi
5. **Distribution:** GitHub releases

---

## 📝 Notes

### Key Differences from Original Plan

**Original Plan Assumed:**
- Need to initialize Tauri from scratch
- Need to create all commands
- Need to set up UI from scratch

**Reality:**
- ✅ Keeper is already a Tauri app
- ✅ Many commands already exist
- ✅ UI is already set up
- ✅ TypeScript bindings already generated

**What We Actually Need:**
- Add protocol registration to existing config
- Add protocol handler module
- Add auto-run command
- Wire up frontend listeners
- Test and package

### Existing Commands We Can Use

From `src/tauri_commands.rs`:
- `queen_status`, `queen_start`, `queen_stop` - For hive management
- `hive_status`, `hive_start`, `hive_stop` - For hive management
- `ssh_list`, `get_installed_hives` - For multi-hive support

**We just need to add:**
- Protocol handler
- Auto-run command
- Frontend listeners

---

**Much simpler than starting from scratch!** 🎉
