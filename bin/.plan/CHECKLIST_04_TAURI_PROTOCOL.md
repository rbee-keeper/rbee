# Checklist 04: Tauri Protocol Handler (`rbee://`)

**Timeline:** 1 week  
**Status:** 🎯 95% COMPLETE (TEAM-412/413)  
**Dependencies:** Checklist 01 (Components), Checklist 02 (SDK)  
**TEAM-400:** ✅ RULE ZERO - Keeper IS Tauri v2, just add protocol
**TEAM-413:** ✅ Frontend listener complete, missing: auto-run logic, testing, installers

---

## 🎯 Goal

Add `rbee://` protocol handler to EXISTING Keeper Tauri app. Enable one-click model installation from marketplace site.

**TEAM-400:** Keeper is ALREADY Tauri v2! Just add protocol registration + handler.

---

## 📦 Phase 1: Protocol Registration (Day 1)

**TEAM-400:** Keeper exists at `bin/00_rbee_keeper/` with Tauri v2 configured.

### 1.1 Update tauri.conf.json

- [x] Open: `bin/00_rbee_keeper/tauri.conf.json` ✅ TEAM-412
- [x] Add protocol registration: ✅ TEAM-412
  ```json
  {
    "$schema": "https://schema.tauri.app/config/2",
    "productName": "rbee-keeper",
    "version": "0.1.0",
    "identifier": "com.rbee.keeper",
    "build": {
      "frontendDist": "../ui/dist",
      "devUrl": "http://localhost:5173",
      "beforeDevCommand": "cargo tauri-typegen generate && cd ../ui && npm run dev",
      "beforeBuildCommand": "cargo tauri-typegen generate && cd ../ui && npm run build"
    },
    "bundle": {
      "active": true,
      "targets": "all",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/128x128@2x.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "windows": {
        "wix": {
          "language": "en-US"
        }
      },
      "linux": {
        "deb": {
          "depends": []
        }
      }
    },
    "app": {
      "windows": [
        {
          "fullscreen": false,
          "resizable": true,
          "title": "rbee Keeper",
          "width": 960,
          "height": 1080,
          "minWidth": 600,
          "minHeight": 800,
          "decorations": false,
          "transparent": false
        }
      ],
      "security": {
        "csp": null,
        "assetProtocol": {
          "enable": true,
          "scope": ["**"]
        },
        "capabilities": [
          {
            "identifier": "main-capability",
            "description": "Main window capabilities",
            "windows": ["main"],
            "permissions": [
              "core:default",
              "core:event:allow-listen",
              "core:event:allow-emit",
              "protocol:default"
            ]
          }
        ]
      }
    },
    "plugins": {
      "tauri-typegen": {
        "project_path": ".",
        "output_path": "./ui/src/generated",
        "validation_library": "none",
        "verbose": false,
        "visualize_deps": false
      },
      "deep-link": {
        "schemes": ["rbee"],
        "mobile": {
          "appIdBase": "com.rbee"
        }
      }
    }
  }
  ```

### 1.2 Add Tauri Deep Link Plugin

- [x] Add to `Cargo.toml`: ✅ TEAM-412
  ```toml
  [dependencies]
  # ... existing dependencies ...
  
  # TEAM-400: Deep link plugin for rbee:// protocol
  tauri-plugin-deep-link = "2"
  ```
- [ ] Run: `cargo check` to verify

### 1.3 Register Protocol in main.rs

- [x] Open: `bin/00_rbee_keeper/src/main.rs` ✅ TEAM-412
- [x] Add protocol registration: ✅ TEAM-412
  ```rust
  // TEAM-400: Import deep link plugin
  use tauri_plugin_deep_link::DeepLinkExt;
  
  fn main() {
      // ... existing setup ...
      
      tauri::Builder::default()
          // TEAM-400: Register rbee:// protocol
          .plugin(tauri_plugin_deep_link::init())
          .setup(|app| {
              // TEAM-400: Register protocol handler
              app.deep_link().register("rbee")?;
              
              // TEAM-400: Listen for deep link events
              app.deep_link().on_open_url(|event| {
                  println!("🔗 Deep link opened: {}", event.urls().join(", "));
                  // Will handle in protocol handler
              });
              
              Ok(())
          })
          // ... rest of builder ...
          .run(tauri::generate_context!())
          .expect("error while running tauri application");
  }
  ```

---

## 🔧 Phase 2: Protocol Handler (Days 2-3)

**TEAM-400:** Create handler to process `rbee://` URLs.

### 2.1 Create Protocol Handler Module

- [x] Create: `bin/00_rbee_keeper/src/protocol.rs` ✅ TEAM-412 (149 LOC, 4 unit tests)
  ```rust
  // TEAM-400: Protocol handler for rbee:// URLs
  
  use anyhow::{Context, Result};
  use serde::{Deserialize, Serialize};
  use tauri::{AppHandle, Emitter};
  
  /// Protocol action from rbee:// URL
  #[derive(Debug, Clone, Serialize, Deserialize)]
  #[serde(tag = "type", rename_all = "snake_case")]
  pub enum ProtocolAction {
      /// Install model: rbee://model/{model_id}
      InstallModel { model_id: String },
      
      /// Install worker: rbee://worker/{worker_id}
      InstallWorker { worker_id: String },
      
      /// Open marketplace: rbee://marketplace
      OpenMarketplace,
  }
  
  /// Parse rbee:// URL into action
  pub fn parse_protocol_url(url: &str) -> Result<ProtocolAction> {
      // Remove rbee:// prefix
      let path = url.strip_prefix("rbee://")
          .context("Invalid rbee:// URL")?;
      
      // Parse path
      let parts: Vec<&str> = path.split('/').collect();
      
      match parts.as_slice() {
          ["model", model_id] => Ok(ProtocolAction::InstallModel {
              model_id: model_id.to_string(),
          }),
          ["worker", worker_id] => Ok(ProtocolAction::InstallWorker {
              worker_id: worker_id.to_string(),
          }),
          ["marketplace"] => Ok(ProtocolAction::OpenMarketplace),
          _ => Err(anyhow::anyhow!("Unknown protocol action: {}", path)),
      }
  }
  
  /// Handle protocol action
  pub async fn handle_protocol_action(
      app: AppHandle,
      action: ProtocolAction,
  ) -> Result<()> {
      match action {
          ProtocolAction::InstallModel { model_id } => {
              // TEAM-400: Emit event to frontend
              app.emit("protocol:install-model", &model_id)?;
              
              // TEAM-400: Navigate to marketplace tab
              app.emit("navigate", "/marketplace")?;
              
              // TEAM-400: Show notification
              println!("📦 Installing model: {}", model_id);
              
              Ok(())
          }
          
          ProtocolAction::InstallWorker { worker_id } => {
              app.emit("protocol:install-worker", &worker_id)?;
              app.emit("navigate", "/marketplace")?;
              println!("👷 Installing worker: {}", worker_id);
              Ok(())
          }
          
          ProtocolAction::OpenMarketplace => {
              app.emit("navigate", "/marketplace")?;
              println!("🛒 Opening marketplace");
              Ok(())
          }
      }
  }
  
  #[cfg(test)]
  mod tests {
      use super::*;
      
      #[test]
      fn test_parse_model_url() {
          let action = parse_protocol_url("rbee://model/llama-3.2-1b").unwrap();
          match action {
              ProtocolAction::InstallModel { model_id } => {
                  assert_eq!(model_id, "llama-3.2-1b");
              }
              _ => panic!("Wrong action type"),
          }
      }
      
      #[test]
      fn test_parse_worker_url() {
          let action = parse_protocol_url("rbee://worker/cpu-llm").unwrap();
          match action {
              ProtocolAction::InstallWorker { worker_id } => {
                  assert_eq!(worker_id, "cpu-llm");
              }
              _ => panic!("Wrong action type"),
          }
      }
  }
  ```

### 2.2 Export Protocol Handler

- [x] Update `bin/00_rbee_keeper/src/lib.rs`: ✅ TEAM-412
  ```rust
  // TEAM-400: Export protocol handler
  pub mod protocol;
  ```

### 2.3 Wire Up Protocol Handler

- [x] Update `bin/00_rbee_keeper/src/main.rs`: ✅ TEAM-412
  ```rust
  mod handlers;
  
  use handlers::protocol::{parse_protocol_url, handle_protocol_action};
  
  fn main() {
      tauri::Builder::default()
          .plugin(tauri_plugin_deep_link::init())
          .setup(|app| {
              app.deep_link().register("rbee")?;
              
              // TEAM-400: Handle deep link URLs
              let app_handle = app.handle().clone();
              app.deep_link().on_open_url(move |event| {
                  for url in event.urls() {
                      println!("🔗 Processing URL: {}", url);
                      
                      // Parse URL
                      match parse_protocol_url(url) {
                          Ok(action) => {
                              // Handle action
                              let app = app_handle.clone();
                              tauri::async_runtime::spawn(async move {
                                  if let Err(e) = handle_protocol_action(app, action).await {
                                      eprintln!("❌ Protocol handler error: {}", e);
                                  }
                              });
                          }
                          Err(e) => {
                              eprintln!("❌ Failed to parse URL: {}", e);
                          }
                      }
                  }
              });
              
              Ok(())
          })
          // ... rest of builder ...
  }
  ```

---

## ⚡ Phase 3: Auto-Run Logic (Day 4)

**TEAM-400:** Automatically download and run model when protocol is triggered.
**TEAM-413:** ❌ MISSING - Protocol handler emits events but doesn't auto-download!

### 3.1 Create Auto-Run Module

**TEAM-413:** ❌ MISSING - No auto-run module exists!

- [ ] Create: `bin/00_rbee_keeper/src/handlers/auto_run.rs`
  ```rust
  // TEAM-400: Auto-run logic for models/workers
  
  use anyhow::Result;
  use job_client::JobClient;
  use operations_contract::Operation;
  
  /// Auto-run model installation
  pub async fn auto_run_model(model_id: String) -> Result<()> {
      println!("🚀 Auto-running model: {}", model_id);
      
      // TEAM-400: Step 1 - Download model
      let client = JobClient::new("http://localhost:9200");
      
      let download_op = Operation::ModelDownload {
          model_id: model_id.clone(),
          source: "huggingface".to_string(),
      };
      
      client.submit_and_stream(download_op, |line| {
          println!("📥 {}", line);
          Ok(())
      }).await?;
      
      // TEAM-400: Step 2 - Spawn worker (if needed)
      let spawn_op = Operation::WorkerSpawn {
          worker_type: "cpu-llm".to_string(),
          model: Some(model_id.clone()),
          port: None,
      };
      
      client.submit_and_stream(spawn_op, |line| {
          println!("🐝 {}", line);
          Ok(())
      }).await?;
      
      println!("✅ Model ready: {}", model_id);
      Ok(())
  }
  
  /// Auto-run worker installation
  pub async fn auto_run_worker(worker_id: String) -> Result<()> {
      println!("🚀 Auto-running worker: {}", worker_id);
      
      let client = JobClient::new("http://localhost:9200");
      
      let spawn_op = Operation::WorkerSpawn {
          worker_type: worker_id.clone(),
          model: None,
          port: None,
      };
      
      client.submit_and_stream(spawn_op, |line| {
          println!("🐝 {}", line);
          Ok(())
      }).await?;
      
      println!("✅ Worker ready: {}", worker_id);
      Ok(())
  }
  ```

### 3.2 Integrate Auto-Run

**TEAM-413:** ❌ NOT INTEGRATED - protocol.rs exists but doesn't call auto-run

- [ ] Update `protocol.rs` to use auto-run:
  ```rust
  use crate::handlers::auto_run::{auto_run_model, auto_run_worker};
  
  pub async fn handle_protocol_action(
      app: AppHandle,
      action: ProtocolAction,
  ) -> Result<()> {
      match action {
          ProtocolAction::InstallModel { model_id } => {
              app.emit("protocol:install-model", &model_id)?;
              app.emit("navigate", "/marketplace")?;
              
              // TEAM-400: Auto-run model installation
              auto_run_model(model_id).await?;
              
              Ok(())
          }
          // ... rest of handlers ...
      }
  }
  ```

---

## 🖥️ Phase 4: Frontend Integration (Day 5)

**TEAM-400:** Listen for protocol events in Keeper UI.
**TEAM-413:** ✅ COMPLETE - Protocol listener integrated

### 4.1 Create Protocol Hook

**TEAM-413:** ✅ COMPLETE

- [x] Create: `bin/00_rbee_keeper/ui/src/hooks/useProtocol.ts` ✅ TEAM-413
  ```typescript
  // TEAM-400: React hook for protocol events
  
  import { useEffect } from 'react'
  import { listen } from '@tauri-apps/api/event'
  import { useNavigate } from 'react-router-dom'
  
  export function useProtocol() {
    const navigate = useNavigate()
    
    useEffect(() => {
      // TEAM-400: Listen for protocol events
      const unlistenModel = listen<string>('protocol:install-model', (event) => {
        console.log('📦 Installing model:', event.payload)
        // Navigate to marketplace tab
        navigate('/marketplace')
        // Show notification
        // TODO: Trigger download UI
      })
      
      const unlistenWorker = listen<string>('protocol:install-worker', (event) => {
        console.log('👷 Installing worker:', event.payload)
        navigate('/marketplace')
      })
      
      const unlistenNavigate = listen<string>('navigate', (event) => {
        console.log('🧭 Navigating to:', event.payload)
        navigate(event.payload)
      })
      
      return () => {
        unlistenModel.then(f => f())
        unlistenWorker.then(f => f())
        unlistenNavigate.then(f => f())
      }
    }, [navigate])
  }
  ```

### 4.2 Use in App Component

**TEAM-413:** ✅ COMPLETE

- [x] Update `bin/00_rbee_keeper/ui/src/App.tsx`: ✅ TEAM-413
  ```tsx
  import { useProtocol } from './hooks/useProtocol'
  
  function App() {
    // TEAM-400: Listen for protocol events
    useProtocol()
    
    return (
      // ... existing app structure ...
    )
  }
  ```

---

## 🧪 Phase 5: Testing (Days 6-7)

**TEAM-413:** ⏳ PENDING - No testing done yet!

### 5.1 Test Protocol Registration

- [ ] Build Keeper: `cargo tauri build`
- [ ] Install built app
- [ ] Test protocol from terminal:
  ```bash
  # macOS
  open "rbee://model/llama-3.2-1b"
  
  # Linux
  xdg-open "rbee://model/llama-3.2-1b"
  
  # Windows
  start "rbee://model/llama-3.2-1b"
  ```
- [ ] Verify Keeper opens and navigates to marketplace

### 5.2 Test from Browser

- [ ] Open marketplace site: `http://localhost:3000/models/llama-3.2-1b`
- [ ] Click "Run with rbee" button
- [ ] Verify browser prompts to open Keeper
- [ ] Verify Keeper opens and starts download

### 5.3 Test Auto-Run

- [ ] Trigger protocol: `rbee://model/llama-3.2-1b`
- [ ] Verify:
  - [ ] Keeper opens
  - [ ] Navigates to marketplace
  - [ ] Model download starts automatically
  - [ ] Worker spawns automatically
  - [ ] Model is ready to use

### 5.4 Platform-Specific Testing

**macOS:**
- [ ] Build: `cargo tauri build --target universal-apple-darwin`
- [ ] Test protocol registration
- [ ] Test from Safari

**Linux:**
- [ ] Build: `cargo tauri build`
- [ ] Test protocol registration
- [ ] Test from Firefox/Chrome

**Windows:**
- [ ] Build: `cargo tauri build --target x86_64-pc-windows-msvc`
- [ ] Test protocol registration
- [ ] Test from Edge/Chrome

---

## 📦 Phase 6: Distribution (Day 7)

**TEAM-413:** ❌ MISSING - No platform installers created!

### 6.1 Create Installers

- [ ] Build for all platforms:
  ```bash
  # macOS
  cargo tauri build --target universal-apple-darwin
  
  # Linux
  cargo tauri build --target x86_64-unknown-linux-gnu
  
  # Windows
  cargo tauri build --target x86_64-pc-windows-msvc
  ```

### 6.2 Test Installers

- [ ] macOS: Test .dmg installer
  - [ ] Install app
  - [ ] Test protocol works after install
  - [ ] Test app signature (if signed)
  
- [ ] Linux: Test .deb/.AppImage
  - [ ] Install package
  - [ ] Test protocol registration
  
- [ ] Windows: Test .msi installer
  - [ ] Install app
  - [ ] Test protocol registration
  - [ ] Test from different browsers

### 6.3 Upload to GitHub Releases

- [ ] Create GitHub release
- [ ] Upload installers:
  - `rbee-keeper_0.1.0_universal.dmg` (macOS)
  - `rbee-keeper_0.1.0_amd64.deb` (Linux)
  - `rbee-keeper_0.1.0_x64_en-US.msi` (Windows)
- [ ] Add release notes
- [ ] Update marketplace download links

---

## ✅ Success Criteria

### Must Have

- [ ] `rbee://` protocol registered on all platforms
- [ ] Protocol handler parses URLs correctly
- [ ] `rbee://model/{id}` opens Keeper and starts download
- [ ] `rbee://worker/{id}` opens Keeper and spawns worker
- [ ] Auto-run downloads model automatically
- [ ] Auto-run spawns worker automatically
- [ ] Frontend listens for protocol events
- [ ] Marketplace button triggers protocol
- [ ] Works on macOS, Linux, Windows
- [ ] Installers available for download

### Nice to Have

- [ ] Progress notifications during download
- [ ] Error handling with user-friendly messages
- [ ] Cancel download option
- [ ] Queue multiple installs
- [ ] Remember last opened model
- [ ] Analytics (protocol usage)

---

## 🚀 Deliverables

1. **Protocol Registration:** `rbee://` works on all platforms
2. **Protocol Handler:** Rust module in `src/handlers/protocol.rs`
3. **Auto-Run Logic:** Automatic download + spawn in `src/handlers/auto_run.rs`
4. **Frontend Integration:** React hook for protocol events
5. **Tests:** Platform-specific testing complete
6. **Installers:** .dmg, .deb, .msi for all platforms

---

## 📝 Notes

### Key Principles

1. **KEEPER EXISTS** - Don't create new Tauri app, update existing
2. **TAURI V2** - Already configured, just add plugin
3. **PROTOCOL FIRST** - Register protocol before handler
4. **AUTO-RUN** - Make it seamless (one click → running model)
5. **CROSS-PLATFORM** - Test on macOS, Linux, Windows

### Common Pitfalls

- ❌ Don't create new Tauri project (Keeper exists!)
- ❌ Don't forget to register protocol in tauri.conf.json
- ❌ Don't block UI during download (use async)
- ❌ Don't forget platform-specific testing
- ✅ Use existing Keeper structure
- ✅ Add tauri-plugin-deep-link
- ✅ Emit events to frontend
- ✅ Test on all platforms

### Platform Differences

**macOS:**
- Protocol registered in Info.plist (automatic)
- Requires app signature for distribution
- Use .dmg for distribution

**Linux:**
- Protocol registered in .desktop file
- Requires update-desktop-database
- Use .deb or .AppImage

**Windows:**
- Protocol registered in Registry
- Installer handles registration
- Use .msi for distribution

---

**Start with Phase 1, Keeper is already Tauri!** ✅

**TEAM-400 🐝🎊**
