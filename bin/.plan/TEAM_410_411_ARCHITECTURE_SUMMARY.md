# TEAM-410/411: Compatibility Matrix Architecture Summary

**Date:** 2025-11-05  
**Status:** ✅ DOCUMENTED

---

## 🏗️ Complete Architecture Overview

### Two Integration Paths

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPATIBILITY MATRIX                          │
│                                                                 │
│  Core Logic: marketplace-sdk/src/compatibility.rs (Rust)       │
│  ├─ check_compatibility()                                      │
│  ├─ filter_compatible_models()                                 │
│  └─ generate_compatibility_matrix()                            │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ├─────────────────┬─────────────────┐
                           │                 │                 │
                      PATH 1: Next.js    PATH 2: Tauri       │
                      (marketplace)      (keeper)            │
                           │                 │                 │
                           ▼                 ▼                 │
```

---

## 📊 PATH 1: Next.js Marketplace (SSG)

### Architecture: SDK → Node → Next.js SSG

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. marketplace-sdk (Rust WASM)                                  │
│    ├─ compatibility.rs (core logic)                             │
│    ├─ wasm_worker.rs (WASM bindings)                            │
│    └─ Build: wasm-pack build --target nodejs                    │
│       Output: marketplace-node/wasm/                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. marketplace-node (TypeScript Wrapper)                        │
│    ├─ Import WASM: import * as wasm from './wasm/...'          │
│    ├─ Export functions:                                         │
│    │  ├─ checkModelCompatibility(model)                        │
│    │  ├─ filterCompatibleModels(models)                        │
│    │  ├─ searchCompatibleModels(query)                         │
│    │  └─ listCompatibleModels(options)                         │
│    └─ Used by: Next.js at BUILD TIME                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Next.js Marketplace (SSG)                                    │
│    ├─ Import: import { listCompatibleModels } from             │
│    │          '@rbee/marketplace-node'                          │
│    ├─ Build time:                                               │
│    │  ├─ generateStaticParams() calls marketplace-node         │
│    │  ├─ Filters models by compatibility                       │
│    │  └─ Generates static HTML pages                           │
│    ├─ Output: Static HTML with compatibility data              │
│    └─ Deploy: Cloudflare Pages                                  │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. GitHub Actions (Cron Jobs)                                   │
│    ├─ Schedule: Daily (0 0 * * *) for top 100 list             │
│    ├─ Action:                                                   │
│    │  ├─ Fetch models from HuggingFace                         │
│    │  ├─ Filter compatible models (marketplace-node)           │
│    │  ├─ Rebuild static pages (next build)                     │
│    │  └─ Deploy (wrangler pages deploy dist/)                  │
│    └─ Cost: $0/month (free tier)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Runtime** | Node.js (build time only) |
| **Format** | WASM (wasm32-unknown-unknown) |
| **Wrapper** | marketplace-node (TypeScript) |
| **Execution** | Build time (SSG) |
| **Updates** | GitHub Actions cron (daily) |
| **Cost** | $0/month |
| **Network** | Yes (HuggingFace API at build time) |

### Data Flow

```
HuggingFace API
      ↓
marketplace-node (WASM)
      ↓
Next.js generateStaticParams()
      ↓
Static HTML pages
      ↓
Cloudflare Pages
      ↓
User Browser
```

---

## 🖥️ PATH 2: Tauri Keeper (Desktop App)

### Architecture: SDK → Tauri Commands → SPA GUI

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. marketplace-sdk (Rust Crate)                                 │
│    ├─ compatibility.rs (core logic)                             │
│    ├─ NO WASM (native Rust)                                     │
│    └─ Used by: Tauri commands directly                          │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Tauri Commands (Rust Backend)                                │
│    ├─ File: bin/00_rbee_keeper/src/commands/compatibility.rs   │
│    ├─ Functions:                                                │
│    │  ├─ #[tauri::command]                                      │
│    │  ├─ check_model_compatibility(model_id, worker_id)        │
│    │  ├─ list_compatible_workers(model_id)                     │
│    │  └─ list_compatible_models(worker_id)                     │
│    └─ Exposed to: Frontend via Tauri IPC                        │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Frontend API Wrapper (TypeScript)                            │
│    ├─ File: bin/00_rbee_keeper/ui/src/api/compatibility.ts     │
│    ├─ Import: import { invoke } from '@tauri-apps/api/tauri'   │
│    ├─ Functions:                                                │
│    │  ├─ checkModelCompatibility(modelId, workerId)            │
│    │  ├─ listCompatibleWorkers(modelId)                        │
│    │  └─ listCompatibleModels(workerId)                        │
│    └─ Used by: React components                                 │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. SPA Frontend (React + TypeScript)                            │
│    ├─ Components:                                               │
│    │  ├─ WorkerSelector (shows compatible workers)             │
│    │  ├─ CompatibilityBadge (shows status)                     │
│    │  ├─ CompatibilityWarningDialog (warns on incompatible)    │
│    │  └─ ModelCard (shows compatibility count)                 │
│    ├─ Pages:                                                    │
│    │  ├─ MarketplacePage (browse models with compat)           │
│    │  └─ WorkerManagementPage (browse workers with compat)     │
│    └─ User Flow:                                                │
│       ├─ Browse marketplace                                     │
│       ├─ Select model                                           │
│       ├─ Check compatibility (Tauri command)                    │
│       ├─ Select compatible worker                               │
│       └─ Install                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

| Aspect | Details |
|--------|---------|
| **Runtime** | Native Rust (Tauri backend) |
| **Format** | Native binary (no WASM) |
| **Wrapper** | Tauri commands (IPC bridge) |
| **Execution** | Runtime (on-demand) |
| **Updates** | Real-time (local checks) |
| **Cost** | $0 (local app) |
| **Network** | Optional (cache HuggingFace data) |

### Data Flow

```
User clicks model in Keeper
      ↓
React component calls API wrapper
      ↓
invoke('check_model_compatibility', { ... })
      ↓
Tauri IPC
      ↓
Rust command handler
      ↓
marketplace-sdk::check_compatibility()
      ↓
CompatibilityResult
      ↓
Tauri IPC
      ↓
React component updates UI
```

---

## 🔄 Comparison: Next.js vs Tauri

| Feature | Next.js (Marketplace) | Tauri (Keeper) |
|---------|----------------------|----------------|
| **Format** | WASM | Native Rust |
| **Wrapper** | marketplace-node | Tauri commands |
| **Execution** | Build time (SSG) | Runtime (on-demand) |
| **Updates** | GitHub Actions (daily) | Real-time (local) |
| **Network** | Yes (build time) | Optional (cache) |
| **Cost** | $0/month | $0 (local) |
| **Use Case** | Public marketplace | Desktop app |
| **Data** | Pre-computed (static) | Computed on-demand |

---

## 📦 Shared Components

### marketplace-sdk (Rust Core)

**Both paths use the same core logic:**

```rust
// bin/79_marketplace_core/marketplace-sdk/src/compatibility.rs

pub fn check_compatibility(
    model: &ModelMetadata,
    worker: &Worker,
) -> CompatibilityResult {
    // Single source of truth for compatibility logic
    // Used by BOTH Next.js (via WASM) and Tauri (native)
}
```

**Key Point:** Same Rust code, different compilation targets:
- Next.js: `wasm32-unknown-unknown` (WASM)
- Tauri: `x86_64-unknown-linux-gnu` (native)

---

## 🚀 Deployment Strategy

### Next.js Marketplace

```yaml
# .github/workflows/update-marketplace.yml
name: Update Marketplace

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
      
      - name: Install dependencies
        run: pnpm install
      
      - name: Build marketplace-sdk (WASM)
        run: |
          cd bin/79_marketplace_core/marketplace-sdk
          wasm-pack build --target nodejs --out-dir ../marketplace-node/wasm
      
      - name: Build marketplace-node
        run: |
          cd bin/79_marketplace_core/marketplace-node
          pnpm run build
      
      - name: Build Next.js marketplace
        run: |
          cd frontend/apps/marketplace
          pnpm run build
      
      - name: Deploy to Cloudflare Pages
        run: wrangler pages deploy frontend/apps/marketplace/out/
        env:
          CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
```

**Cost:** $0/month (GitHub Actions free tier: 2,000 min/month)

### Tauri Keeper

```bash
# Local build (no CI/CD needed)
cd bin/00_rbee_keeper
cargo build --release

# Tauri bundles for distribution
cargo tauri build
```

**Cost:** $0 (local builds)

---

## 📊 Update Intervals

### Next.js Marketplace

| Content | Update Interval | Method | Cost |
|---------|----------------|--------|------|
| **Top 100 list** | 24 hours | GitHub Actions | $0 |
| **Individual pages** | 48 hours | ISR | $0 |
| **Compatibility data** | On-demand | Cache forever | $0 |

**Total:** $0/month

### Tauri Keeper

| Content | Update Interval | Method | Cost |
|---------|----------------|--------|------|
| **Compatibility checks** | Real-time | Local compute | $0 |
| **Model metadata** | On-demand | Cache + HF API | $0 |
| **Worker catalog** | On app start | Local file | $0 |

**Total:** $0

---

## ✅ Implementation Status

### TEAM-410: Next.js Integration ✅ COMPLETE

- ✅ marketplace-sdk WASM bindings
- ✅ marketplace-node TypeScript wrapper
- ✅ Types added (CompatibilityResult, ModelMetadata, etc.)
- ✅ Functions exported (checkModelCompatibility, filterCompatibleModels, etc.)
- ✅ Build system working
- ✅ Documentation complete

**Ready for:** Next.js SSG implementation

### TEAM-411: Tauri Integration ⏳ WAITING

- ⏳ Tauri commands (not started)
- ⏳ Frontend API wrapper (not started)
- ⏳ React components (not started)
- ⏳ Install flow integration (not started)

**Blocked by:** TEAM-410 patterns and components

---

## 🎯 Key Takeaways

### For Next.js (Marketplace)

1. **SDK → Node → Next.js SSG**
2. Uses WASM (marketplace-node wrapper)
3. Build-time compatibility checks (SSG)
4. GitHub Actions updates daily
5. $0/month cost

### For Tauri (Keeper)

1. **SDK → Tauri Commands → SPA GUI**
2. Uses native Rust (no WASM)
3. Runtime compatibility checks (on-demand)
4. Local-first (no network required)
5. $0 cost

### Shared

1. Same core logic (marketplace-sdk)
2. Same compatibility algorithm
3. Same data structures
4. Different compilation targets
5. Different execution models

---

**TEAM-410/411 - Architecture Summary** ✅  
**Both paths documented and ready for implementation!** 🚀
