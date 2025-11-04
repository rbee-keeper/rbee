# 🎉 TEAM-400 Architecture Findings 🎉

**Date:** 2025-11-04  
**Status:** ✅ INVESTIGATION COMPLETE

---

## Executive Summary

After investigating the existing codebase, I found that the current checklists need **MAJOR UPDATES** to align with the actual architecture. Here's what I discovered:

---

## Key Findings

### 1. ✅ Keeper is ALREADY a Tauri v2 App

**Location:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/`

**Current Structure:**
```
bin/00_rbee_keeper/
├── src/
│   ├── main.rs (CLI + Tauri entry point)
│   ├── lib.rs (shared library)
│   ├── cli/ (CLI commands)
│   ├── handlers/ (Tauri command handlers)
│   └── platform/ (platform-specific code)
├── ui/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/ (Dashboard, Hives, Workers, Settings, Logs)
│   │   ├── components/
│   │   ├── store/ (Zustand state management)
│   │   └── api/ (Tauri API bindings)
│   └── package.json (@rbee/keeper-ui)
├── Cargo.toml (Tauri v2 dependencies)
└── tauri.conf.json
```

**Key Dependencies:**
- `tauri = "2"` ✅
- `tauri-specta = "2.0.0-rc.21"` ✅ (TypeScript type generation)
- `specta = "2.0.0-rc.22"` ✅
- `@tauri-apps/api = "^2.9.0"` ✅
- React 19 + Vite + Zustand ✅

**What this means:**
- ❌ We DON'T need to "set up Tauri" - it's already done
- ✅ We just need to ADD protocol handler (`rbee://`)
- ✅ We just need to ADD marketplace UI pages
- ✅ We just need to ADD auto-run logic

---

### 2. ✅ Marketplace Next.js App Already Exists

**Location:** `/home/vince/Projects/llama-orch/frontend/apps/marketplace/`

**Current State:**
- ✅ Next.js 15 configured
- ✅ Cloudflare Pages deployment configured
- ✅ Tailwind CSS 4 configured
- ❌ Only has default `page.tsx` (needs marketplace content)

**What this means:**
- ❌ We DON'T need to create the app from scratch
- ✅ We just need to ADD marketplace pages
- ✅ We just need to ADD components
- ✅ We just need to ADD SSG data fetching

---

### 3. ✅ WASM + tsify Pattern is Standard

**Pattern Found:**
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` - Rust SDK compiled to WASM
- `bin/97_contracts/hive-contract/` - Contract types with `tsify` feature
- TypeScript types are AUTO-GENERATED from Rust via `tsify`

**Example from hive-contract:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct ProcessStats {
    pub pid: u32,
    pub cpu_pct: f64,
    // ...
}
```

**Build Process:**
```bash
# In Rust crate
wasm-pack build --target bundler --out-dir pkg/bundler

# Generates:
pkg/bundler/
├── package.json
├── *.wasm
├── *.js
└── *.d.ts (TypeScript types auto-generated!)
```

**What this means:**
- ✅ Marketplace SDK should be Rust + WASM + tsify
- ✅ Types are auto-generated (no manual sync)
- ✅ Works in both Next.js AND Tauri
- ✅ Single source of truth

---

### 4. ✅ rbee-ui Atomic Design Structure

**Location:** `/home/vince/Projects/llama-orch/frontend/packages/rbee-ui/src/`

**Structure:**
```
rbee-ui/src/
├── atoms/ (90+ components - Button, Badge, Card, etc.)
├── molecules/ (60+ components - StatsGrid, TerminalWindow, etc.)
├── organisms/ (20+ components - domain-specific)
├── templates/ (page sections)
├── pages/ (full pages)
├── marketplace/ (EMPTY - ready for marketplace components!)
├── icons/
├── hooks/
├── providers/
└── utils/
```

**Pattern from Commercial Site:**
```
components/pages/HomePage/
├── HomePage.tsx (DUMB - just renders templates)
├── HomePageProps.tsx (ALL data - perfect for SSG)
└── index.ts

components/templates/HeroTemplate/
├── HeroTemplate.tsx (reusable section)
├── HeroTemplateProps.tsx (props interface)
└── index.ts
```

**What this means:**
- ✅ Marketplace components go in `rbee-ui/src/marketplace/`
- ✅ Follow atomic design: organisms/ → templates/ → pages/
- ✅ Pages are DUMB (all data in Props files)
- ✅ Perfect for SSG (props can be generated at build time)

---

### 5. ✅ React Hooks Pattern for WASM SDKs

**Pattern Found in queen-rbee-react:**
```
packages/queen-rbee-react/
├── src/
│   ├── hooks/
│   │   ├── useQueenSDK.ts (loads WASM SDK)
│   │   ├── useHeartbeat.ts (SSE streaming)
│   │   └── useRhaiScripts.ts (Rhai script management)
│   ├── index.ts (re-exports everything)
│   └── types.ts
└── package.json
```

**Dependencies:**
```json
{
  "dependencies": {
    "@rbee/queen-rbee-sdk": "workspace:*",  // WASM SDK
    "@rbee/sdk-loader": "workspace:*",       // SDK loading utilities
    "@rbee/react-hooks": "workspace:*",      // Shared React hooks
    "@rbee/narration-client": "workspace:*", // SSE narration
    "@tanstack/react-query": "^5.0.0"       // Data fetching
  }
}
```

**What this means:**
- ✅ Create `@rbee/marketplace-react` package
- ✅ Hooks for loading marketplace SDK
- ✅ Hooks for data fetching (models, workers)
- ✅ Use TanStack Query for caching

---

### 6. ✅ Existing Worker Catalog Architecture

**Location:** `bin/25_rbee_hive_crates/worker-catalog/`

**Current Types:**
```rust
pub struct WorkerBinary {
    pub id: String,
    pub worker_type: WorkerType, // CpuLlm, CudaLlm, MetalLlm
    pub platform: Platform,      // Linux, MacOS, Windows
    pub architecture: String,    // x86_64, aarch64
    pub version: String,
    pub path: PathBuf,
    pub size: u64,
    pub status: ArtifactStatus,
    pub added_at: DateTime<Utc>,
}
```

**What this means:**
- ✅ Worker types already exist
- ✅ Can reuse for marketplace
- ❌ Need to decide: WASM bindings OR HTTP client?

---

## Architecture Decisions Needed

Based on my investigation, here are the decisions we need to make:

### Decision 1: Marketplace SDK Implementation

**Option A: Rust + WASM + tsify (RECOMMENDED)**
```
bin/99_shared_crates/marketplace-sdk/
├── src/
│   ├── lib.rs (WASM entry point)
│   ├── huggingface.rs (HuggingFace client)
│   ├── civitai.rs (CivitAI client)
│   ├── worker_catalog.rs (Worker catalog client)
│   └── types.rs (with tsify)
├── Cargo.toml (wasm-bindgen, tsify, reqwest)
└── package.json (wasm-pack build)
```

**Benefits:**
- ✅ Single source of truth (Rust)
- ✅ Auto-generated TypeScript types
- ✅ Works in Next.js AND Tauri
- ✅ Consistent with existing architecture

**Option B: Pure TypeScript**
- ❌ Manual type sync
- ❌ Duplication between Next.js and Tauri
- ❌ Inconsistent with existing architecture

**RECOMMENDATION: Option A**

---

### Decision 2: Worker Catalog Access

**Option A: WASM Bindings**
- Compile worker-catalog to WASM
- Use in marketplace SDK

**Option B: HTTP Client**
- Create thin HTTP client
- Talk to rbee-hive's worker endpoints

**RECOMMENDATION: Option B** because:
- Worker catalog is filesystem-based (not WASM-friendly)
- HTTP client is simpler
- Matches existing job-client pattern

---

### Decision 3: Component Structure

**RECOMMENDED:**
```
rbee-ui/src/marketplace/
├── organisms/
│   ├── ModelCard/
│   ├── WorkerCard/
│   └── MarketplaceGrid/
├── templates/
│   ├── ModelListTemplate/
│   ├── ModelDetailTemplate/
│   └── WorkerListTemplate/
└── pages/
    ├── ModelsPage/
    ├── ModelDetailPage/
    └── WorkersPage/
```

**Pattern:**
- Pages are DUMB (just render templates with props)
- ALL data in Props files (perfect for SSG)
- Templates wrap sections
- Organisms are reusable cards

---

## Updated Implementation Plan

### Phase 1: Marketplace SDK (Rust + WASM)
1. Create `bin/99_shared_crates/marketplace-sdk/`
2. Implement HuggingFace client (Rust)
3. Implement CivitAI client (Rust)
4. Implement Worker HTTP client (Rust)
5. Add tsify for TypeScript types
6. Build with wasm-pack

### Phase 2: Marketplace React Hooks
1. Create `frontend/packages/marketplace-react/`
2. Create `useMarketplaceSDK()` hook
3. Create `useModels()` hook (TanStack Query)
4. Create `useWorkers()` hook (TanStack Query)

### Phase 3: Marketplace Components (rbee-ui)
1. Create organisms in `rbee-ui/src/marketplace/organisms/`
2. Create templates in `rbee-ui/src/marketplace/templates/`
3. Create pages in `rbee-ui/src/marketplace/pages/`

### Phase 4: Next.js Marketplace Site
1. Update `frontend/apps/marketplace/`
2. Add model list page (SSG)
3. Add model detail pages (SSG with dynamic routes)
4. Add worker list page (SSG)
5. Add SEO metadata
6. Generate sitemap

### Phase 5: Keeper Protocol Handler
1. Add `rbee://` protocol registration
2. Add protocol handler in Keeper
3. Add auto-run logic
4. Add marketplace tab in Keeper UI

### Phase 6: Keeper Marketplace UI
1. Add marketplace pages to Keeper
2. Add tab system (if needed)
3. Add worker spawning wizard
4. Integrate with protocol handler

---

## Checklist Updates Required

All 6 checklists need updates:

1. **CHECKLIST_01:** ❌ WRONG - Says create from scratch, should update rbee-ui
2. **CHECKLIST_02:** ❌ WRONG - Says TypeScript, should be Rust + WASM
3. **CHECKLIST_03:** ⚠️ PARTIAL - Marketplace app exists, needs content
4. **CHECKLIST_04:** ⚠️ PARTIAL - Keeper is Tauri, just add protocol
5. **CHECKLIST_05:** ⚠️ PARTIAL - Keeper UI exists, add marketplace tab
6. **CHECKLIST_06:** ✅ OK - Demo plan is fine

---

## Next Steps

1. ✅ Wait for user to answer architecture questions
2. ⏳ Update all 6 checklists based on findings
3. ⏳ Start implementation

**TEAM-400 🐝🎊 - Investigation Complete!**
