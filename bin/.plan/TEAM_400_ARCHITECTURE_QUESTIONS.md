# 🎉 TEAM-400 Architecture Questions 🎉

**Date:** 2025-11-04  
**Status:** 🔍 INVESTIGATING

---

## Questions About Marketplace Architecture

### 1. Marketplace SDK - Rust vs TypeScript?

**What I found:**
- `queen-rbee-sdk` uses **Rust + WASM + tsify** pattern
- Contract crates (like `hive-contract`) use `tsify` feature to auto-generate TypeScript types
- Pattern: Define types in Rust → tsify generates TypeScript → No manual type sync

**Question:**
Should the marketplace SDK follow the same pattern?

**Option A: Rust + WASM + tsify (like queen-rbee-sdk)**
```
bin/99_shared_crates/marketplace-sdk/
├── src/
│   ├── lib.rs (WASM entry point)
│   ├── huggingface.rs (HuggingFace client)
│   ├── civitai.rs (CivitAI client)
│   └── types.rs (with #[cfg_attr(feature = "wasm", derive(Tsify))])
├── Cargo.toml (with tsify, wasm-bindgen)
└── package.json (wasm-pack build)
```

**Option B: Pure TypeScript (as in current checklists)**
```
frontend/packages/marketplace-sdk/
├── src/
│   ├── HuggingFaceClient.ts
│   ├── CivitAIClient.ts
│   └── types.ts
└── package.json
```

**My recommendation:** Option A (Rust + WASM + tsify) because:
- ✅ Single source of truth (Rust)
- ✅ Compiler-verified types
- ✅ Can reuse in both Next.js AND Tauri
- ✅ Consistent with existing architecture
- ✅ No manual type sync

**Is this correct?**
YES

---

### 2. Marketplace Components - Where do they live?

**What I found:**
- `frontend/packages/rbee-ui/src/marketplace/` exists but is EMPTY
- `frontend/packages/rbee-ui/src/` has atomic design structure:
  - `atoms/` - Shared globally (Button, Badge, Card, etc.)
  - `molecules/` - Shared globally (StatsGrid, TerminalWindow, etc.)
  - `organisms/` - Can have domain-specific folders
  - `templates/` - Page sections
  - `pages/` - Full pages

**Question:**
Should marketplace components follow this structure?

```
frontend/packages/rbee-ui/src/marketplace/
├── organisms/
│   ├── ModelCard/
│   │   ├── ModelCard.tsx
│   │   ├── ModelCard.stories.tsx
│   │   └── index.ts
│   └── WorkerCard/
│       ├── WorkerCard.tsx
│       ├── WorkerCard.stories.tsx
│       └── index.ts
├── templates/
│   ├── ModelListTemplate/
│   │   ├── ModelListTemplate.tsx
│   │   ├── ModelListTemplateProps.tsx
│   │   └── index.ts
│   └── ModelDetailTemplate/
│       ├── ModelDetailTemplate.tsx
│       ├── ModelDetailTemplateProps.tsx
│       └── index.ts
└── pages/
    ├── ModelsPage/
    │   ├── ModelsPage.tsx
    │   ├── ModelsPageProps.tsx
    │   └── index.ts
    └── ModelDetailPage/
        ├── ModelDetailPage.tsx
        ├── ModelDetailPageProps.tsx
        └── index.ts
```

**Pattern from commercial site:**
- Pages are DUMB (just render templates with props)
- ALL data is in Props files (perfect for SSG)
- Templates wrap sections with TemplateContainer
- Organisms are reusable card/section components

**Is this the right structure?**
YES

---

### 3. Next.js Marketplace App - Where does it live?

**What I found:**
- `frontend/apps/commercial/` - Commercial marketing site
- `frontend/apps/marketplace/` - **EXISTS!** (I need to check what's in it)
- `frontend/apps/user-docs/` - Documentation site

**Question:**
Should the marketplace Next.js app live in `frontend/apps/marketplace/`?

**Expected structure:**
```
frontend/apps/marketplace/
├── app/
│   ├── page.tsx (home - model list)
│   ├── models/
│   │   └── [modelId]/
│   │       └── page.tsx (model detail)
│   └── workers/
│       └── [workerId]/
│           └── page.tsx (worker detail)
├── components/
│   └── (app-specific components if needed)
├── public/
├── package.json
└── next.config.js
```

**Is this correct?**
YES

---

### 4. Tauri Integration - Keeper or separate app?

**What I found:**
- `bin/00_rbee_keeper/` - Existing Keeper Tauri app
- Checklist 04 says "Keeper is already a Tauri app! Just need to add protocol handler"

**Question:**
Should the marketplace UI be integrated into the existing Keeper app, or should it be a separate Tauri app?

**Option A: Integrate into Keeper**
```
bin/00_rbee_keeper/
├── src-tauri/
│   ├── src/
│   │   ├── main.rs
│   │   ├── protocol.rs (NEW - rbee:// handler)
│   │   └── auto_run.rs (NEW - auto-run logic)
│   └── Cargo.toml
└── ui/
    ├── src/
    │   ├── pages/
    │   │   ├── Dashboard.tsx
    │   │   ├── Marketplace.tsx (NEW - browse models)
    │   │   └── Workers.tsx
    │   └── App.tsx
    └── package.json
```

**Option B: Separate marketplace app**
```
bin/01_marketplace_keeper/
├── src-tauri/
└── ui/
```

**My recommendation:** Option A (integrate into Keeper) because:Y

- ✅ Single app for users
- ✅ Reuse existing Keeper infrastructure
- ✅ Simpler distribution

**Is this correct?**
YES

---

### 5. Shared Business Logic - Rust crate location?

**Question:**
If we create a Rust crate for marketplace business logic (to share between Next.js and Tauri), where should it live?

**Option A: In bin/99_shared_crates/**
```
bin/99_shared_crates/marketplace-sdk/
├── src/
│   ├── lib.rs
│   ├── huggingface.rs
│   ├── civitai.rs
│   └── types.rs
├── Cargo.toml
└── package.json (wasm-pack)
```

**Option B: In bin/97_contracts/**
```
bin/97_contracts/marketplace-contract/
├── src/
│   ├── lib.rs
│   └── types.rs
└── Cargo.toml
```

**My recommendation:** Option A (shared_crates) because:
- ✅ It's not just types (has HTTP client logic)
- ✅ Contracts are for pure types/protocols
- ✅ Shared crates are for reusable logic

**Is this correct?**
HOLD UP HOLD UP HOLD UUUUUUUPPP!!!!

the marketplace-SDK in question 1 and the "Shared business logic" in question 5 are the same crate, right?
It should be... there should not be a difference between those two crates...
I think that we should consolidate these crates

---

### 6. Worker Catalog - Use existing or create new?

**What I found from memory:**
- `bin/25_rbee_hive_crates/worker-catalog/` exists
- WorkerBinary type: id, worker_type, platform, architecture, version, etc.
- WorkerCatalog is READ ONLY from Hive

**Question:**
Should the marketplace SDK:
- **Option A:** Use the existing worker-catalog crate directly?
- **Option B:** Create a new marketplace-specific worker client?

**My recommendation:** Option A (use existing) because:
- ✅ Single source of truth
- ✅ Already has all the types we need
- ✅ No duplication

**But how do we access it from Next.js?**
- Create WASM bindings for worker-catalog?
- Or create a thin HTTP client that talks to rbee-hive's worker endpoints?

**What's the right approach?**
YES
But to be clear: the worker and model catalog are desktop crates originally
If there are shared types between the worker and model catalog that we need to use in the marketplace.
then we need to make a /home/vince/Projects/llama-orch/bin/97_contracts/catelog-contract

---

### 7. Installation Detection - How does it work?

**Question:**
The marketplace needs to detect if Keeper is installed to show the right buttons:
- If installed: "Run with rbee" (opens `rbee://` protocol)
- If not installed: "Download Keeper" (download link)

**How should this work in Next.js (SSG)?**

**Option A: Client-side detection**
```tsx
'use client'
import { useEffect, useState } from 'react'

function useKeeperInstalled() {
  const [installed, setInstalled] = useState(false)
  
  useEffect(() => {
    // Try to open rbee:// protocol
    // If it works, Keeper is installed
    // If it fails, show download button
  }, [])
  
  return installed
}
```

**Option B: Server-side detection** (not possible with SSG)

**My recommendation:** Option A (client-side) because:
- ✅ Works with SSG
- ✅ Can detect on user's machine
- ✅ Progressive enhancement

**Is this correct? Any better approach?**
YES

---

## Summary of Recommendations

1. **Marketplace SDK:** Rust + WASM + tsify (like queen-rbee-sdk)
2. **Components:** Live in `rbee-ui/src/marketplace/` with atomic design structure
3. **Next.js app:** Lives in `frontend/apps/marketplace/`
4. **Tauri:** Integrate into existing Keeper app
5. **Shared logic:** Lives in `bin/99_shared_crates/marketplace-sdk/`
6. **Worker catalog:** Use existing worker-catalog crate (need to decide: WASM or HTTP client)
7. **Installation detection:** Client-side JavaScript

---

## Please Answer

For each question, please confirm or correct my understanding. If I'm wrong, please explain the correct approach!

**TEAM-400 🐝🎊**
