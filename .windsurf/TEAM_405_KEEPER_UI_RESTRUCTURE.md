# TEAM-405: Keeper UI Restructure Complete

**Date:** Nov 4, 2025  
**Status:** ✅ COMPLETE  
**Mission:** Restructure rbee-keeper UI to follow the same pattern as other packages (app + react package)

---

## 🎯 What Was Done

### 1. Directory Restructure

**Before:**
```
bin/00_rbee_keeper/ui/
├── src/
├── package.json
├── vite.config.ts
└── ... (all app files at root)
```

**After:**
```
bin/00_rbee_keeper/ui/
├── app/                          # Main Tauri app
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
└── packages/
    └── marketplace-react/        # React hooks package
        ├── src/
        │   ├── hooks/
        │   │   └── useMarketplaceModels.ts
        │   └── index.ts
        ├── package.json
        ├── tsconfig.json
        └── README.md
```

### 2. Created marketplace-react Package

**File:** `bin/00_rbee_keeper/ui/packages/marketplace-react/`

**Purpose:** Provides React hooks for marketplace operations (mirrors pattern from queen-rbee-react, rbee-hive-react, llm-worker-react)

**Package Structure:**
```typescript
// package.json
{
  "name": "@rbee/marketplace-react",
  "dependencies": {
    "@rbee/marketplace-sdk": "workspace:*",
    "@rbee/narration-client": "workspace:*",
    "@tanstack/react-query": "^5.0.0"
  }
}

// src/index.ts
export { useMarketplaceModels } from './hooks/useMarketplaceModels'
export type { UseMarketplaceModelsResult } from './hooks/useMarketplaceModels'
export type { Model, ModelSource } from '@rbee/marketplace-sdk'

// src/hooks/useMarketplaceModels.ts
export function useMarketplaceModels(options: UseMarketplaceModelsOptions): UseMarketplaceModelsResult {
  const { query, limit = 50, enabled = true } = options

  const { data: models, isLoading, error, refetch } = useQuery({
    queryKey: ['marketplace', 'models', query, limit],
    queryFn: async () => {
      const result = await invoke<Model[]>('marketplace_list_models', {
        query: query || null,
        limit,
      })
      return result
    },
    enabled,
    staleTime: 5 * 60 * 1000,
    retry: 2,
    retryDelay: 1000,
  })

  return { models: models || [], isLoading, error: error as Error | null, refetch }
}
```

### 3. Updated App Package

**Changes:**
- Package name: `@rbee/keeper-ui` → `@rbee/keeper-ui-app`
- Added dependency: `@rbee/marketplace-react`
- Updated imports in `MarketplaceLlmModels.tsx` to use `useMarketplaceModels` hook

**Before:**
```tsx
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";

const { data: models = [], isLoading, error } = useQuery({
  queryKey: ["marketplace", "llm-models", debouncedQuery],
  queryFn: async () => {
    const result = await invoke<Model[]>("marketplace_list_models", {
      query: debouncedQuery || null,
      limit: 50,
    });
    return result;
  },
  staleTime: 5 * 60 * 1000,
});
```

**After:**
```tsx
import { useMarketplaceModels } from "@rbee/marketplace-react";

const { models, isLoading, error } = useMarketplaceModels({
  query: debouncedQuery || undefined,
  limit: 50,
});
```

### 4. Updated Configuration Files

#### pnpm-workspace.yaml
```yaml
packages:
  # ... other packages ...
  - bin/00_rbee_keeper/ui/app                      # ✅ NEW
  - bin/00_rbee_keeper/ui/packages/marketplace-react  # ✅ NEW
  # - bin/00_rbee_keeper/ui                        # ❌ REMOVED
```

#### package.json (root)
```json
{
  "scripts": {
    "dev:keeper": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app",
    "dev:marketplace": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app",
    "dev:queen": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app --filter=@rbee/queen-rbee-ui ...",
    "dev:hive": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app --filter=@rbee/rbee-hive-ui ...",
    "dev:worker": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app --filter=@rbee/llm-worker-ui ...",
    "dev:product": "turbo dev --filter=@rbee/ui --filter=@rbee/marketplace-react --filter=@rbee/keeper-ui-app ..."
  }
}
```

#### tauri.conf.json
```json
{
  "build": {
    "frontendDist": "../ui/app/dist",
    "devUrl": "http://localhost:5173",
    "beforeDevCommand": "cargo tauri-typegen generate && cd ../ui/app && npm run dev",
    "beforeBuildCommand": "cargo tauri-typegen generate && cd ../ui/app && npm run build"
  },
  "plugins": {
    "tauri-typegen": {
      "output_path": "./ui/app/src/generated"
    }
  }
}
```

#### src/tauri_commands.rs
```rust
builder
    .export(Typescript::default(), "ui/app/src/generated/bindings.ts")
    .expect("Failed to export typescript bindings");
```

---

## 📊 Pattern Consistency

Now rbee-keeper follows the **EXACT SAME PATTERN** as other packages:

### Queen
```
bin/10_queen_rbee/ui/
├── app/                    # Main app
└── packages/
    ├── queen-rbee-sdk/     # WASM SDK
    └── queen-rbee-react/   # React hooks
```

### Hive
```
bin/20_rbee_hive/ui/
├── app/                    # Main app
└── packages/
    ├── rbee-hive-sdk/      # WASM SDK
    └── rbee-hive-react/    # React hooks
```

### Worker
```
bin/30_llm_worker_rbee/ui/
├── app/                    # Main app
└── packages/
    ├── llm-worker-sdk/     # WASM SDK
    └── llm-worker-react/   # React hooks
```

### Keeper (NOW)
```
bin/00_rbee_keeper/ui/
├── app/                    # Main app
└── packages/
    └── marketplace-react/  # React hooks (no SDK - uses Tauri commands)
```

**Note:** Keeper doesn't have a WASM SDK because it uses Tauri commands directly. The marketplace-sdk is native Rust (not WASM) and is called from Tauri commands.

---

## 🔄 Data Flow

### Before (Direct Tauri Invoke)
```
React Component
    ↓
invoke("marketplace_list_models")
    ↓
Tauri command (Rust)
    ↓
marketplace-sdk (Native Rust)
    ↓
HuggingFace API
```

### After (React Hook Abstraction)
```
React Component
    ↓
useMarketplaceModels() hook
    ↓
invoke("marketplace_list_models")
    ↓
Tauri command (Rust)
    ↓
marketplace-sdk (Native Rust)
    ↓
HuggingFace API
```

**Benefits:**
- ✅ Consistent pattern across all packages
- ✅ React Query integration (caching, retry, stale data management)
- ✅ Type safety (TypeScript types from marketplace-sdk)
- ✅ Reusable hooks
- ✅ Easier testing

---

## ✅ Verification Checklist

- [x] Directory structure matches other packages
- [x] marketplace-react package created
- [x] package.json updated (app renamed to keeper-ui-app)
- [x] pnpm-workspace.yaml updated
- [x] Root package.json dev scripts updated
- [x] tauri.conf.json paths updated
- [x] tauri_commands.rs export path updated
- [x] MarketplaceLlmModels.tsx updated to use hook
- [x] README.md created for marketplace-react

---

## 🚀 Next Steps

### To Complete Setup
```bash
# Install dependencies
pnpm install

# Build marketplace-react package
cd bin/00_rbee_keeper/ui/packages/marketplace-react
pnpm build

# Test the app
cd ../app
pnpm dev
```

### Future Enhancements
1. Add more hooks to marketplace-react:
   - `useMarketplaceImageModels` (CivitAI)
   - `useMarketplaceWorkers` (Worker Catalog)
2. Add mutation hooks:
   - `useDownloadModel`
   - `useInstallWorker`

---

## 📝 Files Changed

**Created:**
- `bin/00_rbee_keeper/ui/packages/marketplace-react/package.json`
- `bin/00_rbee_keeper/ui/packages/marketplace-react/tsconfig.json`
- `bin/00_rbee_keeper/ui/packages/marketplace-react/.gitignore`
- `bin/00_rbee_keeper/ui/packages/marketplace-react/README.md`
- `bin/00_rbee_keeper/ui/packages/marketplace-react/src/index.ts`
- `bin/00_rbee_keeper/ui/packages/marketplace-react/src/hooks/useMarketplaceModels.ts`

**Modified:**
- `pnpm-workspace.yaml` - Updated workspace paths
- `package.json` (root) - Updated dev scripts
- `bin/00_rbee_keeper/ui/app/package.json` - Renamed package, added marketplace-react dependency
- `bin/00_rbee_keeper/tauri.conf.json` - Updated paths
- `bin/00_rbee_keeper/src/tauri_commands.rs` - Updated export path
- `bin/00_rbee_keeper/ui/app/src/pages/MarketplaceLlmModels.tsx` - Use hook instead of direct invoke

**Moved:**
- All files from `bin/00_rbee_keeper/ui/*` → `bin/00_rbee_keeper/ui/app/*`

---

## 🎓 Key Learnings

### 1. Consistent Package Structure
All rbee packages now follow the same pattern:
- `app/` - Main application
- `packages/` - Reusable packages (SDK, React hooks)

### 2. React Hooks Layer
The React hooks layer provides:
- React Query integration
- Type safety
- Reusable abstractions
- Consistent API across all packages

### 3. Separation of Concerns
- **marketplace-sdk** (Rust) - Native API client
- **marketplace-react** (TypeScript) - React hooks
- **keeper-ui-app** (TypeScript) - Main application

---

**TEAM-405 signing off. Keeper UI now follows the same pattern as all other packages!**
