# Phase 5: Frontend Package Architecture & Build System

**Analysis Date:** November 2, 2025  
**Scope:** All package.json files across frontend/ and bin/*/ui/  
**Status:** ✅ COMPLETE

---

## Executive Summary

Found **29 package.json files** across the monorepo. The frontend uses **pnpm workspaces** with **Turborepo** for parallel builds. Binary UIs use **React 19 + Vite**, marketing uses **Next.js 15**, and keeper uses **Tauri v2**.

---

## 1. Frontend Apps (`frontend/apps/`)

### 1.1 Commercial Marketing Site

**Location:** `frontend/apps/commercial/package.json`

**Package Name:** `@rbee/commercial`

**Framework:** Next.js 15.5.5

**Build Target:** `next build`

**Dev Server:** `next dev -p 7822`

**Deployment:** OpenNext Cloudflare

**Key Dependencies:**
- `next@15.5.5` — React framework
- `react@19.2.0` — UI library
- `@rbee/ui` — Shared component library (workspace)
- `@opennextjs/cloudflare` — Cloudflare deployment
- `@xyflow/react` — Flow diagrams
- `recharts` — Charts

**Local Dependencies:**
```json
"@rbee/ui": "workspace:*"
```

**Scripts:**
```json
{
  "dev": "next dev -p 7822",
  "build": "next build",
  "deploy": "opennextjs-cloudflare build && opennextjs-cloudflare deploy",
  "preview": "opennextjs-cloudflare build && opennextjs-cloudflare preview"
}
```

---

### 1.2 User Documentation Site

**Location:** `frontend/apps/user-docs/package.json`

**Package Name:** `@rbee/user-docs`

**Framework:** Next.js 15.5.5 + Nextra

**Build Target:** `next build`

**Dev Server:** `next dev -p 7811`

**Deployment:** OpenNext Cloudflare

**Key Dependencies:**
- `next@15.5.5` — React framework
- `nextra@4.6.0` — Documentation framework
- `nextra-theme-docs@4.6.0` — Documentation theme
- `@rbee/ui` — Shared component library (workspace)

**Local Dependencies:**
```json
"@rbee/ui": "workspace:*"
```

**Scripts:**
```json
{
  "dev": "next dev -p 7811",
  "build": "next build",
  "deploy": "opennextjs-cloudflare build && opennextjs-cloudflare deploy"
}
```

---

## 2. Binary UIs (`bin/*/ui/app/`)

### 2.1 Keeper UI (Tauri Desktop App)

**Location:** `bin/00_rbee_keeper/ui/package.json`

**Package Name:** `@rbee/keeper-ui`

**Framework:** Vite + Tauri v2

**Build Target:** `tsc -b && vite build`

**Dev Server:** `vite`

**Key Dependencies:**
- `@tauri-apps/api@2.9.0` — Tauri API
- `react@19.1.1` — UI library
- `react-router-dom@7.6.2` — Routing
- `@rbee/ui` — Shared component library (workspace)
- `@rbee/iframe-bridge` — Cross-iframe communication (workspace)
- `@rbee/narration-client` — SSE client (workspace)
- `@rbee/shared-config` — Shared config (workspace)
- `@tanstack/react-query@5.90.5` — Data fetching

**Local Dependencies:**
```json
{
  "@rbee/ui": "workspace:*",
  "@rbee/iframe-bridge": "workspace:*",
  "@rbee/narration-client": "workspace:*",
  "@rbee/shared-config": "workspace:*"
}
```

**Scripts:**
```json
{
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview"
}
```

**Special Features:**
- Tauri v2 integration
- Desktop app (not web)
- React Compiler enabled

---

### 2.2 Queen UI (Web App)

**Location:** `bin/10_queen_rbee/ui/app/package.json`

**Package Name:** `@rbee/queen-rbee-ui`

**Framework:** Vite + React

**Build Target:** `tsc -b && vite build`

**Dev Server:** `vite --port 7834`

**Key Dependencies:**
- `react@19.1.1` — UI library
- `react-router-dom@7.6.2` — Routing
- `@rbee/queen-rbee-react` — Queen React hooks (workspace)
- `@rbee/ui` — Shared component library (workspace)
- `@rbee/dev-utils` — Dev utilities (workspace)
- `@rbee/iframe-bridge` — Cross-iframe communication (workspace)
- `zustand@5.0.8` — State management

**Local Dependencies:**
```json
{
  "@rbee/queen-rbee-react": "workspace:*",
  "@rbee/ui": "workspace:*",
  "@rbee/dev-utils": "workspace:*",
  "@rbee/iframe-bridge": "workspace:*"
}
```

**Scripts:**
```json
{
  "dev": "vite --port 7834",
  "build": "tsc -b && vite build",
  "preview": "vite preview --port 7834"
}
```

**Special Features:**
- WASM plugins (vite-plugin-wasm, vite-plugin-top-level-await)
- React Compiler enabled
- Rolldown Vite (faster build)

---

### 2.3 Hive UI (Web App)

**Location:** `bin/20_rbee_hive/ui/app/package.json`

**Package Name:** `@rbee/rbee-hive-ui`

**Framework:** Vite + React

**Build Target:** `tsc -b && vite build`

**Dev Server:** `vite`

**Key Dependencies:**
- `react@19.1.1` — UI library
- `@rbee/rbee-hive-react` — Hive React hooks (workspace)
- `@rbee/rbee-hive-sdk` — Hive WASM SDK (workspace)
- `@rbee/ui` — Shared component library (workspace)
- `@rbee/dev-utils` — Dev utilities (workspace)
- `@rbee/shared-config` — Shared config (workspace)
- `@rbee/iframe-bridge` — Cross-iframe communication (workspace)
- `@tanstack/react-query@5.62.14` — Data fetching

**Local Dependencies:**
```json
{
  "@rbee/rbee-hive-react": "workspace:*",
  "@rbee/rbee-hive-sdk": "workspace:*",
  "@rbee/ui": "workspace:*",
  "@rbee/dev-utils": "workspace:*",
  "@rbee/shared-config": "workspace:*",
  "@rbee/iframe-bridge": "workspace:*"
}
```

**Scripts:**
```json
{
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview"
}
```

**Special Features:**
- WASM plugins (vite-plugin-wasm, vite-plugin-top-level-await)
- React Compiler enabled

---

### 2.4 Worker UI (Web App)

**Location:** `bin/30_llm_worker_rbee/ui/app/package.json`

**Package Name:** `@rbee/llm-worker-ui`

**Framework:** Vite + React

**Build Target:** `tsc -b && vite build`

**Dev Server:** `vite`

**Key Dependencies:**
- `react@19.1.1` — UI library
- `@rbee/llm-worker-react` — Worker React hooks (workspace)
- `@tanstack/react-query@5.62.14` — Data fetching

**Local Dependencies:**
```json
{
  "@rbee/llm-worker-react": "workspace:*"
}
```

**Scripts:**
```json
{
  "dev": "vite",
  "build": "tsc -b && vite build",
  "preview": "vite preview"
}
```

**Special Features:**
- Minimal dependencies (no shared UI)
- React Compiler enabled
- Rolldown Vite (faster build)

---

## 3. Shared Packages (`frontend/packages/`)

### 3.1 Component Library

**Location:** `frontend/packages/rbee-ui/package.json`

**Package Name:** `@rbee/ui`

**Type:** React component library

**Build Target:** `tsc` (TypeScript compilation)

**Key Dependencies:**
- `react@19.2.0` — Peer dependency
- `@radix-ui/*` — 20+ Radix UI components
- `lucide-react@0.545.0` — Icons
- `class-variance-authority` — Variant styling
- `tailwind-merge` — Tailwind utilities

**Exports:**
```json
{
  "./styles.css": "./dist/index.css",
  "./atoms": "./src/atoms/index.ts",
  "./molecules": "./src/molecules/index.ts",
  "./organisms": "./src/organisms/index.ts",
  "./templates": "./src/templates/index.ts",
  "./pages": "./src/pages/index.ts",
  "./utils": "./src/utils/index.ts",
  "./icons": "./src/icons/index.ts",
  "./providers": "./src/providers/index.ts"
}
```

**Scripts:**
```json
{
  "build": "pnpm run build:styles && pnpm run build:components",
  "build:styles": "postcss ./src/tokens/globals.css -o ./dist/index.css",
  "build:components": "tsc",
  "dev": "pnpm run dev:styles & pnpm run dev:components & storybook dev -p 6006 --no-open",
  "storybook": "storybook dev -p 6006 --no-open"
}
```

**Testing:**
- Vitest for unit tests
- Storybook for component development
- Playwright for component testing

---

### 3.2 Narration Client

**Location:** `frontend/packages/narration-client/package.json`

**Package Name:** `@rbee/narration-client`

**Type:** SSE client library

**Purpose:** Connect to narration SSE streams

---

### 3.3 Iframe Bridge

**Location:** `frontend/packages/iframe-bridge/package.json`

**Package Name:** `@rbee/iframe-bridge`

**Type:** Cross-iframe communication

**Purpose:** Enable communication between embedded iframes

---

### 3.4 React Hooks

**Location:** `frontend/packages/react-hooks/package.json`

**Package Name:** `@rbee/react-hooks`

**Type:** Shared React hooks

**Purpose:** Reusable hooks across apps

---

### 3.5 SDK Loader

**Location:** `frontend/packages/sdk-loader/package.json`

**Package Name:** `@rbee/sdk-loader`

**Type:** WASM SDK loader

**Purpose:** Load and initialize WASM SDKs

---

### 3.6 Dev Utils

**Location:** `frontend/packages/dev-utils/package.json`

**Package Name:** `@rbee/dev-utils`

**Type:** Development utilities

**Purpose:** Shared dev tooling

---

### 3.7 Shared Config

**Location:** `frontend/packages/shared-config/package.json`

**Package Name:** `@rbee/shared-config`

**Type:** Configuration

**Purpose:** Shared configuration values

---

### 3.8 Config Packages

**Location:** `frontend/packages/{tailwind,typescript,eslint,vite}-config/`

**Package Names:**
- `@repo/tailwind-config`
- `@repo/typescript-config`
- `@repo/eslint-config`
- `@repo/vite-config`

**Type:** Build configuration

**Purpose:** Shared build configs across apps

---

## 4. WASM SDK Packages (`bin/*/ui/packages/*-sdk/`)

### 4.1 Queen SDK

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/`

**Package Name (Rust):** `queen-rbee-sdk`

**Package Name (npm):** `@rbee/queen-rbee-sdk`

**Type:** WASM library (Rust → WASM → TypeScript)

**Build:** `wasm-pack build --target web`

**Key Dependencies (Rust):**
- `job-client` — HTTP client (workspace)
- `operations-contract` — Operation types (workspace)
- `wasm-bindgen` — Rust ↔ JS bindings
- `web-sys` — Browser APIs

**Generated Files:**
- `pkg/web/` — Web target output
- `pkg/bundler/` — Bundler target output

---

### 4.2 Hive SDK

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/`

**Package Name (Rust):** `rbee-hive-sdk`

**Package Name (npm):** `@rbee/rbee-hive-sdk`

**Type:** WASM library (Rust → WASM → TypeScript)

**Build:** `wasm-pack build --target web`

**Key Dependencies (Rust):**
- `job-client` — HTTP client (workspace)
- `operations-contract` — Operation types (workspace)
- `hive-contract` — Hive types (workspace)
- `wasm-bindgen` — Rust ↔ JS bindings
- `web-sys` — Browser APIs

---

### 4.3 Worker SDK

**Location:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/`

**Package Name (Rust):** `llm-worker-sdk`

**Package Name (npm):** `@rbee/llm-worker-sdk`

**Type:** WASM library (Rust → WASM → TypeScript)

**Build:** `wasm-pack build --target web`

**Key Dependencies (Rust):**
- `job-client` — HTTP client (workspace)
- `operations-contract` — Operation types (workspace)
- `worker-contract` — Worker types (workspace)
- `wasm-bindgen` — Rust ↔ JS bindings
- `web-sys` — Browser APIs

---

## 5. React Hooks Packages (`bin/*/ui/packages/*-react/`)

### 5.1 Queen React Hooks

**Location:** `bin/10_queen_rbee/ui/packages/queen-rbee-react/`

**Package Name:** `@rbee/queen-rbee-react`

**Type:** React hooks library

**Purpose:** React hooks for queen-rbee-sdk

**Dependencies:**
- `@rbee/queen-rbee-sdk` — WASM SDK (workspace)
- `@tanstack/react-query` — Data fetching

---

### 5.2 Hive React Hooks

**Location:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/`

**Package Name:** `@rbee/rbee-hive-react`

**Type:** React hooks library

**Purpose:** React hooks for rbee-hive-sdk

**Dependencies:**
- `@rbee/rbee-hive-sdk` — WASM SDK (workspace)
- `@tanstack/react-query` — Data fetching

---

### 5.3 Worker React Hooks

**Location:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-react/`

**Package Name:** `@rbee/llm-worker-react`

**Type:** React hooks library

**Purpose:** React hooks for llm-worker-sdk

**Dependencies:**
- `@rbee/llm-worker-sdk` — WASM SDK (workspace)
- `@tanstack/react-query` — Data fetching

---

## 6. Build System Summary

### Package Manager

**pnpm workspaces** with **Turborepo**

**Workspace Configuration:** `pnpm-workspace.yaml`

```yaml
packages:
  - "frontend/apps/*"
  - "frontend/packages/*"
  - "bin/00_rbee_keeper/ui"
  - "bin/10_queen_rbee/ui/app"
  - "bin/10_queen_rbee/ui/packages/*"
  - "bin/20_rbee_hive/ui/app"
  - "bin/20_rbee_hive/ui/packages/*"
  - "bin/30_llm_worker_rbee/ui/app"
  - "bin/30_llm_worker_rbee/ui/packages/*"
```

---

### Build Orchestration

**Turborepo Configuration:** `turbo.json`

**Parallel Builds:**
- All apps build in parallel
- Shared packages built first (dependencies)
- WASM SDKs built before React hooks

**Build Order:**
1. Shared configs (`@repo/*-config`)
2. WASM SDKs (`*-sdk`)
3. React hooks (`*-react`)
4. Shared packages (`@rbee/*`)
5. Apps (`@rbee/*-ui`)

---

### Framework Distribution

| Framework | Count | Apps |
|-----------|-------|------|
| Vite + React | 4 | queen-ui, hive-ui, worker-ui, keeper-ui |
| Next.js | 2 | commercial, user-docs |
| Tauri v2 | 1 | keeper-ui |

---

### React Version

**All apps use React 19:**
- `react@19.1.1` or `react@19.2.0`
- `react-dom@19.1.1` or `react-dom@19.2.0`

---

### Build Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Vite | 6.0.11 / 7.1.14 | Dev server + bundler |
| TypeScript | 5.9.3 | Type checking |
| Tailwind CSS | 4.1.14 | Styling |
| ESLint | 9.36.0+ | Linting |
| Biome | (root) | Formatting |

---

### Special Vite Plugins

**WASM Support:**
- `vite-plugin-wasm@3.3.0` — WASM module support
- `vite-plugin-top-level-await@1.4.4` — Top-level await

**React Optimization:**
- `babel-plugin-react-compiler@19.1.0-rc.3` — React Compiler

**Faster Builds:**
- `vite@npm:rolldown-vite@7.1.14` — Rolldown (Rust-based bundler)

---

## 7. Dependency Graph (Workspace)

```
Shared Configs (@repo/*)
    ↓
WASM SDKs (*-sdk)
    ↓
React Hooks (*-react)
    ↓
Shared Packages (@rbee/ui, @rbee/iframe-bridge, etc.)
    ↓
Apps (@rbee/*-ui, @rbee/commercial, @rbee/user-docs)
```

---

## 8. Port Configuration

| App | Dev Port | Purpose |
|-----|----------|---------|
| commercial | 7822 | Marketing site |
| user-docs | 7811 | Documentation |
| keeper-ui | N/A | Tauri (no port) |
| queen-ui | 7834 | Queen web UI |
| hive-ui | N/A | Default Vite port |
| worker-ui | N/A | Default Vite port |

---

## 9. Key Findings

### WASM Integration

1. **All binary UIs use WASM SDKs**
   - Rust crates compiled to WASM
   - TypeScript types generated automatically
   - Reuses `job-client` from Rust side

2. **WASM build process**
   - `wasm-pack build --target web`
   - Generates `pkg/` directory
   - Published as npm packages

3. **Vite plugins required**
   - `vite-plugin-wasm` for WASM modules
   - `vite-plugin-top-level-await` for async WASM init

### React Compiler

**All apps use React Compiler (experimental):**
- `babel-plugin-react-compiler@19.1.0-rc.3`
- Automatic optimization of React components
- Reduces re-renders

### Rolldown Vite

**Some apps use Rolldown (Rust-based bundler):**
- `vite@npm:rolldown-vite@7.1.14`
- Faster builds than standard Vite
- Drop-in replacement

### Workspace Dependencies

**Extensive use of workspace references:**
- `workspace:*` for local packages
- Ensures version consistency
- Faster installs (no network)

---

**Analysis Complete:** All 5 phases documented
