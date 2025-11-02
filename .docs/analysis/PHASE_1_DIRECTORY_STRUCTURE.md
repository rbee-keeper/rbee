# Phase 1: Directory Structure Analysis

**Analysis Date:** November 2, 2025  
**Scope:** Complete directory and Cargo.toml inventory  
**Status:** ✅ COMPLETE

---

## Executive Summary

Validated the numbered layer system (00, 10, 15, 20, 25, 30, 96, 97, 98, 99) across `/bin`, `/contracts`, `/xtask`, and `/frontend` directories. Found **40 Cargo.toml files** in `/bin` alone, confirming the documented structure.

---

## 1. `/bin` Directory Structure

### Top-Level Organization

```
/bin/
├── 00_rbee_keeper/          # Layer 00: CLI + GUI
├── 10_queen_rbee/           # Layer 10: Orchestrator daemon
├── 15_queen_rbee_crates/    # Layer 15: Queen-specific logic
├── 20_rbee_hive/            # Layer 20: Worker lifecycle daemon
├── 25_rbee_hive_crates/     # Layer 25: Hive-specific logic
├── 30_llm_worker_rbee/      # Layer 30: LLM inference daemon
├── 80-hono-worker-catalog/  # Layer 80: Experimental (Hono catalog)
├── 96_lifecycle/            # Layer 96: Daemon lifecycle management
├── 97_contracts/            # Layer 97: Type-safe contracts
├── 98_security_crates/      # Layer 98: Security primitives
└── 99_shared_crates/        # Layer 99: Cross-service utilities
```

### Layer 00: CLI Tool (`00_rbee_keeper/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/00_rbee_keeper/`  
**Items:** 93 files/directories

**Cargo.toml:**
```toml
[package]
name = "rbee-keeper"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[[bin]]
name = "rbee-keeper"
path = "src/main.rs"

[lib]
name = "rbee_keeper"
path = "src/lib.rs"
```

**Type:** Binary + Library (dual-mode)  
**Purpose:** CLI tool with Tauri GUI support  
**Key Dependencies:**
- `clap` — CLI parsing
- `tauri` — GUI framework (v2)
- `lifecycle-local` — Local daemon management
- `lifecycle-ssh` — Remote daemon management
- `job-client` — HTTP client for job submission
- `observability-narration-core` — Real-time feedback

**UI:** `bin/00_rbee_keeper/ui/` (Tauri app)

---

### Layer 10: Orchestrator (`10_queen_rbee/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/10_queen_rbee/`  
**Items:** 81 files/directories

**Cargo.toml:**
```toml
[package]
name = "queen-rbee"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[[bin]]
name = "queen-rbee"
path = "src/main.rs"
```

**Type:** Binary (HTTP daemon)  
**Purpose:** System orchestrator and job router  
**Default Port:** 7833  
**Key Dependencies:**
- `axum` — HTTP framework
- `job-server` — Job registry
- `telemetry-registry` — Hive + worker telemetry
- `hive-contract` — Hive API types
- `worker-contract` — Worker API types
- `operations-contract` — Operation types

**UI:** `bin/10_queen_rbee/ui/app/` (React app)  
**SDK:** `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/` (Rust → WASM)

---

### Layer 15: Queen-Specific Crates (`15_queen_rbee_crates/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/15_queen_rbee_crates/`  
**Items:** 20 files/directories

**Crates:**

#### 1. `telemetry-registry/`
```toml
[package]
name = "queen-rbee-telemetry-registry"
```
**Type:** Library  
**Purpose:** Hive + worker telemetry storage (RAM)

#### 2. `scheduler/`
```toml
[package]
name = "queen-rbee-scheduler"
```
**Type:** Library  
**Purpose:** Job scheduler (M2 - Rhai scripting stub)

#### 3. `rbee-openai-adapter/`
```toml
[package]
name = "rbee-openai-adapter"
```
**Type:** Library  
**Purpose:** OpenAI-compatible API adapter

---

### Layer 20: Worker Lifecycle (`20_rbee_hive/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/20_rbee_hive/`  
**Items:** 92 files/directories

**Cargo.toml:**
```toml
[package]
name = "rbee-hive"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[[bin]]
name = "rbee-hive"
path = "src/main.rs"

[lib]
name = "rbee_hive"
path = "src/lib.rs"
```

**Type:** Binary + Library  
**Purpose:** Worker lifecycle manager (runs on GPU machine)  
**Default Port:** 7835  
**Key Dependencies:**
- `axum` — HTTP framework
- `job-server` — Job registry
- `model-catalog` — Model storage
- `worker-catalog` — Worker binary management
- `device-detection` — GPU/CPU detection
- `lifecycle-local` — Local worker management

**UI:** `bin/20_rbee_hive/ui/app/` (React app)  
**SDK:** `bin/20_rbee_hive/ui/packages/rbee-hive-sdk/` (Rust → WASM)

---

### Layer 25: Hive-Specific Crates (`25_rbee_hive_crates/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/25_rbee_hive_crates/`  
**Items:** 45 files/directories

**Crates:**

#### 1. `artifact-catalog/`
```toml
[package]
name = "rbee-hive-artifact-catalog"
```
**Type:** Library  
**Purpose:** Shared catalog abstraction

#### 2. `model-catalog/`
```toml
[package]
name = "rbee-hive-model-catalog"
```
**Type:** Library  
**Purpose:** Model storage (uses artifact-catalog)

#### 3. `worker-catalog/`
```toml
[package]
name = "rbee-hive-worker-catalog"
```
**Type:** Library  
**Purpose:** Worker binary management (READ ONLY from hive)

#### 4. `model-provisioner/`
```toml
[package]
name = "rbee-hive-model-provisioner"
```
**Type:** Library  
**Purpose:** HuggingFace downloads

#### 5. `device-detection/`
```toml
[package]
name = "rbee-hive-device-detection"
```
**Type:** Library  
**Purpose:** GPU/CPU detection

#### 6. `monitor/`
```toml
[package]
name = "rbee-hive-monitor"
```
**Type:** Library  
**Purpose:** Worker telemetry (cgroup, nvidia-smi)

#### 7. `download-tracker/`
```toml
[package]
name = "rbee-hive-download-tracker"
```
**Type:** Library  
**Purpose:** Download progress tracking

#### 8. `vram-checker/`
```toml
[package]
name = "rbee-hive-vram-checker"
```
**Type:** Library  
**Purpose:** VRAM availability checking

#### 9. `model-preloader/`
```toml
[package]
name = "rbee-hive-model-preloader"
```
**Type:** Library  
**Purpose:** Model preloading optimization

---

### Layer 30: LLM Worker (`30_llm_worker_rbee/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/30_llm_worker_rbee/`  
**Items:** 94 files/directories

**Cargo.toml:**
```toml
[package]
name = "llm-worker-rbee"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"

[[bin]]
name = "llm-worker-rbee"
path = "src/main.rs"

[[bin]]
name = "llm-worker-rbee-cpu"
path = "src/bin/cpu.rs"
required-features = ["cpu"]

[[bin]]
name = "llm-worker-rbee-cuda"
path = "src/bin/cuda.rs"
required-features = ["cuda"]

[[bin]]
name = "llm-worker-rbee-metal"
path = "src/bin/metal.rs"
required-features = ["metal"]

[lib]
name = "llm_worker_rbee"
path = "src/lib.rs"
```

**Type:** Binary + Library (multi-backend)  
**Purpose:** LLM inference executor  
**Default Ports:** 9300+ (dynamic allocation)  
**Key Dependencies:**
- `candle-core` — Tensor operations
- `candle-nn` — Neural network functions
- `candle-transformers` — Model implementations
- `tokenizers` — HuggingFace tokenizers
- `job-server` — Job registry
- `worker-contract` — Worker API types

**Backend Variants:**
- `llm-worker-rbee-cpu` — CPU backend
- `llm-worker-rbee-cuda` — NVIDIA CUDA backend
- `llm-worker-rbee-metal` — Apple Metal backend

**UI:** `bin/30_llm_worker_rbee/ui/app/` (React app)  
**SDK:** `bin/30_llm_worker_rbee/ui/packages/llm-worker-sdk/` (Rust → WASM)

---

### Layer 80: Experimental (`80-hono-worker-catalog/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/80-hono-worker-catalog/`  
**Items:** 16 files/directories

**Note:** Experimental Hono-based worker catalog (not in main architecture)

---

### Layer 96: Lifecycle Management (`96_lifecycle/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/96_lifecycle/`  
**Items:** 52 files/directories

**Crates:**

#### 1. `lifecycle-local/`
```toml
[package]
name = "lifecycle-local"
```
**Type:** Library  
**Purpose:** Local daemon management (ALWAYS monitored)

#### 2. `lifecycle-ssh/`
```toml
[package]
name = "lifecycle-ssh"
```
**Type:** Library  
**Purpose:** SSH daemon management (remote only)

#### 3. `lifecycle-shared/`
```toml
[package]
name = "lifecycle-shared"
```
**Type:** Library  
**Purpose:** Shared types and utilities

#### 4. `health-poll/`
```toml
[package]
name = "health-poll"
```
**Type:** Library  
**Purpose:** HTTP health polling utility

---

### Layer 97: Type Contracts (`97_contracts/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/97_contracts/`  
**Items:** 44 files/directories

**Crates:**

#### 1. `operations-contract/`
```toml
[package]
name = "operations-contract"
```
**Type:** Library  
**Purpose:** Operation types (queen ↔ hive)

#### 2. `hive-contract/`
```toml
[package]
name = "hive-contract"
```
**Type:** Library  
**Purpose:** Hive API types

#### 3. `worker-contract/`
```toml
[package]
name = "worker-contract"
```
**Type:** Library  
**Purpose:** Worker API types

#### 4. `jobs-contract/`
```toml
[package]
name = "jobs-contract"
```
**Type:** Library  
**Purpose:** Jobs HTTP API contract

#### 5. `shared-contract/`
```toml
[package]
name = "shared-contract"
```
**Type:** Library  
**Purpose:** Shared types (workers + hives)

#### 6. `keeper-config-contract/`
```toml
[package]
name = "keeper-config-contract"
```
**Type:** Library  
**Purpose:** Keeper configuration schema

---

### Layer 98: Security (`98_security_crates/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/98_security_crates/`  
**Items:** 69 files/directories

**Crates:**

#### 1. `auth-min/`
```toml
[package]
name = "auth-min"
```
**Type:** Library  
**Purpose:** Authentication primitives (timing-safe)

#### 2. `audit-logging/`
```toml
[package]
name = "audit-logging"
```
**Type:** Library  
**Purpose:** Immutable audit trail (GDPR)

#### 3. `input-validation/`
```toml
[package]
name = "input-validation"
```
**Type:** Library  
**Purpose:** Injection prevention

#### 4. `secrets-management/`
```toml
[package]
name = "secrets-management"
```
**Type:** Library  
**Purpose:** Credential handling (file-based)

#### 5. `deadline-propagation/`
```toml
[package]
name = "deadline-propagation"
```
**Type:** Library  
**Purpose:** Timeout enforcement

#### 6. `jwt-guardian/`
```toml
[package]
name = "jwt-guardian"
```
**Type:** Library  
**Purpose:** JWT validation

---

### Layer 99: Shared Utilities (`99_shared_crates/`)

**Directory:** `/home/vince/Projects/llama-orch/bin/99_shared_crates/`  
**Items:** 90 files/directories

**Crates:**

#### 1. `narration-core/`
```toml
[package]
name = "observability-narration-core"
```
**Type:** Library  
**Purpose:** Observability framework (SSE streaming)

#### 2. `job-server/`
```toml
[package]
name = "job-server"
```
**Type:** Library  
**Purpose:** Job-based architecture pattern

#### 3. `job-client/`
```toml
[package]
name = "job-client"
```
**Type:** Library  
**Purpose:** HTTP client for job submission

#### 4. `timeout-enforcer/`
```toml
[package]
name = "timeout-enforcer"
```
**Type:** Library  
**Purpose:** Hard timeout enforcement

#### 5. `timeout-enforcer-macros/`
```toml
[package]
name = "timeout-enforcer-macros"
```
**Type:** Procedural macro  
**Purpose:** Timeout macro support

#### 6. `heartbeat-registry/`
```toml
[package]
name = "heartbeat-registry"
```
**Type:** Library  
**Purpose:** Generic heartbeat registry

#### 7. `auto-update/`
```toml
[package]
name = "auto-update"
```
**Type:** Library  
**Purpose:** Dependency-aware auto-update

#### 8. `ssh-config-parser/`
```toml
[package]
name = "ssh-config-parser"
```
**Type:** Library  
**Purpose:** SSH config parsing

---

## 2. `/contracts` Directory Structure

**Directory:** `/home/vince/Projects/llama-orch/contracts/`  
**Items:** 5 subdirectories

```
/contracts/
├── api-types/          # Legacy API type definitions
├── config-schema/      # Configuration schemas
├── openapi/            # OpenAPI specifications
├── pacts/              # Contract testing (Pact files)
└── schemas/            # JSON schemas
```

### `api-types/`
```toml
[package]
name = "contracts-api-types"
```
**Type:** Library  
**Purpose:** Legacy API type definitions

### `config-schema/`
```toml
[package]
name = "config-schema"
```
**Type:** Library  
**Purpose:** Configuration schemas

---

## 3. `/xtask` Directory Structure

**Directory:** `/home/vince/Projects/llama-orch/xtask/`  
**Items:** 4 items (Cargo.toml, README.md, src/, tests/)

**Cargo.toml:**
```toml
[package]
name = "xtask"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "xtask"
path = "src/main.rs"

[lib]
name = "xtask"
path = "src/lib.rs"
```

**Type:** Binary + Library  
**Purpose:** Build automation and testing harness

**Modules:**
- `src/tasks/` — Build tasks
- `src/e2e/` — End-to-end tests
- `src/integration/` — Integration tests
- `src/chaos/` — Chaos testing

---

## 4. `/frontend` Directory Structure

**Directory:** `/home/vince/Projects/llama-orch/frontend/`  
**Items:** 8 items

```
/frontend/
├── apps/               # Frontend applications
│   ├── commercial/     # Marketing site (Next.js)
│   └── user-docs/      # Documentation site (Vue)
├── packages/           # Shared packages
│   ├── rbee-ui/        # Component library
│   ├── narration-client/  # SSE client
│   ├── iframe-bridge/  # Cross-iframe communication
│   └── ... (9 more)
└── tools/              # Frontend tooling
```

**Note:** Only 1 Rust file in frontend: `shared-constants.rs`

---

## Summary Statistics

### Cargo.toml Count by Layer

| Layer | Directory | Crates | Type |
|-------|-----------|--------|------|
| 00 | `00_rbee_keeper/` | 1 | Binary + Library |
| 10 | `10_queen_rbee/` | 1 | Binary |
| 15 | `15_queen_rbee_crates/` | 3 | Libraries |
| 20 | `20_rbee_hive/` | 1 | Binary + Library |
| 25 | `25_rbee_hive_crates/` | 9 | Libraries |
| 30 | `30_llm_worker_rbee/` | 1 | Binary + Library (multi-backend) |
| 96 | `96_lifecycle/` | 4 | Libraries |
| 97 | `97_contracts/` | 6 | Libraries |
| 98 | `98_security_crates/` | 6 | Libraries |
| 99 | `99_shared_crates/` | 8 | Libraries (1 proc macro) |
| - | `contracts/` | 2 | Libraries |
| - | `xtask/` | 1 | Binary + Library |
| **TOTAL** | | **43** | |

### Validation Status

✅ **Numbered layer system confirmed** (00, 10, 15, 20, 25, 30, 96, 97, 98, 99)  
✅ **4 main binaries** (rbee-keeper, queen-rbee, rbee-hive, llm-worker-rbee)  
✅ **39 supporting crates** organized by layer  
✅ **All Cargo.toml files located and inventoried**

---

**Next Phase:** [PHASE_2_DEPENDENCY_GRAPH.md](./PHASE_2_DEPENDENCY_GRAPH.md)
