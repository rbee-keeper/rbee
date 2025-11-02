# Phase 2: Shared Crates Dependency Graph

**Analysis Date:** November 2, 2025  
**Scope:** Dependency relationships for `99_shared_crates/`  
**Status:** ✅ COMPLETE

---

## Executive Summary

Mapped dependency relationships for 8 shared crates across 16 binaries. **`narration-core` is the most widely used** (16 dependents), followed by `job-server` (10 dependents) and `job-client` (7 dependents).

---

## 1. Narration-Core Dependencies

### Dependents (16 total)

**Main Binaries (4):**
1. `rbee-keeper` — CLI narration
2. `queen-rbee` — Job routing narration
3. `rbee-hive` — Worker lifecycle narration
4. `llm-worker-rbee` — Inference narration

**Queen Crates (2):**
5. `rbee-openai-adapter` — OpenAI adapter narration
6. `scheduler` — Scheduler narration

**Hive Crates (2):**
7. `artifact-catalog` — Catalog operation narration
8. `model-provisioner` — Download narration

**Lifecycle Crates (4):**
9. `lifecycle-local` — Local daemon lifecycle narration
10. `lifecycle-ssh` — SSH operation narration
11. `lifecycle-shared` — Shared lifecycle narration
12. `health-poll` — Health check narration

**Shared Crates (3):**
13. `job-server` — Job execution narration
14. `timeout-enforcer` — Timeout narration
15. `auto-update` — Rebuild narration

**Build Tools (1):**
16. `xtask` — Test harness narration

### Dependency Graph (Text)

```
observability-narration-core (16 dependents)
├── Main Binaries
│   ├── rbee-keeper
│   ├── queen-rbee
│   ├── rbee-hive
│   └── llm-worker-rbee
├── Service Crates
│   ├── rbee-openai-adapter (queen)
│   ├── scheduler (queen)
│   ├── artifact-catalog (hive)
│   └── model-provisioner (hive)
├── Lifecycle Crates
│   ├── lifecycle-local
│   ├── lifecycle-ssh
│   ├── lifecycle-shared
│   └── health-poll
├── Shared Crates
│   ├── job-server
│   ├── timeout-enforcer
│   └── auto-update
└── Build Tools
    └── xtask
```

### Usage Pattern

**Purpose:** Real-time feedback via SSE streaming

**Key Features:**
- `n!()` macro for narration emission
- SSE sink for job-scoped routing
- Thread-local context for job_id propagation
- Multiple modes (human, cute, story)

---

## 2. Job-Server Dependencies

### Dependents (10 total)

**Main Binaries (3):**
1. `rbee-keeper` — Job submission
2. `queen-rbee` — Job registry
3. `rbee-hive` — Job registry

**Worker Binary (1):**
4. `llm-worker-rbee` — Job execution

**Lifecycle Crates (3):**
5. `lifecycle-local` — Job-based operations
6. `lifecycle-ssh` — Job-based operations
7. `lifecycle-shared` — Job-based operations

**Shared Crates (2):**
8. `auto-update` — Job-based rebuild
9. `narration-core` — Job context (circular dependency)

**Build Tools (1):**
10. `xtask` — Test harness

### Dependency Graph (Text)

```
job-server (10 dependents)
├── Main Binaries
│   ├── rbee-keeper (client)
│   ├── queen-rbee (server)
│   └── rbee-hive (server)
├── Worker Binary
│   └── llm-worker-rbee (server)
├── Lifecycle Crates
│   ├── lifecycle-local
│   ├── lifecycle-ssh
│   └── lifecycle-shared
├── Shared Crates
│   ├── auto-update
│   └── narration-core (circular)
└── Build Tools
    └── xtask
```

### Usage Pattern

**Purpose:** Job-based architecture pattern

**Key Features:**
- `JobRegistry<T>` for job state management
- `execute_and_stream()` for SSE streaming
- Implements `jobs-contract::JobRegistryInterface`
- Generic over token type `T`

---

## 3. Job-Client Dependencies

### Dependents (7 total)

**Main Binaries (2):**
1. `rbee-keeper` — HTTP client for queen
2. `queen-rbee` — HTTP client for hive forwarding

**WASM SDKs (3):**
3. `queen-rbee-sdk` — WASM wrapper
4. `rbee-hive-sdk` — WASM wrapper
5. `llm-worker-sdk` — WASM wrapper

**Shared Crates (1):**
6. `job-client` (self-test)

**Narration Core (1):**
7. `narration-core` (for testing)

### Dependency Graph (Text)

```
job-client (7 dependents)
├── Main Binaries
│   ├── rbee-keeper (queen client)
│   └── queen-rbee (hive forwarder)
├── WASM SDKs
│   ├── queen-rbee-sdk
│   ├── rbee-hive-sdk
│   └── llm-worker-sdk
└── Testing
    ├── job-client (self)
    └── narration-core
```

### Usage Pattern

**Purpose:** HTTP client for job submission and SSE streaming

**Key Features:**
- `JobClient::submit_and_stream()` — Submit + stream SSE
- `JobClient::submit()` — Submit only
- Generic line handler callback
- Automatic [DONE] detection

---

## 4. Timeout-Enforcer Dependencies

### Dependents (7 total)

**Main Binaries (2):**
1. `rbee-keeper` — Operation timeouts
2. `queen-rbee` — Job timeouts

**Lifecycle Crates (3):**
3. `lifecycle-local` — Daemon operation timeouts
4. `lifecycle-ssh` — SSH operation timeouts
5. `lifecycle-shared` — Shared timeout logic

**Shared Crates (1):**
6. `timeout-enforcer-macros` — Proc macro support

**Self-Test (1):**
7. `timeout-enforcer` (tests)

### Dependency Graph (Text)

```
timeout-enforcer (7 dependents)
├── Main Binaries
│   ├── rbee-keeper
│   └── queen-rbee
├── Lifecycle Crates
│   ├── lifecycle-local
│   ├── lifecycle-ssh
│   └── lifecycle-shared
├── Proc Macro
│   └── timeout-enforcer-macros
└── Self-Test
    └── timeout-enforcer
```

### Usage Pattern

**Purpose:** Hard timeout enforcement with narration

**Key Features:**
- `TimeoutEnforcer::new(duration)` — Create enforcer
- `.with_job_id(job_id)` — Enable SSE routing
- `.enforce(future)` — Enforce timeout
- Narration events for timeout warnings

---

## 5. Auto-Update Dependencies

### Dependents (1 total)

**Build Tools (1):**
1. `xtask` — Dependency-aware rebuild

### Dependency Graph (Text)

```
auto-update (1 dependent)
└── Build Tools
    └── xtask
```

### Usage Pattern

**Purpose:** Dependency-aware auto-update

**Key Features:**
- Detects source file changes
- Rebuilds affected binaries
- Narration for rebuild progress

---

## 6. Heartbeat-Registry Dependencies

### Dependents (0 total)

**Status:** Currently unused (prepared for future use)

### Dependency Graph (Text)

```
heartbeat-registry (0 dependents)
└── (prepared for future use)
```

---

## 7. SSH-Config-Parser Dependencies

### Dependents (2 total)

**Main Binaries (2):**
1. `rbee-keeper` — SSH config parsing for hive discovery
2. `queen-rbee` — SSH config parsing for hive discovery

### Dependency Graph (Text)

```
ssh-config-parser (2 dependents)
└── Main Binaries
    ├── rbee-keeper
    └── queen-rbee
```

### Usage Pattern

**Purpose:** SSH config parsing for hive discovery

**Key Features:**
- Parse `~/.ssh/config` format
- Extract hive connection details
- Support for `Host`, `HostName`, `User`, `Port`

---

## 8. Timeout-Enforcer-Macros Dependencies

### Dependents (1 total)

**Shared Crates (1):**
1. `timeout-enforcer` — Proc macro support

### Dependency Graph (Text)

```
timeout-enforcer-macros (1 dependent)
└── Shared Crates
    └── timeout-enforcer
```

---

## Complete Dependency Graph

### All Shared Crates (Ranked by Dependents)

```
1. observability-narration-core (16 dependents) ████████████████
2. job-server (10 dependents)                   ██████████
3. job-client (7 dependents)                    ███████
4. timeout-enforcer (7 dependents)              ███████
5. ssh-config-parser (2 dependents)             ██
6. auto-update (1 dependent)                    █
7. timeout-enforcer-macros (1 dependent)        █
8. heartbeat-registry (0 dependents)            
```

### Cross-Crate Dependencies

```
narration-core
    ↓
job-server ←→ narration-core (circular)
    ↓
job-client
    ↓
[WASM SDKs]

timeout-enforcer ← timeout-enforcer-macros
    ↓
[Lifecycle crates]

ssh-config-parser
    ↓
[Main binaries]
```

---

## Dependency Patterns

### 1. Narration-First Pattern

**All user-facing operations use narration:**
- Main binaries emit narration for CLI/GUI feedback
- Lifecycle crates emit narration for operation progress
- Shared crates emit narration for internal operations

### 2. Job-Based Architecture

**All operations are jobs:**
- `job-server` manages job state
- `job-client` submits jobs and streams SSE
- `narration-core` routes events via job_id

### 3. Timeout Enforcement

**All long-running operations have timeouts:**
- `timeout-enforcer` wraps futures
- Narration events for timeout warnings
- `.with_job_id()` enables SSE routing

### 4. Lifecycle Abstraction

**Daemon lifecycle is abstracted:**
- `lifecycle-local` for local operations
- `lifecycle-ssh` for remote operations
- `lifecycle-shared` for common types

---

## Circular Dependencies

### Identified Circular Dependency

**`narration-core` ↔ `job-server`:**
- `job-server` depends on `narration-core` for narration
- `narration-core` depends on `job-server` for testing

**Resolution:** Testing dependency only (dev-dependencies)

---

## Summary Statistics

### Shared Crate Usage

| Crate | Dependents | Category |
|-------|-----------|----------|
| `narration-core` | 16 | Observability |
| `job-server` | 10 | Architecture |
| `job-client` | 7 | HTTP Client |
| `timeout-enforcer` | 7 | Reliability |
| `ssh-config-parser` | 2 | Configuration |
| `auto-update` | 1 | Build Tools |
| `timeout-enforcer-macros` | 1 | Proc Macro |
| `heartbeat-registry` | 0 | Future Use |

### Dependency Depth

**Level 0 (No dependencies on other shared crates):**
- `ssh-config-parser`
- `heartbeat-registry`
- `timeout-enforcer-macros`

**Level 1 (Depends on Level 0):**
- `timeout-enforcer` (depends on `timeout-enforcer-macros`)
- `narration-core` (standalone)

**Level 2 (Depends on Level 1):**
- `job-server` (depends on `narration-core`)
- `auto-update` (depends on `narration-core`)

**Level 3 (Depends on Level 2):**
- `job-client` (depends on `job-server` types)

---

**Next Phase:** [PHASE_3_NARRATION_USAGE.md](./PHASE_3_NARRATION_USAGE.md)
