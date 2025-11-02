# rbee Communication Contracts

**Author:** TEAM-384  
**Date:** Nov 2, 2025  
**Status:** ✅ CANONICAL REFERENCE  
**Purpose:** Define ALL inter-component communication patterns

---

## Golden Rule

**EVERY connection between components uses the job-client/job-server pattern.**

There are NO exceptions. There are NO other communication patterns.

---

## The Pattern

### Job Server (Receives Operations)

**Responsibilities:**
1. Exposes HTTP endpoints: `POST /v1/jobs` and `GET /v1/jobs/{job_id}/stream`
2. Creates jobs and SSE channels
3. Executes operations
4. Streams results via SSE
5. Sends completion markers: `[DONE]`, `[ERROR]`, `[CANCELLED]`

**Implementation:** Uses `job-server` crate + `jobs-contract` for types

### Job Client (Submits Operations)

**Responsibilities:**
1. Submits operations via `POST /v1/jobs`
2. Receives `JobResponse` with `job_id` and `sse_url`
3. Connects to SSE stream
4. Processes narration events
5. Detects completion markers

**Implementation:** Uses `job-client` crate + `jobs-contract` for types

---

## ALL Component Connections

### 1. rbee-keeper → queen-rbee

**Pattern:** Job Client → Job Server

```
rbee-keeper (CLI)                    queen-rbee (HTTP Daemon)
===============                      =========================
Uses: job-client                     Uses: job-server
Role: Job Client                     Role: Job Server

POST /v1/jobs                   →    Creates job
  Body: { "HiveList": {} }           Spawns background task
                                     Returns job_id + sse_url

GET /v1/jobs/{id}/stream       →    Streams SSE events
                                     Sends [DONE] on completion
```

**Operations:**
- `HiveList`, `HiveInstall`, `HiveStart`, `HiveStop`
- `WorkerList`, `WorkerSpawn`, `WorkerGet`
- `ModelList`, `ModelDownload`
- `Infer` (queen routes to worker)

**Port:** `http://localhost:7833` (queen)

---

### 2. rbee-keeper → rbee-hive

**Pattern:** Job Client → Job Server

```
rbee-keeper (CLI)                    rbee-hive (HTTP Daemon)
===============                      =========================
Uses: job-client                     Uses: job-server
Role: Job Client                     Role: Job Server

POST /v1/jobs                   →    Creates job
  Body: { "ModelList": {...} }       Spawns background task
                                     Returns job_id + sse_url

GET /v1/jobs/{id}/stream       →    Streams SSE events
                                     Sends [DONE] on completion
```

**Operations:**
- `ModelList`, `ModelDownload`, `ModelGet`, `ModelDelete`
- `WorkerProcessList`, `WorkerSpawn`
- `HiveCheck` (diagnostic)

**Port:** `http://localhost:7835` (hive)

**Note:** TEAM-384 made this direct (bypassing queen for model operations)

---

### 3. queen-rbee → rbee-hive

**Pattern:** Job Client → Job Server

```
queen-rbee (HTTP Daemon)             rbee-hive (HTTP Daemon)
=======================              =========================
Uses: job-client                     Uses: job-server
Role: Job Client                     Role: Job Server
(Forwarding from keeper)             (Executing operations)

POST /v1/jobs                   →    Creates job
  Body: { "WorkerSpawn": {...} }     Spawns background task
                                     Returns job_id + sse_url

GET /v1/jobs/{id}/stream       →    Streams SSE events
                                     Sends [DONE] on completion
```

**Operations Queen Forwards:**
- `WorkerSpawn` - Start a worker process
- `ModelDownload` - Download model from HuggingFace
- `HiveStatus` - Get hive capabilities

**Port:** `http://localhost:7835` (hive)

**Flow:** keeper → queen → hive (queen acts as forwarding job-client)

---

### 4. queen-rbee → llm-worker-rbee

**Pattern:** Job Client → Job Server

```
queen-rbee (HTTP Daemon)             llm-worker-rbee (HTTP Daemon)
=======================              =============================
Uses: job-client                     Uses: job-server
Role: Job Client                     Role: Job Server
(Routing inference)                  (Running inference)

POST /v1/jobs                   →    Creates job
  Body: { "Infer": {...} }           Loads model
                                     Runs inference
                                     Returns job_id + sse_url

GET /v1/jobs/{id}/stream       →    Streams tokens
                                     Sends [DONE] when complete
```

**Operations:**
- `Infer` - Run inference with prompt

**Port:** Dynamic (assigned by hive when worker spawns)

**Flow:** keeper → queen → worker (queen circumvents hive for inference)

---

### 5. rbee-hive → llm-worker-rbee

**Pattern:** Job Client → Job Server

```
rbee-hive (HTTP Daemon)              llm-worker-rbee (HTTP Daemon)
======================               =============================
Uses: job-client                     Uses: job-server
Role: Job Client                     Role: Job Server
(Managing worker lifecycle)          (Reporting status)

POST /v1/jobs                   →    Creates job
  Body: { "HealthCheck": {} }        Checks model loaded
                                     Returns status
                                     Returns job_id + sse_url

GET /v1/jobs/{id}/stream       →    Streams status
                                     Sends [DONE]
```

**Operations:**
- `HealthCheck` - Worker health/status
- `Shutdown` - Graceful worker shutdown

**Port:** Dynamic (worker reports to hive)

**Flow:** hive → worker (direct management, no queen)

---

## Summary Table

| Connection | Client | Server | Operations | Port |
|------------|--------|--------|------------|------|
| **keeper → queen** | rbee-keeper | queen-rbee | Hive mgmt, inference routing | 7833 |
| **keeper → hive** | rbee-keeper | rbee-hive | Model mgmt (direct) | 7835 |
| **queen → hive** | queen-rbee | rbee-hive | Worker spawn (forwarding) | 7835 |
| **queen → worker** | queen-rbee | llm-worker-rbee | Inference | Dynamic |
| **hive → worker** | rbee-hive | llm-worker-rbee | Health checks | Dynamic |

---

## Contract Enforcement

### Shared Types (jobs-contract)

ALL connections use these shared types:

```rust
// POST /v1/jobs response
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

// SSE completion markers
pub mod completion_markers {
    pub const DONE: &str = "[DONE]";
    pub const ERROR_PREFIX: &str = "[ERROR]";
    pub const CANCELLED: &str = "[CANCELLED]";
}

// Endpoint paths
pub mod endpoints {
    pub const SUBMIT_JOB: &str = "/v1/jobs";
    pub fn stream_job(job_id: &str) -> String;
    pub fn cancel_job(job_id: &str) -> String;
}
```

### Implementation Requirements

**Every Job Server MUST:**
1. Import `job-server` crate
2. Import `jobs-contract` for types
3. Expose `POST /v1/jobs` endpoint
4. Expose `GET /v1/jobs/{job_id}/stream` endpoint
5. Return `JobResponse` from job creation
6. Send `[DONE]` when job completes
7. Drop SSE sender to close channel

**Every Job Client MUST:**
1. Import `job-client` crate
2. Import `jobs-contract` for types
3. POST to `/v1/jobs` with operation
4. Parse `JobResponse` from response
5. Connect to SSE stream
6. Check for completion markers
7. Handle `[DONE]`, `[ERROR]`, `[CANCELLED]`

---

## Why This Matters

### Before (Chaos)

❌ Multiple communication patterns (HTTP, SSH, direct calls)  
❌ Inconsistent error handling  
❌ Different streaming protocols  
❌ No shared types  
❌ Duplication everywhere  

### After (Order)

✅ **ONE pattern** for ALL communication  
✅ **Consistent** error handling  
✅ **Unified** SSE streaming  
✅ **Shared** types and contracts  
✅ **Zero** duplication  

---

## Anti-Patterns (FORBIDDEN)

### ❌ Direct Function Calls

```rust
// WRONG - Bypasses job pattern
hive.spawn_worker(model)?;

// RIGHT - Uses job pattern
job_client.submit_and_stream(
    Operation::WorkerSpawn { ... },
    |line| println!("{}", line)
).await?;
```

### ❌ SSH/Remote Execution

```rust
// WRONG - Direct SSH command
ssh_client.execute("rbee-hive spawn-worker")?;

// RIGHT - Uses job pattern over HTTP
job_client.submit_and_stream(
    Operation::WorkerSpawn { ... },
    |line| println!("{}", line)
).await?;
```

### ❌ Custom HTTP Endpoints

```rust
// WRONG - Custom endpoint
POST /spawn-worker

// RIGHT - Standard job endpoint
POST /v1/jobs
Body: { "WorkerSpawn": {...} }
```

### ❌ Custom Response Formats

```rust
// WRONG - Custom response
{ "id": "123", "url": "/stream/123" }

// RIGHT - JobResponse from contract
{ "job_id": "123", "sse_url": "/v1/jobs/123/stream" }
```

---

## Exception: Worker Telemetry

**ONLY exception to job-client/job-server pattern:**

Workers send telemetry directly to queen via SSE POST:

```
llm-worker-rbee                      queen-rbee
===============                      ==========
POST /v1/telemetry/worker      →     Receives telemetry
  Body: { gpu_util, model, ... }     Updates registry
```

**Why?** Telemetry is push-based (worker → queen), not request/response.

**Everything else:** Uses job-client/job-server.

---

## Migration Guide

### If You Find Other Patterns

1. **Identify:** What components are communicating?
2. **Determine:** Client (initiates) vs Server (responds)
3. **Convert:** Client uses `job-client`, Server uses `job-server`
4. **Verify:** Uses `jobs-contract` types
5. **Test:** Confirm `[DONE]` marker works

### Example Migration

**Before:**
```rust
// Custom HTTP call
let response = reqwest::get(format!("{}/list-models", hive_url)).await?;
let models: Vec<Model> = response.json().await?;
```

**After:**
```rust
// Uses job pattern
let job_client = JobClient::new(hive_url);
job_client.submit_and_stream(
    Operation::ModelList { hive_id },
    |line| {
        if !line.starts_with('[') {
            println!("{}", line);  // Narration
        }
        Ok(())
    }
).await?;
```

---

## Verification Checklist

For ANY inter-component communication:

- [ ] Client imports `job-client` crate
- [ ] Server imports `job-server` crate
- [ ] Both import `jobs-contract` crate
- [ ] Client POSTs to `/v1/jobs`
- [ ] Server returns `JobResponse`
- [ ] Client connects to SSE stream
- [ ] Server sends narration events
- [ ] Server sends `[DONE]` on completion
- [ ] Client detects completion marker
- [ ] No custom endpoints
- [ ] No direct function calls
- [ ] No SSH commands

---

## Documentation

**Contract Definition:**
- `bin/97_contracts/jobs-contract/src/lib.rs` - Types and constants
- `bin/97_contracts/jobs-contract/HTTP_API_CONTRACT.md` - Integration guide

**Implementation:**
- `bin/99_shared_crates/job-server/` - Server-side crate
- `bin/99_shared_crates/job-client/` - Client-side crate

**Architecture:**
- `.arch/02_SHARED_INFRASTRUCTURE_PART_3.md` - Job pattern details
- `.arch/03_DATA_FLOW_PART_4.md` - Request flow patterns

---

## Final Word

**There is ONE way components talk to each other: job-client → job-server.**

No exceptions (except worker telemetry push).  
No custom patterns.  
No special cases.  
No shortcuts.

**ONE PATTERN. ALWAYS.**

---

**TEAM-384:** Communication contracts established. ALL connections are job-client/job-server. Period. 🔒
