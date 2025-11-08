# TEAM-424: Source Code Deep Dive - Documentation Gaps

**Date:** 2025-11-08  
**Status:** 📋 ANALYSIS COMPLETE  
**Mission:** Identify what's missing from user docs based on actual source code

---

## Executive Summary

After analyzing the actual source code, the user docs are **missing critical implementation details** that users need to understand and use the system. The docs describe the **conceptual architecture** but lack **operational details**.

**Key Finding:** Docs describe "what" but not "how" or "why"

---

## 1. Critical Missing: Job-Based Architecture

### Current Docs
The docs mention "orchestration" but don't explain the **job-based pattern**.

### Actual Implementation
**EVERYTHING in rbee uses a job-based pattern:**

```bash
# Submit operation → Get job_id + SSE URL
POST /v1/jobs → {job_id: "uuid", sse_url: "/v1/jobs/{uuid}/stream"}

# Stream narration events
GET /v1/jobs/{uuid}/stream → SSE stream with progress

# Example: Spawn worker
curl -X POST http://localhost:7835/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "worker_spawn",
    "hive_id": "localhost",
    "model": "meta-llama/Llama-3.2-1B",
    "worker": "cpu",
    "device": 0
  }'

# Response
{"job_id": "abc-123", "sse_url": "/v1/jobs/abc-123/stream"}

# Then connect to SSE stream
curl http://localhost:7835/v1/jobs/abc-123/stream

# Stream output
data: {"action":"worker_spawn_start","message":"🚀 Spawning worker..."}
data: {"action":"worker_spawn_health_check","message":"Waiting for worker..."}
data: {"action":"worker_spawn_complete","message":"✅ Worker spawned (PID: 1234)"}
data: [DONE]
```

**What's Missing in Docs:**
- No explanation of job submission pattern
- No SSE streaming examples
- No narration event format documentation
- No job lifecycle explanation

---

## 2. Critical Missing: Queen vs Hive API Split

### Current Docs
Docs say "Queen orchestrates everything" but don't explain the **API split**.

### Actual Implementation
**Queen and Hive have SEPARATE job servers:**

#### Queen Job Server (Port 7833)
**Only 2 operations:**
```bash
POST /v1/jobs
- Operation::Status   # Query registries
- Operation::Infer    # Schedule inference
```

**Also provides:**
```bash
POST /openai/v1/chat/completions  # OpenAI-compatible
GET  /openai/v1/models             # OpenAI-compatible
GET  /v1/heartbeats/stream         # SSE heartbeat stream
```

#### Hive Job Server (Port 7835)
**8 operations:**
```bash
POST /v1/jobs
- WorkerSpawn, WorkerProcessList, WorkerProcessGet, WorkerProcessDelete
- ModelDownload, ModelList, ModelGet, ModelDelete
```

#### rbee-keeper CLI Pattern
```bash
rbee-keeper CLI
  ├─→ Queen (http://localhost:7833/v1/jobs)  # Inference only
  └─→ Hive (http://localhost:7835/v1/jobs)   # Worker/model management
```

**NO PROXYING** - Keeper talks to BOTH servers directly.

**What's Missing in Docs:**
- No explanation of why there are TWO job servers
- No port number documentation (7833 vs 7835)
- No operation routing guide (which ops go where)
- No examples of talking to hive directly
- Current docs imply everything goes through Queen (WRONG)

---

## 3. Critical Missing: Heartbeat Architecture

### Current Docs
Docs say "Queen monitors everything" vaguely.

### Actual Implementation (TEAM-288)
**Event-driven heartbeat system:**

```
┌─────────────┐
│   Worker    │ POST /v1/worker-heartbeat → Queen (every 30s)
└─────────────┘

┌─────────────┐
│    Hive     │ POST /v1/hive-heartbeat → Queen (every 30s)
└─────────────┘

┌─────────────┐
│   Queen     │ Timer (every 2.5s) → Broadcast to SSE clients
└─────────────┘
                        ↓
               GET /v1/heartbeats/stream (SSE)
                        ↓
                   ┌─────────┐
                   │ Clients │
                   └─────────┘
```

**Three heartbeat event types:**
```json
// Queen's heartbeat (every 2.5s)
{
  "type": "queen",
  "workers_online": 2,
  "workers_available": 1,
  "hives_online": 1,
  "timestamp": "2025-11-08T13:00:00Z"
}

// Worker heartbeat (real-time forwarding)
{
  "type": "worker",
  "worker_id": "worker-1",
  "status": "Ready",
  "timestamp": "2025-11-08T13:00:01Z"
}

// Hive heartbeat (real-time forwarding)
{
  "type": "hive",
  "hive_id": "gpu-0",
  "status": "Online",
  "timestamp": "2025-11-08T13:00:02Z"
}
```

**Critical Discovery Protocol:**
```
1. Queen starts, reads SSH config
2. Queen → GET /capabilities?queen_url=http://queen:7833 → Each hive
3. Hive receives queen_url, starts heartbeat task
4. Hive → POST /v1/hive-heartbeat → Queen (every 30s)
```

**What's Missing in Docs:**
- No SSE endpoint documentation (`/v1/heartbeats/stream`)
- No heartbeat event format examples
- No discovery protocol explanation
- No queen_url configuration guide
- **CRITICAL BUG DOCS:** Remote hives need Queen's public IP, not localhost!

---

## 4. Critical Missing: Worker Binary Types

### Current Docs
Docs say "workers run inference" but don't explain the **three binaries**.

### Actual Implementation
**Three feature-gated worker binaries:**

| Binary | Feature | Device | Use Case |
|--------|---------|--------|----------|
| `llorch-cpu-candled` | `cpu` | CPU | x86 Linux/Windows, macOS CPU |
| `llorch-cuda-candled` | `cuda` | CUDA | NVIDIA GPU |
| `llorch-metal-candled` | `metal` | Metal | Apple Silicon GPU |

**Usage:**
```bash
# CPU worker
./llorch-cpu-candled \
  --worker-id test-worker \
  --model /path/to/llama/ \
  --port 9001 \
  --queen-url http://localhost:7833

# CUDA worker (device selection)
./llorch-cuda-candled \
  --worker-id test-worker \
  --model /path/to/llama/ \
  --port 9001 \
  --cuda-device 0 \
  --queen-url http://192.168.1.100:7833  # ← Remote queen!

# Metal worker
./llorch-metal-candled \
  --worker-id test-worker \
  --model /path/to/llama/ \
  --port 9001 \
  --metal-device 0 \
  --queen-url http://localhost:7833
```

**Model Format Support:**
- ✅ SafeTensors (current)
- ⏳ GGUF (planned)

**What's Missing in Docs:**
- No explanation of three worker binaries
- No device selection guide (--cuda-device, --metal-device)
- No model format documentation (SafeTensors vs GGUF)
- No --queen-url configuration examples

---

## 5. Critical Missing: Catalog Architecture

### Current Docs
Docs say "models are stored" but don't explain the **catalog system**.

### Actual Implementation (TEAM-273)
**Shared artifact abstraction:**

```
artifact-catalog (generic)
    ├─→ Artifact trait
    │   ├─ id() -> &str
    │   ├─ path() -> &Path
    │   ├─ size() -> u64
    │   └─ status() -> ArtifactStatus
    │
    ├─→ ArtifactCatalog trait
    │   ├─ add(artifact)
    │   ├─ get(id)
    │   ├─ list()
    │   └─ remove(id)
    │
    └─→ FilesystemCatalog<T: Artifact>
        └─→ Generic JSON metadata implementation

model-catalog (concrete)
    └─→ ModelEntry (implements Artifact)
        └─→ ModelCatalog (wraps FilesystemCatalog<ModelEntry>)

worker-catalog (concrete)
    └─→ WorkerBinary (implements Artifact)
        └─→ WorkerCatalog (wraps FilesystemCatalog<WorkerBinary>)
```

**Filesystem Layout:**
```
~/.cache/rbee/models/
├── meta-llama-Llama-2-7b/
│   ├── metadata.json          # ModelEntry serialized
│   └── model.safetensors      # Actual model file
├── mistralai-Mistral-7B/
│   ├── metadata.json
│   └── model.safetensors
└── ...

~/.cache/rbee/workers/
├── cpu-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json          # WorkerBinary serialized
│   └── llorch-cpu-candled     # Actual binary
├── cuda-llm-worker-rbee-v0.1.0-linux/
│   ├── metadata.json
│   └── llorch-cuda-candled
└── ...
```

**What's Missing in Docs:**
- No catalog directory structure documentation
- No metadata.json format examples
- No explanation of why there are two catalogs
- No manual model installation guide

---

## 6. Critical Missing: Keeper is TWO Interfaces

### Current Docs
Docs say "Keeper is the GUI" but miss the **CLI interface**.

### Actual Implementation
**Keeper has TWO interfaces sharing business logic:**

#### CLI (Command Line)
```bash
rbee-keeper infer --model llama-7b --prompt "Hello"
rbee-keeper hive start --host localhost
rbee-keeper workers list
```

**Use when:**
- Scripting/automation
- Remote SSH sessions
- CI/CD pipelines
- Server environments

#### GUI (Graphical)
```bash
rbee-keeper-gui  # Launches Tauri desktop app
```

**Use when:**
- Interactive exploration
- Visual feedback needed
- Desktop environment
- New users learning

**GUI Architecture:**
```
rbee-keeper GUI
  ├─→ Queen Web UI (iframe: http://localhost:7833/)
  ├─→ Hive Web UI (iframe: http://localhost:7835/)
  └─→ Worker Web UI (iframe: http://localhost:8080/)
```

**What's Missing in Docs:**
- No CLI command reference
- No GUI vs CLI decision guide
- No iframe architecture explanation
- No port numbers for web UIs

---

## 7. Critical Missing: OpenAI Compatibility Details

### Current Docs
Docs mention "OpenAI-compatible" but lack **implementation details**.

### Actual Implementation
**Queen provides OpenAI-compatible endpoints:**

```
POST /openai/v1/chat/completions  # Chat endpoint
GET  /openai/v1/models             # List models
GET  /openai/v1/models/{model}     # Get model
POST /openai/v1/completions        # Legacy completions
POST /openai/v1/embeddings         # Planned (not yet)
```

**Translation Layer:**
```python
# OpenAI request
client = OpenAI(
    base_url="http://localhost:7833/openai",  # ← /openai prefix!
    api_key="not-needed"
)

# Internally translates to:
Operation::Infer {
    model: "llama-3-8b",
    prompt: "Hello!",
    ...
}
```

**Response Format:**
```json
// OpenAI format
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "llama-3-8b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

**What's Missing in Docs:**
- No `/openai` prefix documentation (critical!)
- No parameter mapping guide (OpenAI → rbee)
- No response format examples
- No streaming format documentation
- No "what's not supported" list

---

## 8. Critical Missing: Queen URL Configuration

### Current Docs
No mention of this **critical bug pattern**.

### Actual Implementation
**THE BUG:** Remote hives need Queen's public IP, not localhost!

❌ **WRONG (causes silent failure):**
```bash
# Remote hive spawned with localhost queen URL
ssh remote-machine "rbee-hive --queen-url http://localhost:7833"
# ❌ Worker sends heartbeat to its OWN localhost
# ❌ Queen never receives heartbeat
# ❌ Queen thinks worker is offline
```

✅ **CORRECT:**
```bash
# Remote hive spawned with Queen's public IP
ssh remote-machine "rbee-hive --queen-url http://192.168.1.100:7833"
# ✅ Worker sends heartbeat to Queen's IP
# ✅ Queen receives heartbeat
# ✅ Everything works
```

**Configuration Required:**
```toml
# ~/.config/rbee/config.toml
[queen]
port = 7833
public_address = "192.168.1.100"  # ← REQUIRED for remote hives!
# OR
public_hostname = "queen.local"   # ← Alternative
```

**What's Missing in Docs:**
- **CRITICAL:** No queen_url configuration guide
- **CRITICAL:** No remote hive setup troubleshooting
- **CRITICAL:** No explanation of localhost vs public IP
- No public_address config documentation

---

## 9. Critical Missing: Inference Request Flow

### Current Docs
Docs show ASCII diagrams but lack **actual HTTP flow**.

### Actual Implementation
**Complete flow with HTTP calls:**

```
┌─────────────────────────────────────────────────────────┐
│ Client: rbee-keeper or OpenAI SDK                       │
└──────────────────┬──────────────────────────────────────┘
                   │ POST /v1/jobs (Operation::Infer)
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Queen (http://localhost:7833)                           │
│  1. Admission control (accept/reject)                   │
│  2. Check worker registry for available worker          │
│  3. If no worker:                                        │
│     a. POST /v1/jobs (WorkerSpawn) → Hive               │
│     b. Wait for worker heartbeat                        │
│  4. POST /v1/infer → Worker DIRECTLY                    │
│  5. Relay SSE stream back to client                     │
└──────────────────┬──────────────────────────────────────┘
                   │ (if worker spawn needed)
                   │ POST http://localhost:7835/v1/jobs
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Hive (http://localhost:7835)                            │
│  - Receives WorkerSpawn job                             │
│  - Spawns worker process with --queen-url               │
│  - Worker sends POST /v1/worker-heartbeat → Queen       │
│  - Hive NEVER sees inference requests                   │
└──────────────────┬──────────────────────────────────────┘
                   │ spawns process
                   ▼
┌─────────────────────────────────────────────────────────┐
│ Worker (http://localhost:9001)                          │
│  - POST /v1/workers/ready → Queen (once at startup)     │
│  - POST /v1/worker-heartbeat → Queen (every 30s)        │
│  - Receives POST /v1/infer from Queen                   │
│  - Returns SSE stream to Queen                          │
└─────────────────────────────────────────────────────────┘
                   ▲
                   │ POST /v1/infer (DIRECT from Queen)
                   │
                 Queen
```

**What's Missing in Docs:**
- No actual HTTP endpoints in flow diagrams
- No port numbers in examples
- No complete request/response examples
- No error handling documentation

---

## 10. Critical Missing: Security & Auth

### Current Docs
Docs mention "optional API key" but no details.

### Actual Implementation
**Security crates exist:**
```
bin/98_security_crates/
├── audit-logging/         # Audit trail for compliance
├── auth-min/              # Minimal authentication
├── deadline-propagation/  # Request timeout handling
├── input-validation/      # Input sanitization
├── jwt-guardian/          # JWT token management
└── secrets-management/    # Secret storage
```

**What's Missing in Docs:**
- **CRITICAL:** No authentication setup guide
- No API key generation instructions
- No audit logging configuration
- No GDPR compliance implementation details
- No security best practices

---

## 11. Documentation Components Needed (Updated)

### A. Architecture Deep Dives

#### 1. Job-Based Pattern Guide
**File:** `/docs/architecture/job-based-pattern`

**Content:**
- Job submission pattern explained
- SSE streaming format
- Narration event types
- Job lifecycle states
- Error handling patterns

**Components Needed:**
- `<TerminalWindow>` - Show curl commands
- `<CodeTabs>` - Python/JS/cURL examples
- `<Callout>` - Highlight SSE format
- `<StepGuide>` - Submit → Stream → Complete

---

#### 2. API Split Architecture
**File:** `/docs/architecture/api-split`

**Content:**
- Why two job servers?
- Queen operations (2)
- Hive operations (8)
- Port number reference (7833, 7835)
- Routing decision tree

**Components Needed:**
- `<APIParameterTable>` - Operations by server
- `<FeatureComparison>` - Queen vs Hive API
- `<ArchitectureDiagram>` - Visual flow

---

#### 3. Heartbeat Architecture
**File:** `/docs/architecture/heartbeats`

**Content:**
- Event-driven heartbeat system
- Three heartbeat types
- SSE endpoint usage
- Discovery protocol
- Monitoring best practices

**Components Needed:**
- `<CodeSnippet>` - SSE connection examples
- `<TerminalWindow>` - Live heartbeat output
- `<Tabs>` - Worker/Hive/Queen heartbeat examples

---

### B. Getting Started Enhancements

#### 4. Worker Type Selection Guide
**File:** `/docs/getting-started/worker-types`

**Content:**
- Three worker binaries explained
- CPU vs CUDA vs Metal comparison
- Device selection guide
- Performance expectations
- Model format support

**Components Needed:**
- `<FeatureComparison>` - Worker binary comparison
- `<CodeSnippet>` - Launch examples
- `<Callout>` - GGUF not yet supported warning

---

#### 5. Remote Hive Setup
**File:** `/docs/getting-started/remote-hives`

**Content:**
- **CRITICAL:** Queen URL configuration
- localhost vs public IP problem
- SSH setup for remote hives
- Troubleshooting heartbeat issues
- Network requirements

**Components Needed:**
- `<Callout variant="warning">` - localhost bug warning
- `<StepGuide>` - Remote hive setup steps
- `<CodeSnippet>` - Config file examples
- `<TerminalWindow>` - SSH commands

---

### C. API Reference Enhancements

#### 6. Complete Job Operations Reference
**File:** `/docs/reference/job-operations`

**Content:**
- ALL operations documented
- Queen operations (2)
- Hive operations (8)
- Request/response examples
- Error codes

**Components Needed:**
- `<APIParameterTable>` - All parameters
- `<Tabs>` - Request/response examples
- `<CodeSnippet>` - cURL examples

---

#### 7. OpenAI Compatibility Details
**File:** `/docs/reference/openai-compatibility`

**Content:**
- **CRITICAL:** `/openai` prefix documented
- Supported endpoints
- Not supported features
- Parameter mapping
- Response format differences

**Components Needed:**
- `<CodeTabs>` - Python/JS examples
- `<FeatureComparison>` - Supported vs not
- `<Callout>` - /openai prefix requirement

---

#### 8. Heartbeat SSE Endpoint
**File:** `/docs/reference/heartbeat-stream`

**Content:**
- SSE endpoint documentation
- Event format specification
- Connection examples
- Monitoring use cases

**Components Needed:**
- `<CodeSnippet>` - SSE connection
- `<TerminalWindow>` - Live event stream
- `<Tabs>` - Python/JS/cURL

---

### D. Configuration Guides

#### 9. Queen Configuration
**File:** `/docs/configuration/queen`

**Content:**
- **CRITICAL:** public_address config
- Port configuration
- Remote hive support
- SSH key setup
- Database options (ephemeral vs persistent)

**Components Needed:**
- `<CodeSnippet>` - config.toml examples
- `<Callout>` - Remote hive requirements
- `<StepGuide>` - Initial setup

---

#### 10. Security Configuration
**File:** `/docs/configuration/security`

**Content:**
- API key setup
- Authentication configuration
- Audit logging
- GDPR compliance
- Best practices

**Components Needed:**
- `<Callout variant="warning">` - Security warnings
- `<CodeSnippet>` - Config examples
- `<StepGuide>` - Setup steps

---

### E. Troubleshooting Guides

#### 11. Common Issues
**File:** `/docs/troubleshooting/common-issues`

**Content:**
- **CRITICAL:** Worker heartbeat not received
- **CRITICAL:** localhost vs public IP
- Port conflicts (7833, 7835, 9000+)
- Model not found errors
- Worker spawn failures

**Components Needed:**
- `<Accordion>` - Collapsible issues
- `<Callout>` - Root cause explanations
- `<TerminalWindow>` - Debug commands

---

### F. CLI Reference

#### 12. rbee-keeper CLI Commands
**File:** `/docs/reference/cli`

**Content:**
- ALL CLI commands documented
- Command syntax
- Options and flags
- Examples for each command
- Environment variables

**Components Needed:**
- `<CodeSnippet>` - Command examples
- `<APIParameterTable>` - Flags reference
- `<Tabs>` - Basic/Advanced usage

---

## 12. Priority Ranking

### 🔴 **Critical (Week 1)** - Missing functionality docs
1. **Remote Hive Setup** - Queen URL configuration (users are stuck!)
2. **Job-Based Pattern** - Users can't use the API properly
3. **API Split Guide** - Users don't know which port to use
4. **OpenAI /openai Prefix** - Current docs are wrong without this

### 🟡 **High Priority (Week 2)** - Operational knowledge
5. **Worker Types Guide** - Users need to choose correct binary
6. **Heartbeat Architecture** - Users need to monitor systems
7. **Job Operations Reference** - Complete API docs
8. **CLI Reference** - Command documentation

### 🟢 **Medium Priority (Week 3)** - Deep dives
9. **Catalog Architecture** - Advanced users
10. **Security Configuration** - Production deployments
11. **Troubleshooting** - Common issues
12. **Configuration Guides** - Advanced setup

---

## 13. Components Priority (Updated from TEAM_424_USER_DOCS_COMPONENT_PLAN.md)

Based on source code analysis, the component priorities are now:

### Phase 1: Critical (Operational Docs)
1. **`<CodeSnippet>` with copy** - For API examples (most critical!)
2. **`<Callout>` (Alert wrapper)** - For queen_url warning, /openai prefix
3. **`<APIParameterTable>`** - For job operations reference
4. **`<CodeTabs>`** - For multi-language API examples
5. **`<TerminalWindow>`** - For showing actual HTTP requests/responses

### Phase 2: Navigation & Structure
6. **`<TopNavBar>`** - Site navigation
7. **`<LinkCard>` + `<CardGrid>`** - Better page navigation
8. **`<Breadcrumbs>`** - Page context
9. **`<Accordion>`** - Troubleshooting sections

### Phase 3: Visual Enhancements
10. **`<ArchitectureDiagram>`** - Flow diagrams with ports
11. **`<FeatureComparison>`** - Queen vs Hive, Worker types
12. **`<StepGuide>`** - Setup walkthroughs

---

## 14. Immediate Action Items

### For Next Team

1. **Create:** `/docs/getting-started/remote-hives` with queen_url warning
2. **Create:** `/docs/architecture/job-based-pattern` with SSE examples
3. **Update:** `/docs/reference/api-openai-compatible` with /openai prefix
4. **Create:** `/docs/reference/job-operations` with complete API
5. **Create:** `/docs/reference/cli` with all commands

### Component Development Priority

1. **`<CodeSnippet>`** - Most used, implement first
2. **`<Callout>`** - For critical warnings
3. **`<APIParameterTable>`** - For API docs
4. **`<CodeTabs>`** - For multi-language examples
5. **`<TerminalWindow>`** - For HTTP flow examples

---

## 15. Documentation Accuracy Issues

### Current Docs Issues to Fix

1. **Port 8500 → 7833** (Queen) - Already fixed by TEAM-458
2. **Port 9000 → 7835** (Hive) - Already fixed by TEAM-458
3. **Missing /openai prefix** in OpenAI compatibility examples
4. **No mention of two job servers** (Queen vs Hive)
5. **Inference flow diagrams lack HTTP endpoints**
6. **No worker binary types documented**
7. **No queen_url configuration documented**

---

## Conclusion

The user docs have **excellent high-level architecture** but are missing **critical operational details** that users need to actually use the system.

**The gap:** Conceptual understanding ✅ | Operational knowledge ❌

**Solution:** Add 12 new documentation pages covering:
- Job-based pattern with SSE
- API split (Queen vs Hive)
- Heartbeat architecture
- Worker types and binaries
- Remote hive configuration (CRITICAL)
- Complete API reference
- CLI command reference

**Component needs:** Same as original plan, but prioritize operational components (CodeSnippet, Callout, APIParameterTable) over visual polish.

---

**Next:** Implement Phase 1 components + create the 4 critical documentation pages (Remote Hives, Job Pattern, API Split, OpenAI Prefix)
