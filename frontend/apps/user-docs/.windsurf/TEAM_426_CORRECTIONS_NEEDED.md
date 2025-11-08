# TEAM-426 Documentation Corrections Needed

**Date:** 2025-11-08  
**Status:** 🚨 CRITICAL - Documentation is OUTDATED  
**Action Required:** Update Heartbeat Architecture page

---

## 🚨 Critical Finding

**The heartbeat architecture has COMPLETELY CHANGED!**

The MD files we used as reference are outdated. The actual source code shows a different architecture.

---

## ❌ What We Documented (WRONG)

### Heartbeat Flow (Documented)

```
Worker → POST /v1/worker-heartbeat → Queen
Hive   → POST /v1/hive-heartbeat   → Queen
Queen  → SSE broadcast (every 2.5s) → Clients
```

**Event Types Documented:**
- Queen heartbeat (every 2.5s)
- Worker heartbeat (every 30s) - forwarded from workers
- Hive heartbeat (every 30s) - forwarded from hives

---

## ✅ What Actually Exists (CORRECT)

### Actual Heartbeat Flow (From Source Code)

```
Hive → POST /v1/hive/ready (ONE-TIME discovery) → Queen
Queen → Subscribes to GET /v1/heartbeats/stream on Hive → Hive SSE stream
Hive → Sends telemetry via SSE (every 1s) → Queen
Queen → Broadcasts to clients via GET /v1/heartbeats/stream (every 2.5s) → Clients
```

**Actual Event Types (from source):**

1. **HiveTelemetry** (from Hive SSE stream):
   ```rust
   HiveTelemetry {
       hive_id: String,
       timestamp: String,
       workers: Vec<ProcessStats>,  // Worker process stats
   }
   ```

2. **Queen** (Queen's own heartbeat):
   ```rust
   Queen {
       workers_online: usize,
       workers_available: usize,
       hives_online: usize,
       hives_available: usize,
       worker_ids: Vec<String>,
       hive_ids: Vec<String>,
       timestamp: String,
   }
   ```

**NO separate Worker or Hive heartbeat events!**

---

## 🔍 Source Code Evidence

### File: `/bin/10_queen_rbee/src/main.rs` (Line 176-179)

```rust
// TEAM-374: DELETED /v1/hive-heartbeat route - replaced by SSE subscription
// TEAM-373: Hive ready callback (discovery) - triggers SSE subscription
.route("/v1/hive/ready", post(http::handle_hive_ready)) // TEAM-373: One-time discovery callback
.route("/v1/heartbeats/stream", get(http::handle_heartbeat_stream)) // TEAM-285: Live heartbeat streaming for web UI
```

**Key findings:**
- ❌ NO `/v1/worker-heartbeat` endpoint
- ❌ NO `/v1/hive-heartbeat` endpoint (DELETED by TEAM-374)
- ✅ NEW `/v1/hive/ready` endpoint (one-time discovery)
- ✅ `/v1/heartbeats/stream` exists (SSE broadcast)

### File: `/bin/10_queen_rbee/src/http/heartbeat.rs` (Line 79-81)

```rust
// TEAM-374: DELETED handle_hive_heartbeat() - replaced by SSE subscription
// Old POST-based continuous telemetry receiver is deprecated.
// Queen now subscribes to hive SSE streams (hive_subscriber.rs)
```

### File: `/bin/10_queen_rbee/src/hive_subscriber.rs` (Line 1-5)

```rust
//! Queen subscribes to Hive SSE streams
//!
//! After discovery handshake, Queen connects to each hive's
//! GET /v1/heartbeats/stream and aggregates telemetry.
```

---

## 🔄 Correct Architecture (TEAM-373, TEAM-374)

### Discovery Protocol

1. **Hive starts** and detects Queen
2. **Hive sends** `POST /v1/hive/ready` with:
   ```json
   {
     "hive_id": "gpu-0",
     "hive_url": "http://192.168.1.100:7835"
   }
   ```
3. **Queen receives** callback and starts SSE subscription
4. **Queen subscribes** to `GET http://192.168.1.100:7835/v1/heartbeats/stream`
5. **Hive streams** telemetry events (every 1s)
6. **Queen aggregates** and broadcasts to clients (every 2.5s)

### Telemetry Flow

```
┌─────────────────────────────────────────────────────┐
│ Hive                                                │
│  - Monitors worker processes (ps)                   │
│  - Streams telemetry via SSE (every 1s)             │
│  - GET /v1/heartbeats/stream                        │
└──────────────────┬──────────────────────────────────┘
                   │ SSE Stream (Queen subscribes)
                   ▼
┌─────────────────────────────────────────────────────┐
│ Queen (Subscriber + Aggregator)                     │
│  - Subscribes to each hive's SSE stream             │
│  - Receives HiveTelemetry events                    │
│  - Updates TelemetryRegistry                        │
│  - Broadcasts to clients (every 2.5s)               │
└──────────────────┬──────────────────────────────────┘
                   │ SSE Broadcast
                   ▼
┌─────────────────────────────────────────────────────┐
│ Clients (Web UI, CLI)                               │
│  - Subscribe: GET /v1/heartbeats/stream             │
│  - Receive HiveTelemetry + Queen events             │
└─────────────────────────────────────────────────────┘
```

**Key differences:**
- ✅ **Pull-based** (Queen subscribes to Hive), not push-based
- ✅ **One-time discovery** (`/v1/hive/ready`), not continuous POST
- ✅ **SSE from Hive to Queen**, not POST requests
- ✅ **Workers included in Hive telemetry**, not separate events

---

## ✅ Job Operations (CORRECT)

**Good news:** Job operations documentation is mostly correct!

### Verified from Source

**File:** `/bin/97_contracts/operations-contract/src/lib.rs` (Lines 90-185)

**Queen Operations (Port 7833):**
- ✅ `Status` - Query registries
- ✅ `Infer` - Schedule inference
- ✅ `ImageGeneration` - NEW! (TEAM-397)
- ✅ `ImageTransform` - NEW! (TEAM-397)
- ✅ `ImageInpaint` - NEW! (TEAM-397)
- ✅ `RhaiScriptSave/Test/Get/List/Delete` - RHAI script management
- ✅ `QueenCheck` - Diagnostic

**Hive Operations (Port 7835):**
- ✅ `WorkerCatalogList/Get` - NEW! (TEAM-388) Query catalog server
- ✅ `WorkerListInstalled/InstalledGet` - NEW! (TEAM-388)
- ✅ `WorkerInstall/Remove` - NEW! (TEAM-388)
- ✅ `WorkerSpawn` - Spawn worker process
- ✅ `WorkerProcessList/Get/Delete` - Worker process management
- ✅ `ModelDownload/List/Get/Delete` - Model management
- ✅ `ModelLoad/Unload` - NEW!
- ✅ `HiveCheck` - Diagnostic

**Additional operations we didn't document:**
- Image generation operations (3 new)
- RHAI script management (5 operations)
- Worker catalog operations (6 new)
- Model load/unload (2 new)

---

## 🔧 Required Fixes

### 1. Update Heartbeat Architecture Page

**File:** `/app/docs/architecture/heartbeats/page.mdx`

**Changes needed:**

1. **Remove incorrect sections:**
   - ❌ "Worker Heartbeat" event type
   - ❌ "Hive Heartbeat" event type
   - ❌ POST `/v1/worker-heartbeat` endpoint
   - ❌ POST `/v1/hive-heartbeat` endpoint
   - ❌ "Workers send directly to Queen every 30s"
   - ❌ "Hives send directly to Queen every 30s"

2. **Add correct sections:**
   - ✅ Discovery protocol (`POST /v1/hive/ready`)
   - ✅ SSE subscription (Queen → Hive)
   - ✅ HiveTelemetry event type
   - ✅ Pull-based architecture
   - ✅ Telemetry frequency (1s from Hive, 2.5s from Queen)

3. **Update diagrams:**
   - Show one-time discovery callback
   - Show Queen subscribing to Hive SSE
   - Show workers included in Hive telemetry
   - Remove direct worker → Queen communication

### 2. Enhance Job Operations Page (Optional)

**File:** `/app/docs/reference/job-operations/page.mdx`

**Optional additions:**
- Image generation operations (ImageGeneration, ImageTransform, ImageInpaint)
- RHAI script management operations
- Worker catalog operations (new in TEAM-388)
- Model load/unload operations

**Current documentation is correct but incomplete.**

---

## 📊 Impact Assessment

### Heartbeat Architecture

**Severity:** 🚨 **CRITICAL**  
**Impact:** Documentation is completely wrong  
**User Impact:** HIGH - Users will try to use endpoints that don't exist  
**Fix Priority:** IMMEDIATE

### Job Operations

**Severity:** ⚠️ **MEDIUM**  
**Impact:** Documentation is correct but incomplete  
**User Impact:** LOW - Core operations documented correctly  
**Fix Priority:** MEDIUM (can be done later)

---

## 🎯 Action Plan

### Immediate (TEAM-427 or TEAM-426 fix)

1. **Rewrite Heartbeat Architecture page** based on actual source code
2. **Test endpoints** to verify behavior
3. **Update examples** to match real API

### Follow-up (Future team)

1. **Add image generation operations** to Job Operations page
2. **Add RHAI script operations** to Job Operations page
3. **Add worker catalog operations** to Job Operations page
4. **Create separate page** for advanced operations

---

## 📚 Source Files to Reference

**For Heartbeat Architecture:**
- `/bin/10_queen_rbee/src/main.rs` (routes)
- `/bin/10_queen_rbee/src/http/heartbeat.rs` (discovery callback)
- `/bin/10_queen_rbee/src/hive_subscriber.rs` (SSE subscription)
- `/bin/10_queen_rbee/src/http/heartbeat_stream.rs` (SSE broadcast)

**For Job Operations:**
- `/bin/97_contracts/operations-contract/src/lib.rs` (Operation enum)
- `/bin/10_queen_rbee/src/job_router.rs` (routing logic)
- `/bin/10_queen_rbee/src/http/jobs.rs` (HTTP endpoints)

---

## ✅ What We Got Right

**Job Operations Reference:**
- ✅ API split (Queen vs Hive) - CORRECT
- ✅ NO PROXYING principle - CORRECT
- ✅ Port assignments (7833 vs 7835) - CORRECT
- ✅ Status and Infer operations - CORRECT
- ✅ Worker and Model operations - CORRECT
- ✅ Job pattern (submit → stream → [DONE]) - CORRECT

**Only missing:** New operations added after MD files were written

---

## 🚨 Lesson Learned

**ALWAYS verify against source code, not MD files!**

MD files can be outdated. Source code is the truth.

**Next time:**
1. Read MD files for context
2. Verify against actual source code
3. Check git history for recent changes
4. Test endpoints if possible

---

**TEAM-426 Signature** ✅

**Status:** Documentation corrections identified  
**Next Action:** TEAM-427 should fix Heartbeat Architecture page  
**Priority:** CRITICAL for Heartbeat, MEDIUM for Job Operations
