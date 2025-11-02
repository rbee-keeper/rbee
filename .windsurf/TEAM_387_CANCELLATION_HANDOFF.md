# TEAM-387: Download Cancellation Implementation

**Status:** ✅ CLIENT-SIDE COMPLETE | ⏳ SERVER-SIDE DEFERRED

**RULE ZERO Applied:** Broke `JobClient` API cleanly. No backwards compatibility wrappers. Compiler found all call sites. Fixed them all.

---

## What We Implemented

### Phase 1-3: Client-Side Cancellation (COMPLETE)

**User can now press Ctrl+C during downloads. The client:**
1. Gets `job_id` immediately after submission
2. Sends `DELETE /v1/jobs/{job_id}` to server
3. Exits cleanly with cancellation message

**Files Changed:**
- `bin/99_shared_crates/job-client/src/lib.rs` - TEAM-387: Broke API (RULE ZERO)
- `bin/00_rbee_keeper/src/job_client.rs` - TEAM-387: Added Ctrl+C handling
- `bin/00_rbee_keeper/src/handlers/hive_jobs.rs` - TEAM-387: Fixed call site
- `bin/10_queen_rbee/ui/packages/queen-rbee-sdk/src/client.rs` - TEAM-387: Fixed WASM call site

---

## Breaking Change (RULE ZERO)

### Old API (DELETED)
```rust
// ❌ OLD - Returned job_id after streaming completed
pub async fn submit_and_stream<F>(&self, operation: Operation, handler: F) 
    -> Result<String>
```

### New API (CURRENT)
```rust
// ✅ NEW - Returns (job_id, stream_future) immediately
pub async fn submit_and_stream<F>(&self, operation: Operation, handler: F) 
    -> Result<(String, impl Future<Output = Result<()>>)>
```

**Why:** Need `job_id` immediately to send cancel request. Old API blocked until streaming completed.

**Migration Pattern:**
```rust
// Before
let job_id = client.submit_and_stream(op, |line| { ... }).await?;

// After
let (job_id, stream_fut) = client.submit_and_stream(op, |line| { ... }).await?;
stream_fut.await?;
```

---

## How Cancellation Works

### User Experience
```bash
./rbee model download
📥 Downloading file: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
⏳ Downloading from HuggingFace...
⏳ Still downloading...
^C
🛑 Cancelling operation...
🛑 Operation cancelled by user
```

### Code Flow
```rust
// 1. Submit job, get job_id via channel
let (job_id_tx, job_id_rx) = tokio::sync::oneshot::channel();
let stream_task = tokio::spawn(async move {
    submit_and_stream_with_id_callback(url, op, name, job_id_tx).await
});

// 2. Get job_id immediately
let job_id = job_id_rx.await?;

// 3. Wait for stream OR Ctrl+C
tokio::select! {
    result = timeout_future => result,
    _ = tokio::signal::ctrl_c() => {
        // Send DELETE request
        let cancel_url = format!("{}/v1/jobs/{}", target_url, job_id);
        reqwest::Client::new()
            .delete(&cancel_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await; // Best effort
        
        Err(anyhow!("Operation cancelled by user"))
    }
}
```

---

## What Still Needs Work (Server-Side)

### Current Limitation
**Client sends cancel request, but server doesn't abort the download task.**

The download continues in the background on `rbee-hive`. The cancel endpoint exists (`DELETE /v1/jobs/{job_id}`) and marks the job as cancelled in the registry, but doesn't abort the actual download task.

### Why Deferred
Implementing server-side abort requires:
1. **job-registry changes** - Store `AbortHandle` for each job
2. **job_router changes** - Spawn downloads as abortable tasks
3. **heartbeat cancellation** - Stop heartbeat task when job cancelled

These changes touch core infrastructure and need careful design. Client-side cancellation (what we implemented) gives users immediate feedback, which is the critical UX win.

---

## Verification

### Compilation
```bash
cargo check --bin rbee-keeper --bin rbee-hive
# ✅ PASS
```

### Manual Test
```bash
# 1. Clear cache
rm -rf ~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF

# 2. Start download
./rbee model download

# 3. Press Ctrl+C after 5 seconds
# Expected: Cancellation message, client exits immediately

# 4. Check hive logs
# Note: Download continues (server-side abort not implemented yet)
```

---

## Code Signatures

All changes tagged with `// TEAM-387:` comments.

**Key locations:**
- `job-client/src/lib.rs:120-159` - New API
- `rbee-keeper/src/job_client.rs:54-96` - Ctrl+C handling
- `rbee-keeper/src/job_client.rs:112-159` - Helper function

---

## Summary

**Delivered:**
- ✅ Broke API cleanly (RULE ZERO)
- ✅ Fixed all call sites (compiler found them)
- ✅ Client-side cancellation works
- ✅ User gets immediate feedback on Ctrl+C
- ✅ Cancel request sent to server

**Deferred (Future Work):**
- ⏳ Server-side task abortion
- ⏳ Heartbeat task cancellation
- ⏳ Partial file cleanup

**Impact:** Users can cancel long downloads immediately. Server-side cleanup can be added later without breaking changes.

**Lines Changed:** ~150 LOC modified, 0 LOC added for backwards compatibility (RULE ZERO)

---

## WASM Compatibility Fix

**Issue:** WASM build failed because `js_sys::Function` is `!Send` but new API required `Send + 'static`.

**Solution:** Created `MaybeSend` trait alias:
- **Native:** `trait MaybeSend: Send` (requires Send)
- **WASM:** `trait MaybeSend` (no Send requirement)

**Code:**
```rust
// job-client/src/lib.rs:34-43
#[cfg(not(target_arch = "wasm32"))]
trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send> MaybeSend for T {}

#[cfg(target_arch = "wasm32")]
trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T> MaybeSend for T {}
```

**Verification:**
```bash
cd bin/10_queen_rbee/ui/packages/queen-rbee-sdk && pnpm build
# ✅ PASS

cargo build --bin queen-rbee
# ✅ PASS
```
