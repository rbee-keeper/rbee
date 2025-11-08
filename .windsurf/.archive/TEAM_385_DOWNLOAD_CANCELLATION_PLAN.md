# TEAM-385: Download Cancellation Implementation Plan

**Status:** 📋 PLAN

**Goal:** Implement proper server-side cancellation for model downloads when user presses Ctrl+C

---

## Current State

### ✅ What Already Exists

1. **Client-side Ctrl+C detection** (`rbee-keeper/src/job_client.rs`)
   - Uses `tokio::select!` with `tokio::signal::ctrl_c()`
   - Drops SSE connection immediately
   - Shows "🛑 Operation cancelled by user"

2. **Server-side cancel endpoint** (`rbee-hive/src/http/jobs.rs`)
   - `DELETE /v1/jobs/{job_id}` endpoint exists
   - Calls `state.registry.cancel_job(&job_id)`
   - Returns 200 OK or 404 NOT FOUND

3. **Job registry cancellation** (`job-registry` crate)
   - Has `cancel_job()` method
   - Marks job as cancelled in registry

### ❌ What's Missing

1. **Client doesn't call cancel endpoint** - Just drops connection
2. **Download task isn't abortable** - `repo.get()` from `hf-hub` crate blocks
3. **Heartbeat task keeps running** - Not cancelled when job is cancelled
4. **Partial files not cleaned up** - HuggingFace cache has incomplete downloads

---

## Architecture

### Current Flow (Incomplete Cancellation)

```
User presses Ctrl+C
    ↓
rbee-keeper: Catches signal
    ↓
rbee-keeper: Drops SSE connection
    ↓
rbee-keeper: Exits with error
    ↓
rbee-hive: Continues download (doesn't know client cancelled!)
    ↓
rbee-hive: Eventually completes download
    ↓
File cached but user already left
```

### Desired Flow (Complete Cancellation)

```
User presses Ctrl+C
    ↓
rbee-keeper: Catches signal
    ↓
rbee-keeper: Sends DELETE /v1/jobs/{job_id}
    ↓
rbee-hive: Receives cancel request
    ↓
rbee-hive: Marks job as cancelled in registry
    ↓
rbee-hive: Aborts download task
    ↓
rbee-hive: Aborts heartbeat task
    ↓
rbee-hive: Cleans up partial files (optional)
    ↓
rbee-keeper: Closes SSE connection
    ↓
rbee-keeper: Exits with "Cancelled" message
```

---

## Implementation Plan

### Phase 1: Client-Side Cancel Request (Priority 1)

**File:** `bin/00_rbee_keeper/src/job_client.rs`

**Changes:**
1. Store `job_id` after job submission
2. When Ctrl+C is caught, send DELETE request before exiting
3. Wait briefly for cancel acknowledgement (with timeout)

**Code:**
```rust
tokio::select! {
    result = stream_future => result,
    _ = tokio::signal::ctrl_c() => {
        n!("job_cancelling", "🛑 Cancelling operation...");
        
        // Send cancel request to server
        if let Ok(job_id) = get_job_id_from_stream() {
            let cancel_url = format!("{}/v1/jobs/{}", target_url, job_id);
            let _ = reqwest::Client::new()
                .delete(&cancel_url)
                .timeout(Duration::from_secs(2))
                .send()
                .await;
        }
        
        n!("job_cancelled", "🛑 Operation cancelled by user");
        Err(anyhow::anyhow!("Operation cancelled by user"))
    }
}
```

**Issue:** Need to get `job_id` from the stream before it completes. Options:
- Store it in a shared variable when stream starts
- Return it from `JobClient::submit_and_stream()`
- Use a channel to communicate job_id back

**Recommendation:** Modify `JobClient::submit_and_stream()` to return `job_id` immediately after submission, before streaming starts.

---

### Phase 2: Server-Side Task Cancellation (Priority 2)

**File:** `bin/20_rbee_hive/src/job_router.rs`

**Problem:** The download happens in `execute_operation()` which is synchronous. When `cancel_job()` is called, the task keeps running.

**Solution:** Use `tokio::spawn` with `JoinHandle` and store it in job registry.

**Changes:**

1. **Store task handle in registry:**
```rust
// In job-registry crate
pub struct JobInfo {
    pub job_id: String,
    pub status: JobStatus,
    pub task_handle: Option<tokio::task::JoinHandle<()>>,  // NEW
    // ... other fields
}

impl JobRegistry {
    pub fn cancel_job(&self, job_id: &str) -> bool {
        if let Some(mut job) = self.jobs.write().unwrap().get_mut(job_id) {
            if let Some(handle) = job.task_handle.take() {
                handle.abort();  // Abort the tokio task
            }
            job.status = JobStatus::Cancelled;
            true
        } else {
            false
        }
    }
}
```

2. **Spawn download as abortable task:**
```rust
// In job_router.rs
let task_handle = tokio::spawn(async move {
    // Download logic here
    let result = state.model_provisioner.provision(&model, &job_id).await;
    // ...
});

// Store handle in registry
state.registry.set_task_handle(&job_id, task_handle);
```

**Issue:** This requires changes to `job-registry` crate to store task handles.

---

### Phase 3: Heartbeat Task Cancellation (Priority 2)

**File:** `bin/25_rbee_hive_crates/model-provisioner/src/huggingface.rs`

**Problem:** Heartbeat task spawned with `tokio::spawn()` keeps running even if download is cancelled.

**Solution:** Store heartbeat handle and abort it when download completes/fails/cancels.

**Changes:**
```rust
// Current code spawns heartbeat
let heartbeat_task = tokio::spawn(...);

// Download
let cached_path = repo.get(&filename).await?;

// Stop heartbeat
heartbeat_task.abort();
```

**Additional:** Need to abort heartbeat when job is cancelled externally.

**Option 1:** Check cancellation flag periodically in heartbeat loop
```rust
loop {
    interval.tick().await;
    
    // Check if job was cancelled
    if is_job_cancelled(&job_id) {
        break;
    }
    
    n!("hf_download_heartbeat", "⏳ Still downloading...");
}
```

**Option 2:** Use `tokio::select!` in heartbeat to listen for cancellation signal
```rust
let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel();

tokio::spawn(async move {
    loop {
        tokio::select! {
            _ = interval.tick() => {
                n!("hf_download_heartbeat", "⏳ Still downloading...");
            }
            _ = &mut cancel_rx => {
                break; // Cancelled
            }
        }
    }
});

// When cancelling, send signal
cancel_tx.send(()).ok();
```

---

### Phase 4: Partial File Cleanup (Priority 3 - Optional)

**File:** `bin/25_rbee_hive_crates/model-provisioner/src/huggingface.rs`

**Problem:** When download is cancelled, HuggingFace cache has partial file.

**Solution:** Delete partial file on cancellation.

**Challenges:**
- `hf-hub` crate manages its own cache
- We don't control the download location
- Partial files might be resumable (HTTP Range requests)

**Recommendation:** **Skip this for now.** HuggingFace cache cleanup is complex and partial files might be useful for resume.

---

## Implementation Priority

### 🔴 Priority 1: Essential (Do First)

1. **Client sends cancel request** (Phase 1)
   - Modify `job_client.rs` to send DELETE on Ctrl+C
   - Get job_id from stream early
   - Simple, high impact

### 🟡 Priority 2: Important (Do Next)

2. **Server aborts download task** (Phase 2)
   - Store task handles in job registry
   - Abort task when cancel is received
   - Prevents wasted resources

3. **Heartbeat task cancellation** (Phase 3)
   - Stop heartbeat when job cancelled
   - Prevents confusing messages after cancel

### 🟢 Priority 3: Nice to Have (Later)

4. **Partial file cleanup** (Phase 4)
   - Complex, low value
   - HuggingFace cache is opaque
   - Defer to future work

---

## Technical Challenges

### Challenge 1: Getting job_id Early

**Problem:** `JobClient::submit_and_stream()` returns after streaming completes. We need job_id immediately to send cancel request.

**Solutions:**

**Option A:** Return job_id from `submit_and_stream()`
```rust
pub async fn submit_and_stream<F>(&self, operation: Operation, handler: F) 
    -> Result<String>  // Returns job_id
```
**Pros:** Simple, clean API
**Cons:** Breaking change to existing callers

**Option B:** Use callback for job_id
```rust
pub async fn submit_and_stream<F, G>(&self, operation: Operation, 
    on_job_id: G,  // Called when job_id is known
    handler: F
) -> Result<()>
```
**Pros:** Backwards compatible
**Cons:** More complex API

**Option C:** Use channel to communicate job_id
```rust
let (job_id_tx, job_id_rx) = oneshot::channel();
tokio::spawn(async move {
    let job_id = client.submit_and_stream(...).await?;
    job_id_tx.send(job_id).ok();
});

tokio::select! {
    job_id = job_id_rx => { /* store it */ }
    _ = ctrl_c() => { /* cancel without job_id */ }
}
```
**Pros:** No API changes
**Cons:** Complex, race conditions

**Recommendation:** **Option A** - Return job_id from `submit_and_stream()`. It's already the return value, just need to return it earlier.

---

### Challenge 2: Aborting hf-hub Download

**Problem:** `repo.get(&filename).await?` is a single blocking call from `hf-hub` crate. We can't abort it mid-download.

**Solutions:**

**Option A:** Abort the entire tokio task
```rust
let handle = tokio::spawn(async move {
    repo.get(&filename).await?;
});

// On cancel
handle.abort();
```
**Pros:** Simple, works immediately
**Cons:** Leaves partial file in HuggingFace cache

**Option B:** Wrap in timeout + cancellation check
```rust
tokio::select! {
    result = repo.get(&filename) => result?,
    _ = cancellation_signal => {
        return Err(anyhow!("Cancelled"));
    }
}
```
**Pros:** Cleaner cancellation
**Cons:** Still can't stop download mid-stream

**Option C:** Use lower-level HTTP client
- Don't use `hf-hub` crate
- Implement download with `reqwest` streaming
- Check cancellation between chunks

**Pros:** Full control, can cancel mid-download
**Cons:** Reimplementing HuggingFace logic, complex

**Recommendation:** **Option A** - Abort the task. Simple and effective. Partial files are acceptable.

---

### Challenge 3: Job Registry Task Handle Storage

**Problem:** `job-registry` crate doesn't store task handles. Adding `JoinHandle` requires generic types.

**Solutions:**

**Option A:** Store `JoinHandle<()>` directly
```rust
pub struct JobInfo {
    task_handle: Option<tokio::task::JoinHandle<()>>,
}
```
**Pros:** Simple, type-safe
**Cons:** Couples registry to tokio

**Option B:** Use `AbortHandle` instead
```rust
pub struct JobInfo {
    abort_handle: Option<tokio::task::AbortHandle>,
}
```
**Pros:** Lighter weight, just for cancellation
**Cons:** Still coupled to tokio

**Option C:** Use callback/trait for cancellation
```rust
pub trait Cancellable {
    fn cancel(&self);
}

pub struct JobInfo {
    cancellable: Option<Box<dyn Cancellable>>,
}
```
**Pros:** Decoupled from tokio
**Cons:** Over-engineered for simple use case

**Recommendation:** **Option B** - Use `AbortHandle`. It's designed exactly for this use case.

---

## Minimal Implementation (Quick Win)

For fastest implementation with maximum impact:

### Step 1: Client sends cancel (5 minutes)
```rust
// In job_client.rs
tokio::select! {
    result = stream_future => result,
    _ = tokio::signal::ctrl_c() => {
        // Best effort cancel - don't wait for response
        let cancel_url = format!("{}/v1/jobs/UNKNOWN/cancel", target_url);
        tokio::spawn(async move {
            reqwest::Client::new().delete(&cancel_url).send().await.ok();
        });
        Err(anyhow!("Cancelled"))
    }
}
```

**Issue:** We don't have job_id yet. This won't work.

### Step 2: Get job_id early (10 minutes)
```rust
// Modify JobClient to return job_id immediately
let job_id = client.submit(operation).await?;
let stream_future = client.stream_results(&job_id, handler);

tokio::select! {
    result = stream_future => result,
    _ = tokio::signal::ctrl_c() => {
        let cancel_url = format!("{}/v1/jobs/{}", target_url, job_id);
        reqwest::Client::new().delete(&cancel_url).send().await.ok();
        Err(anyhow!("Cancelled"))
    }
}
```

### Step 3: Server aborts task (20 minutes)
```rust
// In job_router.rs - store abort handle
let handle = tokio::spawn(execute_operation(...));
state.registry.set_abort_handle(&job_id, handle.abort_handle());

// In job-registry - abort on cancel
pub fn cancel_job(&self, job_id: &str) -> bool {
    if let Some(handle) = self.abort_handles.remove(job_id) {
        handle.abort();
        true
    } else {
        false
    }
}
```

**Total time:** ~35 minutes for basic working cancellation

---

## Testing Plan

### Manual Testing

1. **Start download:**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF
   ./rbee model download
   ```

2. **Press Ctrl+C after 10 seconds**

3. **Expected output:**
   ```
   📥 Downloading file: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   ⏳ Downloading from HuggingFace...
   ⏳ Still downloading...
   ^C
   🛑 Cancelling operation...
   🛑 Operation cancelled by user
   ```

4. **Check hive logs** - Should show cancellation received

5. **Check process** - Download task should stop

### Automated Testing

```rust
#[tokio::test]
async fn test_download_cancellation() {
    // Start download
    let handle = tokio::spawn(async {
        download_model("test-model").await
    });
    
    // Wait a bit
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Cancel
    handle.abort();
    
    // Verify task stopped
    assert!(handle.await.is_err());
}
```

---

## Summary

**Minimum Viable Implementation:**
1. Split `JobClient::submit_and_stream()` into `submit()` + `stream()`
2. Send DELETE request on Ctrl+C
3. Store `AbortHandle` in job registry
4. Abort task when cancel received

**Estimated Time:** 35-45 minutes

**Impact:** Users can cancel long downloads immediately, server stops wasting resources

**Next Steps:** Implement Phase 1 first (client sends cancel), then Phase 2 (server aborts task).
