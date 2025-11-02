# TEAM-379: Sync API Migration Plan

## Why Switch to Sync API?

**Current Problem:**
- Using `hf-hub` tokio async API
- No progress callbacks available
- Only get heartbeat messages every 10 seconds
- Can't show real download progress

**Solution:**
- Switch to `hf-hub` sync API
- Use `hf-hub-simple-progress` for real-time progress
- Wrap in `tokio::task::spawn_blocking` to stay async-compatible
- Get real progress: "Downloaded 234 MB / 640 MB (36.5%)"

## Why We're Async (Currently)

Looking at the call chain:
1. `VendorSource::download()` - async trait method
2. `HuggingFaceVendor::download()` - async implementation
3. Called from `ModelProvisioner::provision()` - async
4. Called from `job_router.rs` - async context

**BUT:** The actual HuggingFace download is just blocking I/O!
- We're using `repo.get()` which is a network download
- It blocks the entire time anyway
- We're not doing any concurrent operations during the download

**Conclusion:** We should use sync API + `spawn_blocking`!

## Migration Plan

### Phase 1: Update HuggingFaceVendor to Sync API ✅ READY

**Changes:**
1. Switch from `hf_hub::api::tokio::Api` to `hf_hub::api::sync::Api`
2. Add `hf-hub-simple-progress` dependency
3. Wrap download in `tokio::task::spawn_blocking`
4. Use progress callback to update `DownloadTracker`

**Files:**
- `Cargo.toml` - Add `hf-hub-simple-progress = "0.1.2"`
- `huggingface.rs` - Switch to sync API

### Phase 2: Implement Progress Callback

**Code Pattern:**
```rust
// In huggingface.rs download()
let tracker_clone = tracker.clone();
let api_clone = self.api.clone(); // Arc<sync::Api>
let repo_id_clone = repo_id.to_string();
let filename_clone = filename.clone();

let download_result = tokio::select! {
    _ = cancel_token.cancelled() => {
        Err(anyhow::anyhow!("Download cancelled by user"))
    }
    result = tokio::task::spawn_blocking(move || {
        let repo = api_clone.model(repo_id_clone);
        
        // Progress callback updates tracker in real-time
        let callback = callback_builder(move |progress: ProgressEvent| {
            // ProgressEvent has: url, percentage, elapsed_time, remaining_time
            // Update tracker with percentage (we don't have total_bytes)
            // The heartbeat will pick this up and show it
        });
        
        repo.download_with_progress(&filename_clone, callback)
    }) => {
        result
            .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
            .map_err(|e| anyhow::anyhow!("HuggingFace download failed: {}", e))
    }
};
```

### Phase 3: Update DownloadTracker to Show Percentage

**Current Heartbeat:**
```
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (0.00 MB downloaded)...
```

**After Migration:**
```
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (36.5% complete)...
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (72.3% complete)...
```

**Changes:**
- Add `percentage` field to `DownloadTracker`
- Update `update_progress()` to accept percentage
- Modify heartbeat message to show percentage

### Phase 4: Handle Cancellation in Blocking Task

**Challenge:** `spawn_blocking` task can't be cancelled mid-execution

**Solutions:**

**Option A: Accept it (RECOMMENDED)**
- Downloads are usually fast enough
- Cancellation happens between chunks
- User sees "cancelling..." message
- Good enough for v0.1.0

**Option B: Periodic cancellation checks**
- Would need to modify hf-hub download loop
- Too invasive
- Not worth it

**Decision:** Go with Option A

## Implementation Steps

### Step 1: Update Cargo.toml
```toml
hf-hub = { version = "0.4", features = [] }  # Remove "tokio" feature
hf-hub-simple-progress = "0.1.2"
```

### Step 2: Update HuggingFaceVendor struct
```rust
pub struct HuggingFaceVendor {
    api: Arc<hf_hub::api::sync::Api>,  // Wrap in Arc for clone
}

impl HuggingFaceVendor {
    pub fn new() -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::new().build()?;
        Ok(Self { api: Arc::new(api) })
    }
}
```

### Step 3: Update find_gguf_file()
```rust
async fn find_gguf_file(&self, repo_id: &str, cancel_token: &CancellationToken) -> Result<String> {
    let api_clone = self.api.clone();
    let repo_id_clone = repo_id.to_string();
    let cancel_token_clone = cancel_token.clone();
    
    tokio::task::spawn_blocking(move || {
        let repo = api_clone.model(repo_id_clone.clone());
        // ... existing logic ...
    })
    .await
    .map_err(|e| anyhow::anyhow!("Task join error: {}", e))?
}
```

### Step 4: Update download() with progress
```rust
async fn download(...) -> Result<u64> {
    // ... setup ...
    
    let tracker_clone = tracker.clone();
    let callback = callback_builder(move |progress: ProgressEvent| {
        // Update tracker with percentage
        // Heartbeat will show it
    });
    
    let download_result = tokio::select! {
        _ = cancel_token.cancelled() => {
            Err(anyhow::anyhow!("Download cancelled by user"))
        }
        result = tokio::task::spawn_blocking(move || {
            let repo = api_clone.model(repo_id_clone);
            repo.download_with_progress(&filename_clone, callback)
        }) => {
            result??
        }
    };
    
    // ... rest ...
}
```

### Step 5: Update DownloadTracker
```rust
pub struct DownloadTracker {
    bytes_downloaded: Arc<AtomicU64>,
    total_size: Option<u64>,
    percentage: Arc<AtomicU64>,  // Store as u64 (0-10000 for 0.00-100.00%)
    // ...
}

impl DownloadTracker {
    pub fn update_percentage(&self, pct: f64) {
        // Store percentage * 100 as integer (36.5% = 3650)
        let pct_int = (pct * 100.0) as u64;
        self.percentage.store(pct_int, Ordering::Relaxed);
    }
    
    pub fn current_progress(&self) -> DownloadProgress {
        let pct_int = self.percentage.load(Ordering::Relaxed);
        let percentage = if pct_int > 0 {
            Some(pct_int as f64 / 100.0)
        } else {
            None
        };
        
        DownloadProgress {
            bytes_downloaded: self.bytes_downloaded.load(Ordering::Relaxed),
            total_size: self.total_size,
            percentage,
        }
    }
}
```

### Step 6: Update heartbeat message
```rust
// In download_tracker.rs start_heartbeat()
if let Some(pct) = progress.percentage {
    n!(
        "download_progress",
        "⏳ Still downloading {} ({:.1}% complete)...",
        artifact_name,
        pct
    );
} else {
    n!(
        "download_progress",
        "⏳ Still downloading {}...",
        artifact_name
    );
}
```

## Testing Strategy

### Unit Tests
- ✅ Existing tests should still pass (no API changes)
- Add test for percentage tracking

### Manual Testing
```bash
# Delete cached model
rm -rf ~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF

# Download and watch progress
./rbee model download
```

**Expected Output:**
```
📥 Downloading model 'TheBloke/TinyLlama...' from HuggingFace
📥 Downloading file: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (12.3% complete)...
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (45.7% complete)...
⏳ Still downloading tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (78.2% complete)...
✅ Model ready: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (0.64 GB)
```

## Benefits

✅ **Real Progress** - See actual percentage  
✅ **Better UX** - Users know how long to wait  
✅ **Same Cancellation** - Still works with `tokio::select!`  
✅ **No Breaking Changes** - API stays the same  
✅ **Reusable** - Pattern works for all vendors  

## Risks & Mitigations

### Risk 1: spawn_blocking overhead
**Impact:** Minimal - we're downloading GB files, thread spawn is negligible  
**Mitigation:** None needed

### Risk 2: Cancellation delay
**Impact:** Download continues until current chunk completes  
**Mitigation:** Document behavior, acceptable for v0.1.0

### Risk 3: Thread pool exhaustion
**Impact:** Multiple concurrent downloads could block tokio threads  
**Mitigation:** 
- Downloads are rare (not hundreds per second)
- Tokio's blocking pool auto-scales
- Not a concern for typical usage

## Timeline

**Estimated Time:** 30-45 minutes

1. Update Cargo.toml (2 min)
2. Update HuggingFaceVendor struct (5 min)
3. Update find_gguf_file() (5 min)
4. Update download() with progress (10 min)
5. Update DownloadTracker (10 min)
6. Update heartbeat message (3 min)
7. Test (10 min)

## Decision

**Recommendation:** ✅ **DO IT**

The benefits far outweigh the minimal risks. Real progress tracking is a huge UX improvement, and the implementation is straightforward.

**Ready to implement?** Just say the word! 🚀
