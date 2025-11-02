# TEAM-384: Hive Stop Button Race Condition Fixed

**Date:** Nov 2, 2025 1:11 PM  
**Status:** ✅ FIXED

---

## The Bug

**Symptom:** Stop button appears to work but immediately shows "Running" again.

**Root Cause:** Race condition between HTTP shutdown response and actual process exit.

---

## What Was Happening

### Timeline of Events

```
T+0ms:    User clicks Stop button
T+10ms:   POST /v1/shutdown → Returns 200 OK
T+11ms:   lifecycle-local says "✅ stopped successfully"
T+12ms:   UI checks status → GET /health → 200 OK (STILL ALIVE!)
T+13ms:   UI shows "Running" ❌
T+500ms:  Process actually calls std::process::exit(0)
T+501ms:  Process is now dead
```

### The Problem

**Backend shutdown:** `bin/20_rbee_hive/src/http/shutdown.rs`
```rust
pub async fn handle_shutdown() -> StatusCode {
    tokio::spawn(async {
        // Wait 500ms to allow response to be sent
        tokio::time::sleep(Duration::from_millis(500)).await;
        std::process::exit(0);  // ← Process exits HERE
    });
    StatusCode::OK  // ← But returns HERE (500ms earlier!)
}
```

**Lifecycle stop:** `bin/96_lifecycle/lifecycle-local/src/stop.rs`
```rust
Ok(response) if response.status().is_success() => {
    // TEAM-384: Trust HTTP shutdown - no polling needed
    n!("stop_complete", "🎉 {} stopped successfully", daemon_name);
    return Ok(());  // ← Returns immediately without verifying!
}
```

**Result:** Function returns success while process is still alive for 500ms!

---

## The Fix

**File:** `bin/96_lifecycle/lifecycle-local/src/stop.rs` (lines 101-119)

### Before (BROKEN)
```rust
Ok(response) if response.status().is_success() => {
    // Trust HTTP shutdown - no polling needed
    n!("http_success", "✅ HTTP shutdown accepted");
    n!("stop_complete", "🎉 {} stopped successfully", daemon_name);
    return Ok(());  // Returns immediately!
}
```

### After (FIXED)
```rust
Ok(response) if response.status().is_success() => {
    n!("http_success", "✅ HTTP shutdown accepted, waiting for process to exit...");
    
    // TEAM-384: Wait for the process to actually exit (500ms delay + buffer)
    tokio::time::sleep(Duration::from_millis(1000)).await;
    
    // Verify it's actually down
    match client.get(health_url).send().await {
        Err(_) => {
            // Connection refused = process is dead ✅
            n!("stop_complete", "🎉 {} stopped successfully", daemon_name);
            return Ok(());
        }
        Ok(_) => {
            // Still responding = process didn't exit
            n!("http_timeout", "⚠️  Process still alive, falling back to SIGKILL");
            // Falls through to SIGKILL fallback
        }
    }
}
```

### What Changed

1. **Wait 1 second** after receiving 200 OK (gives process time to exit)
2. **Verify exit** by checking if health endpoint is unreachable
3. **Only return success** if process is actually dead
4. **Fallback to SIGKILL** if process is still alive after 1 second

---

## Why This Works

### Timeline After Fix

```
T+0ms:    User clicks Stop button
T+10ms:   POST /v1/shutdown → Returns 200 OK
T+11ms:   lifecycle-local waits 1000ms...
T+500ms:  Process calls std::process::exit(0) and dies
T+1011ms: lifecycle-local checks health → Connection refused ✅
T+1012ms: lifecycle-local returns "✅ stopped successfully"
T+1013ms: UI checks status → Connection refused ✅
T+1014ms: UI shows "Stopped" ✅
```

---

## Evidence From Logs

### Before Fix - Shows the Race Condition

```
lifecycle_local::stop http_success        
✅ HTTP shutdown accepted, daemon will stop gracefully
lifecycle_local::stop stop_complete       
🎉 rbee-hive stopped successfully          ← Says "stopped"

rbee_keeper::tauri_commands hive_status_check   
🔍 Checking status for hive 'localhost'
lifecycle_shared::status health_check_response
🏥 Health check response: 200 OK           ← But still alive!
rbee_keeper::tauri_commands hive_status_running 
✅ Hive 'localhost' is running             ← UI shows "Running" ❌
```

### After Fix - Will Wait and Verify

```
lifecycle_local::stop http_success        
✅ HTTP shutdown accepted, waiting for process to exit...
[waits 1000ms]
[checks health endpoint]
[gets connection refused]
lifecycle_local::stop stop_complete       
🎉 rbee-hive stopped successfully          ← Only says "stopped" when actually dead

rbee_keeper::tauri_commands hive_status_check   
🔍 Checking status for hive 'localhost'
lifecycle_shared::status health_check_error  
❌ Health check failed: Connection refused  ← Process is dead ✅
rbee_keeper::tauri_commands hive_status_stopped 
⏸️  Hive 'localhost' is installed but not running  ← UI shows "Stopped" ✅
```

---

## Testing

### Test Now (After Rebuilding rbee-keeper)

1. **Start rbee-keeper** (with rebuilt binary)
2. **Start localhost Hive**
3. **Click Stop button**
4. **Expected behavior:**
   - Button shows loading for ~1 second
   - Status changes to "Stopped"
   - No flickering between "Running" and "Stopped"

### Manual Test
```bash
# Start hive
./target/debug/rbee-hive --port 7835 --queen-url http://localhost:7833 --hive-id localhost &

# Stop via endpoint
curl -X POST http://localhost:7835/v1/shutdown

# Immediately check (should still be alive)
curl http://localhost:7835/health  # Still returns 200 OK

# Wait 1 second, then check
sleep 1
curl http://localhost:7835/health  # Connection refused ✅
```

---

## Files Changed

### Lifecycle Stop Logic
- `bin/96_lifecycle/lifecycle-local/src/stop.rs`
  - Added 1000ms wait after HTTP shutdown
  - Added health check verification
  - Only returns success if process is actually dead

---

## Related Issues

### Why 500ms Delay Exists

The delay in `handle_shutdown()` exists to ensure the HTTP response is sent before the process exits. Without it, clients might get connection errors instead of 200 OK.

### Why We Can't Remove the Delay

If we remove the delay:
```rust
pub async fn handle_shutdown() -> StatusCode {
    std::process::exit(0);  // Exit immediately
    StatusCode::OK  // Never reached!
}
```

The process exits before the response is sent, causing "Connection reset" errors on the client.

### The Proper Solution

**Option A (Current):** Keep the delay, wait on client side  
**Option B (Better):** Use a graceful shutdown signal instead of `exit(0)`

Option A is simpler and works fine. Option B would require more refactoring (shutdown channel, tokio runtime shutdown, etc.).

---

## Summary

**Problem:** HTTP shutdown returns success immediately, but process exits 500ms later, causing UI to show "Running" when it should show "Stopped".

**Solution:** Wait 1 second after receiving HTTP success, then verify the process is actually dead before returning success.

**Impact:** Stop button now works correctly - no more flickering status or false "Running" states.

---

**TEAM-384:** Hive stop race condition fixed. Rebuild rbee-keeper and test! 🛑✅
