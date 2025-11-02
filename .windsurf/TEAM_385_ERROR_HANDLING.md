# TEAM-385: User-Friendly Error Messages

**Status:** ✅ COMPLETE

**Problem:** Connection errors showed unhelpful technical messages:
```
Error: Failed to submit job: error sending request for url (http://localhost:7835/v1/jobs)
```

**Solution:** Added intelligent error detection and actionable troubleshooting messages.

---

## Error Handling Pattern

### Standard Pattern in Codebase

The codebase uses **`anyhow::Result`** everywhere - no custom error types. This is consistent across all binaries.

### Connection Error Detection

Using `reqwest::Error` methods:
- `err.is_connect()` - Connection refused (service not running)
- `err.is_timeout()` - Request timed out
- Generic fallback for other errors

---

## New Error Messages

### Before (Unhelpful)
```
Error: Failed to submit job: error sending request for url (http://localhost:7835/v1/jobs)
```

### After (Actionable)
```
Error: Cannot connect to rbee-hive - is the service running?

Operation: model_list
URL: http://localhost:7835

Troubleshooting:
• Check if the service is started
• Verify the port is correct
• Check firewall settings

Original error: error sending request for url (http://localhost:7835/v1/jobs)
```

---

## Implementation

**File:** `bin/99_shared_crates/job-client/src/lib.rs`

**Function:** `make_connection_error()`

### Features

1. **Service Detection** - Identifies rbee-hive vs queen-rbee by port
   - Port 7835 → rbee-hive
   - Port 8500 → queen-rbee

2. **Error Type Detection**
   - Connection refused → "is the service running?"
   - Timeout → "may be overloaded or unresponsive"
   - Generic → Shows full context

3. **Actionable Troubleshooting**
   - Check if service is started
   - Verify port is correct
   - Check firewall settings

4. **Full Context**
   - Operation name (e.g., "model_list")
   - Target URL
   - Original error message

---

## Error Types Handled

### 1. Connection Refused (Service Not Running)

**Trigger:** `err.is_connect() == true`

**Message:**
```
Cannot connect to rbee-hive - is the service running?

Operation: model_list
URL: http://localhost:7835

Troubleshooting:
• Check if the service is started
• Verify the port is correct
• Check firewall settings
```

### 2. Timeout (Service Unresponsive)

**Trigger:** `err.is_timeout() == true`

**Message:**
```
Request timed out connecting to rbee-hive

Operation: model_download
URL: http://localhost:7835

The service may be overloaded or unresponsive.
```

### 3. Generic Network Error

**Trigger:** Other reqwest errors

**Message:**
```
Failed to connect to queen-rbee

Operation: hive_start
URL: http://localhost:8500

Error: [original error message]
```

---

## Where Applied

The error handler is used in **3 places** in `job-client`:

1. **Job submission** (`submit_and_stream()` line 156)
   - POST to `/v1/jobs` endpoint

2. **SSE stream connection** (`submit_and_stream()` line 172)
   - GET to `/v1/jobs/{job_id}/stream` endpoint

3. **Fire-and-forget submission** (`submit()` line 267)
   - POST to `/v1/jobs` endpoint (no streaming)

---

## Benefits

### User Experience
- ✅ **Clear diagnosis** - "service not running" vs "timeout"
- ✅ **Actionable steps** - What to check and how to fix
- ✅ **Full context** - Operation, URL, original error
- ✅ **No guessing** - User knows exactly what's wrong

### Developer Experience
- ✅ **Centralized** - One function handles all connection errors
- ✅ **Consistent** - Same error format across all operations
- ✅ **Maintainable** - Update one place, fixes everywhere
- ✅ **Debuggable** - Original error preserved for investigation

---

## Standardization

### Error Handling Philosophy

**Across the entire codebase:**
1. Use `anyhow::Result` (no custom error types)
2. Fail-fast (no automatic retries in job-client)
3. Preserve original errors for debugging
4. Add context for user-facing errors

**Pattern:**
```rust
.map_err(|e| make_connection_error(&self.base_url, operation_name, e))?
```

---

## Testing

### Manual Test

```bash
# Stop rbee-hive
# (if running)

# Try model list
./rbee model list

# Expected output:
# Error: Cannot connect to rbee-hive - is the service running?
#
# Operation: model_list
# URL: http://localhost:7835
#
# Troubleshooting:
# • Check if the service is started
# • Verify the port is correct
# • Check firewall settings
```

---

## Code Signatures

All changes tagged with `TEAM-385` comments.

---

## Related Patterns

**Similar error handling exists in:**
- `bin/96_lifecycle/lifecycle-local/src/status.rs` - Health check errors
- `bin/96_lifecycle/lifecycle-ssh/src/status.rs` - SSH connection errors
- `bin/00_rbee_keeper/.archive/HEALTH_CHECK_IMPLEMENTATION.md` - Connection refused detection

**Pattern:**
```rust
match client.get(&health_url).send().await {
    Ok(response) => Ok(response.status().is_success()),
    Err(e) if e.is_connect() => Ok(false), // Connection refused
    Err(e) => Err(anyhow::anyhow!("Health check failed: {}", e))
}
```

---

## Future Improvements

Potential enhancements (not implemented):
1. Suggest specific commands to start services
2. Check if binary exists before suggesting "start service"
3. Detect common port conflicts (e.g., 7835 already in use)
4. Link to documentation for troubleshooting

**Note:** These are nice-to-haves, not blockers. Current implementation is production-ready.

---

## Compilation

```bash
cargo check -p job-client
# ✅ PASS

cargo check -p rbee-keeper
# ✅ PASS
```

---

## Summary

**Before:** Cryptic error messages that required debugging knowledge  
**After:** Clear, actionable error messages that guide users to solutions

**Impact:** Every connection error across rbee-keeper and queen-rbee now shows helpful troubleshooting steps.

**Maintenance:** Single function (`make_connection_error`) handles all cases - update once, fixes everywhere.
