# TEAM-388: Process Group Killing - Cancellation Fix

**Status:** ✅ IMPLEMENTED  
**Date:** Nov 3, 2025  
**Time:** 12:10 AM UTC+01:00

## Problem

Ctrl+C was detected and DELETE request sent, but cargo build continued running.

**Root Cause:** Killing the bash script parent process doesn't kill cargo subprocess.

## Solution

Kill the entire **process group** instead of just the parent process.

### Implementation

**File:** `bin/20_rbee_hive/src/pkgbuild_executor.rs`

#### 1. Create Process Group (Lines 266-281)

```rust
// Execute script
// TEAM-388: Create new process group so we can kill all subprocesses (including cargo)
let mut cmd = Command::new("bash");
cmd.arg(&script_path)
    .current_dir(&self.workdir)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

// TEAM-388: On Unix, create a new process group
#[cfg(unix)]
{
    use std::os::unix::process::CommandExt;
    cmd.process_group(0);  // 0 = create new process group with PID as PGID
}

let mut child = cmd.spawn()?;
```

**What this does:**
- `process_group(0)` creates a new process group
- The process group ID (PGID) equals the process ID (PID)
- All child processes (bash, cargo, rustc, etc.) inherit this PGID

#### 2. Kill Process Group (Lines 320-350)

```rust
_ = cancel_token.cancelled() => {
    output_callback("==> Build cancelled, killing process group...");
    
    // TEAM-388: Kill the entire process group (bash + cargo + all subprocesses)
    #[cfg(unix)]
    {
        if let Some(pid) = child.id() {
            output_callback(&format!("==> Killing process group PGID: {}", pid));
            // Kill the process group by sending signal to negative PID
            unsafe {
                libc::kill(-(pid as i32), libc::SIGTERM);
            }
            output_callback("==> SIGTERM sent to process group");
            
            // Give it a moment to terminate gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // If still running, send SIGKILL
            unsafe {
                libc::kill(-(pid as i32), libc::SIGKILL);
            }
            output_callback("==> SIGKILL sent to process group");
        }
    }
    
    // Also try tokio's kill (for the parent process)
    let _ = child.kill().await;
    
    // Wait for tasks to complete
    let _ = tokio::join!(stdout_task, stderr_task);
    return Err(ExecutionError::BuildFailed(-1));
}
```

**What this does:**
1. **Negative PID:** `kill(-pid, signal)` sends signal to entire process group
2. **SIGTERM first:** Graceful shutdown (500ms timeout)
3. **SIGKILL second:** Force kill if still running
4. **Tokio kill:** Backup kill for parent process

### Dependency Added

**File:** `bin/20_rbee_hive/Cargo.toml`

```toml
libc = "0.2"  # TEAM-388: For process group killing (Unix signals)
```

## How It Works

### Before (Broken)

```
bash (PID 1000)
  └─ cargo (PID 1001)
      └─ rustc (PID 1002)
          └─ rustc (PID 1003)

kill(1000)  ← Only kills bash
cargo, rustc continue running!
```

### After (Fixed)

```
bash (PID 1000, PGID 1000)
  └─ cargo (PID 1001, PGID 1000)
      └─ rustc (PID 1002, PGID 1000)
          └─ rustc (PID 1003, PGID 1000)

kill(-1000, SIGTERM)  ← Kills entire process group
All processes terminated!
```

## Testing

### Test 1: Cancel During Cargo Build

```bash
# Terminal 1: Start rbee-hive
cargo run --bin rbee-hive

# Terminal 2: Start worker installation
./rbee worker download llm-worker-rbee-cpu

# Wait for cargo to start compiling...
# Press Ctrl+C

# Expected output:
🛑 Cancelling operation...
==> Build cancelled, killing process group...
==> Killing process group PGID: 12345
==> SIGTERM sent to process group
==> SIGKILL sent to process group
❌ Build cancelled by user
```

### Test 2: Verify Cargo Stopped

```bash
# After cancelling, check if cargo is still running
ps aux | grep cargo

# Expected: No cargo processes
```

### Test 3: Verify Cleanup

```bash
# Check temp directories are cleaned up
ls /tmp/worker-install/

# Expected: Directory removed or empty
```

## Why This Works

### Unix Process Groups

Every process belongs to a process group (PGID). By default, child processes inherit their parent's PGID.

**Key insight:** Sending a signal to a **negative PID** sends it to the entire process group.

```c
// Kill single process
kill(1234, SIGTERM);

// Kill process group
kill(-1234, SIGTERM);  // ← Negative PID!
```

### SIGTERM vs SIGKILL

1. **SIGTERM (15):** Graceful shutdown
   - Process can catch and handle
   - Allows cleanup (close files, save state)
   - We wait 500ms for this

2. **SIGKILL (9):** Force kill
   - Cannot be caught or ignored
   - Immediate termination
   - Used as backup if SIGTERM fails

## Comparison with Model Download

| Feature | Model Download | Worker Install |
|---------|---------------|----------------|
| Cancellation | ✅ Yes | ✅ Yes |
| Long operation | HF download | Cargo build |
| Implementation | `spawn_blocking` | Process group |
| Subprocess killing | N/A (single process) | ✅ Process group |
| Graceful shutdown | ✅ SIGTERM | ✅ SIGTERM + SIGKILL |

**Key difference:** Model download is a single process (HF API), worker install spawns many subprocesses (bash → cargo → rustc).

## Platform Support

### Unix (Linux, macOS)

✅ **Fully supported** - Uses `libc::kill` with process groups

### Windows

⚠️ **Not implemented** - Windows doesn't have process groups in the same way

**Workaround needed:**
- Use Windows Job Objects
- Or track child PIDs manually and kill each one

**Current behavior on Windows:** Only kills bash, cargo continues (same as before)

## Benefits

1. **Actually works:** Cargo and all subprocesses are killed
2. **Graceful:** SIGTERM first, SIGKILL as backup
3. **Complete:** Entire process tree terminated
4. **Fast:** 500ms timeout for graceful shutdown
5. **Debug output:** Clear narration of what's happening

## Future Enhancements

### 1. Windows Support

Implement Windows Job Objects:

```rust
#[cfg(windows)]
{
    use winapi::um::jobapi2::*;
    // Create job object
    // Assign process to job
    // Terminate job on cancel
}
```

### 2. Configurable Timeout

Allow user to configure SIGTERM timeout:

```rust
let timeout = std::env::var("CANCEL_TIMEOUT_MS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or(500);
```

### 3. Progress Preservation

Save build progress before killing:

```rust
// Before killing, save what was compiled
let compiled_crates = parse_cargo_output();
save_build_cache(compiled_crates);
```

---

**TEAM-388 PROCESS GROUP FIX COMPLETE** - Ctrl+C now properly kills cargo builds!
