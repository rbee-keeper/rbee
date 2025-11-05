# TEAM-421: Unsafe Code Elimination

**Status:** ✅ COMPLETE

**Mission:** Replace unsafe `libc::kill` calls with safe `nix` crate wrappers

## Problem

Worker provisioner had 2 unsafe blocks for killing process groups:

```rust
unsafe {
    libc::kill(-(pid as i32), libc::SIGTERM);
}
```

**Issues with unsafe approach:**
- ❌ No type safety (raw i32 PIDs)
- ❌ No error handling (returns -1 on failure, ignored)
- ❌ Subtle semantics (negative PID = process group)
- ❌ Requires manual safety reasoning
- ❌ Triggers `-W unsafe-code` warnings

## Solution: `nix` Crate

Replaced `libc` with `nix` - the standard safe Unix syscall wrapper for Rust.

### Before (Unsafe)
```rust
#[cfg(unix)]
{
    if let Some(pid) = child.id() {
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        unsafe {
            libc::kill(-(pid as i32), libc::SIGKILL);
        }
    }
}
```

### After (Safe)
```rust
#[cfg(unix)]
{
    use nix::sys::signal::{killpg, Signal};
    use nix::unistd::Pid;
    
    if let Some(pid) = child.id() {
        // Send SIGTERM to process group (safe wrapper)
        if let Err(e) = killpg(Pid::from_raw(pid as i32), Signal::SIGTERM) {
            output_callback(&format!("==> Warning: SIGTERM failed: {}", e));
        }
        
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Send SIGKILL if still running
        if let Err(e) = killpg(Pid::from_raw(pid as i32), Signal::SIGKILL) {
            output_callback(&format!("==> Warning: SIGKILL failed: {}", e));
        }
    }
}
```

## Benefits

✅ **Type Safety:** `Pid` type prevents invalid PIDs
✅ **Error Handling:** Returns `Result`, errors are logged
✅ **No Unsafe:** Zero unsafe blocks
✅ **Explicit API:** `killpg()` clearly shows process group kill
✅ **Better Debugging:** Error messages show what went wrong
✅ **Standard Practice:** `nix` is the de facto Unix syscall wrapper

## Changes

### 1. Cargo.toml
```toml
[target.'cfg(unix)'.dependencies]
nix = { version = "0.29", features = ["signal", "process"] }
```

### 2. executor.rs
- Removed 2 unsafe blocks
- Added proper error handling
- Added error logging to output callback
- Updated TEAM signature (388 → 421)

## Verification

✅ **Compilation:** `cargo check --package rbee-hive-worker-provisioner` passes
✅ **No Unsafe:** `cargo clippy -- -W unsafe-code` shows zero warnings
✅ **Functionality:** Same behavior, better error reporting

## Is `nix` Overkill?

**No!** This is exactly what `nix` is designed for:

1. **Common Use Case:** Process group management is a standard Unix operation
2. **Safety Critical:** Signal handling bugs can cause zombie processes, hangs, or security issues
3. **Zero Cost:** `nix` is a thin wrapper with no runtime overhead
4. **Ecosystem Standard:** Used by tokio, nushell, alacritty, and hundreds of other projects
5. **Maintenance:** Better than maintaining custom unsafe FFI code

## Alternative Considered

**Keep unsafe blocks?** ❌ Bad idea because:
- Manual error checking required (check return value == -1)
- Easy to get wrong (negative PID convention is subtle)
- No type safety
- Fails `-W unsafe-code` lint

## Files Modified

1. `bin/25_rbee_hive_crates/worker-provisioner/Cargo.toml`
   - Replaced `libc = "0.2"` with `nix = { version = "0.29", features = ["signal", "process"] }`

2. `bin/25_rbee_hive_crates/worker-provisioner/src/pkgbuild/executor.rs`
   - Lines 324-351: Replaced unsafe blocks with safe `killpg()` calls
   - Added error handling and logging
   - Updated TEAM signature

## Impact

- **LOC:** ~30 lines modified
- **Unsafe Blocks:** 2 → 0
- **Error Handling:** None → Proper Result handling
- **Dependencies:** +1 (nix, Unix-only)

---

**TEAM-421 Complete** - Zero unsafe code in worker provisioner
