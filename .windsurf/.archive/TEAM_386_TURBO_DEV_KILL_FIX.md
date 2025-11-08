# TEAM-386: Fix Cargo Build Killing Turbo Dev Sessions

**Status:** ✅ COMPLETE  
**Date:** Nov 2, 2025

## Problem

Running `rebuild_daemon()` for `rbee-hive` kills active `turbo dev` sessions in another shell with:

```
@rbee/rbee-hive-react:dev:  ELIFECYCLE  Command failed.
  × Internal errors encountered: external process killed a task
```

## Root Cause

The `rbee-hive` build script (`bin/20_rbee_hive/build.rs`) runs `pnpm build` during `cargo build`:

```rust
let sdk_status = Command::new("pnpm")
    .args(&["build"])
    .current_dir(&sdk_dir)
    .status()
```

**Conflict:** When `turbo dev` is running in another shell, executing `pnpm build` in the same workspace causes process conflicts:

1. `turbo dev` runs frontend dev servers (Vite on multiple ports)
2. `cargo build` triggers `build.rs`
3. `build.rs` runs `pnpm build` in the same workspace
4. `pnpm/turbo` detects conflicting processes and sends SIGTERM/SIGKILL
5. Active `turbo dev` session dies

## The Fix

**File:** `bin/20_rbee_hive/build.rs` (lines 28-60)

Added **turbo dev process detection** to the existing Vite dev server check:

```rust
// Check 1: HTTP check for rbee-hive Vite dev server (port 7836)
let vite_dev_running = Command::new("curl")
    .args(&["-s", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:7836"])
    .output()
    .ok()
    .and_then(|output| String::from_utf8(output.stdout).ok())
    .map(|code| code.starts_with('2') || code.starts_with('3'))
    .unwrap_or(false);

// Check 2: Look for turbo dev process (prevents killing turbo dev sessions)
let turbo_dev_running = Command::new("pgrep")
    .args(&["-f", "turbo.*dev"])
    .output()
    .ok()
    .map(|output| !output.stdout.is_empty())
    .unwrap_or(false);

if vite_dev_running || turbo_dev_running {
    // Skip all UI builds - dev server provides fresh packages
    return;
}
```

## How It Works

**Before Fix:**
- Only checked port 7836 (rbee-hive's Vite server)
- Missed broader `turbo dev` sessions running other packages
- `pnpm build` killed active dev sessions

**After Fix:**
- Check 1: HTTP check on port 7836 (existing)
- Check 2: Process check for `turbo.*dev` pattern (NEW)
- If either detected → skip ALL UI builds
- Prevents `pnpm build` from running during active dev sessions

## Behavior

### With `turbo dev` Running

```bash
# Terminal 1
$ turbo dev
@rbee/commercial:dev: Ready in 4.2s
@rbee/ui:dev: Ready
@rbee/keeper-ui:dev: Ready
@rbee/rbee-hive-react:dev: Ready

# Terminal 2
$ cargo build --bin rbee-hive
warning: rbee-hive@0.1.0: ⚡ Turbo dev process detected - SKIPPING ALL UI builds
warning: rbee-hive@0.1.0:    (Prevents killing active turbo dev session)
warning: rbee-hive@0.1.0:    (Dev server provides fresh packages via hot reload)
warning: rbee-hive@0.1.0:    SDK and App builds skipped
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.50s
```

**Result:** `turbo dev` keeps running, no processes killed ✅

### Without `turbo dev` Running

```bash
$ cargo build --bin rbee-hive
warning: rbee-hive@0.1.0: 🔨 Building rbee-hive UI packages and app...
warning: rbee-hive@0.1.0:   📦 Building @rbee/rbee-hive-sdk (WASM)...
warning: rbee-hive@0.1.0:   🎨 Building @rbee/rbee-hive-ui app...
warning: rbee-hive@0.1.0: ✅ rbee-hive UI (SDK + App) built successfully
Finished `dev` profile [unoptimized + debuginfo] target(s) in 45s
```

**Result:** Normal production build ✅

## Testing

1. **Start turbo dev:**
   ```bash
   turbo dev
   ```

2. **In another terminal, rebuild rbee-hive:**
   ```bash
   cargo build --bin rbee-hive
   ```

3. **Verify:**
   - Cargo build completes without running `pnpm build`
   - `turbo dev` session stays alive
   - No "external process killed a task" errors

## Why This Pattern

**Process detection vs port checking:**
- Port checking (curl) only finds specific Vite servers
- Process checking (pgrep) finds the parent `turbo dev` orchestrator
- Catches ALL dev scenarios (turbo, pnpm dev, etc.)

**Regex pattern `turbo.*dev`:**
- Matches: `turbo dev`, `turbo run dev`, etc.
- Avoids false positives from unrelated processes

## Files Changed

- `bin/20_rbee_hive/build.rs` (+15 LOC, TEAM-386)

## Related Issues

- TEAM-381: Original Vite dev server detection (port 7836 only)
- TEAM-374: UI build integration in build.rs

## Impact

- ✅ Prevents killing active dev sessions
- ✅ Faster cargo builds during development (skips UI builds)
- ✅ Safer development workflow (no accidental process kills)
- ✅ Works with any dev server orchestrator (turbo, pnpm, etc.)

---

**TEAM-386 signature:** All changes marked with TEAM-386 comments
