# Mac Compatibility Fixes

**Branch**: `mac-compat`  
**Status**: ✅ Build working, GUI launching  
**Team**: TEAM-XXX

## Summary

This document tracks all changes made to ensure rbee builds and runs on macOS (Darwin). The project was originally developed on Linux (Arch/Ubuntu) and had several platform-specific dependencies.

## Fixed Issues

### 1. ✅ Build System - WASM Build Order

**Problem**: TypeScript compilation failed because WASM modules weren't generated yet.

**Files Changed**:
- `bin/79_marketplace_core/marketplace-node/package.json`

**Fix**:
```json
// Before: "build": "tsc && pnpm build:wasm"
// After:  "build": "pnpm build:wasm && tsc"
```

**Reason**: TypeScript imports `'../wasm/marketplace_sdk'` which must exist before `tsc` runs.

---

### 2. ✅ Build System - wasm-bindgen Race Condition

**Problem**: Multiple wasm-pack invocations tried to install wasm-bindgen-cli concurrently, causing "failed to move wasm-bindgen" errors on macOS.

**Files Changed**:
- `scripts/build-all.sh`

**Fix**:
```bash
# TEAM-XXX: mac compat - Ensure cargo bin is on PATH and wasm-bindgen is installed to avoid wasm-pack race conditions
if [[ -d "$HOME/.cargo/bin" ]]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if ! command -v wasm-bindgen &> /dev/null; then
  echo "→ Installing wasm-bindgen CLI (via cargo) to avoid wasm-pack auto-install conflicts..."
  cargo install wasm-bindgen-cli || true
fi
```

**Reason**: wasm-pack auto-installs wasm-bindgen if missing. Multiple parallel builds caused file conflicts on macOS.

---

### 3. ✅ Rust - u32/u64 Type Mismatch (Darwin statvfs)

**Problem**: `nix::sys::statvfs` returns `u32` on Darwin but `u64` on Linux, causing compilation errors.

**Files Changed**:
- `bin/98_security_crates/audit-logging/src/writer.rs`

**Fix**:
```rust
// TEAM-XXX: mac compat - compute as u64 to avoid u32/u64 mismatch on Darwin
let available: u64 = u64::from(stats.blocks_available()) * u64::from(stats.block_size());
```

**Reason**: Platform-specific integer sizes in `statvfs` API.

---

### 4. ✅ Rust - xtask Binary Path (Debug vs Release)

**Problem**: `./rbee` wrapper looked for `target/debug/rbee-keeper`, but `scripts/build-all.sh` uses `--release`.

**Files Changed**:
- `xtask/src/tasks/rbee.rs`

**Fix**:
```rust
// TEAM-XXX: mac compat - check both debug and release, prefer release (build-all.sh uses --release)
const TARGET_BINARY_DEBUG: &str = "target/debug/rbee-keeper";
const TARGET_BINARY_RELEASE: &str = "target/release/rbee-keeper";

// In run_rbee_keeper():
let binary_path = {
    let release_path = workspace_root.join(TARGET_BINARY_RELEASE);
    let debug_path = workspace_root.join(TARGET_BINARY_DEBUG);
    
    if release_path.exists() {
        release_path
    } else if debug_path.exists() {
        debug_path
    } else {
        anyhow::bail!(
            "rbee-keeper binary not found. Run 'cargo build --release' or 'cargo build --bin rbee-keeper'"
        );
    }
};
```

**Reason**: Build script uses `--release` profile, but xtask was hardcoded to debug.

---

### 5. ✅ Tauri - deep-link Plugin Config

**Problem**: Tauri deep-link plugin caused deserialization error on startup.

**Files Changed**:
- `bin/00_rbee_keeper/tauri.conf.json`
- `bin/00_rbee_keeper/src/main.rs`

**Fix**:
1. Removed deep-link config from `tauri.conf.json`
2. Commented out deep-link plugin initialization in `main.rs`

**Reason**: Plugin configuration format incompatibility. Temporarily disabled to unblock GUI launch.

**TODO**: Re-enable with correct format once plugin docs are verified.

---

### 6. ✅ Frontend Build - Empty dist Directory

**Problem**: rbee-keeper GUI failed to launch because frontend wasn't built.

**Files Changed**: None (build process fix)

**Fix**:
```bash
cd bin/00_rbee_keeper/ui && pnpm run build
```

**Reason**: Frontend build wasn't included in `scripts/build-all.sh`. The Rust build.rs handles this, but only when building rbee-keeper directly.

---

## Platform-Specific Code Inventory

### Linux-Only Features (Not Fixed - Gracefully Degraded)

The following features use `#[cfg(target_os = "linux")]` and are **intentionally Linux-only**:

1. **Process Monitoring** (`bin/25_rbee_hive_crates/monitor/`)
   - cgroup v2 integration
   - `/proc` filesystem access
   - nvidia-smi GPU monitoring
   - **macOS behavior**: Stubs return empty/default values

2. **Platform Module** (`bin/00_rbee_keeper/src/platform/`)
   - Linux: Uses `/proc/{pid}` for process checks
   - macOS: Has separate `macos.rs` implementation
   - Windows: Has separate `windows.rs` implementation

3. **Secrets Management** (`bin/98_security_crates/secrets-management/`)
   - systemd credential loading (Linux-only)
   - **macOS behavior**: Falls back to file-based secrets

### Cross-Platform Code

These modules work on all platforms:
- All WASM packages (marketplace-sdk, queen-rbee-sdk, etc.)
- HTTP clients and APIs
- Tauri GUI (with deep-link disabled)
- CLI commands (except platform-specific monitoring)

---

## Testing Checklist

### ✅ Build
- [x] `sh scripts/build-all.sh` completes without errors
- [x] Frontend builds successfully
- [x] Rust workspace builds in release mode
- [x] WASM packages generate correctly

### ✅ Execution
- [x] `./rbee --version` works
- [x] `./rbee` launches GUI (with deep-link disabled)
- [x] xtask finds release binary

### ⚠️ Known Limitations on macOS
- [ ] Deep-link plugin disabled (rbee:// protocol not working)
- [ ] Process monitoring returns stubs (no cgroup support)
- [ ] GPU monitoring unavailable (nvidia-smi Linux-only)
- [ ] systemd credentials not available (use file-based secrets)

---

## Verification Commands

```bash
# Full build
sh scripts/build-all.sh

# Test rbee wrapper
./rbee --version

# Launch GUI (should open window)
./rbee

# Build individual components
cargo build --release --bin rbee-keeper
cd bin/00_rbee_keeper/ui && pnpm run build
```

---

## Future Work

1. **Re-enable deep-link plugin**: Investigate correct Tauri v2 config format
2. **macOS process monitoring**: Implement using `libproc` or `sysctl`
3. **macOS GPU monitoring**: Investigate Metal performance APIs
4. **CI/CD**: Add macOS to GitHub Actions matrix

---

## References

- Tauri v2 docs: https://v2.tauri.app/
- nix crate Darwin support: https://docs.rs/nix/latest/nix/
- wasm-pack: https://rustwasm.github.io/wasm-pack/

---

**Last Updated**: 2025-11-05  
**Verified On**: macOS (Darwin)
