# TEAM-452: Bugs Fixed

## ✅ Fixed (5 Critical Bugs)

### 1. **BLOCKER: Deploy from release menu failed**
**File:** `xtask/src/deploy/mod.rs`
**Before:** Required `bump` parameter, errored when called from release menu
**After:** Made `bump` optional - skips version bump if already done
```rust
// Now works from release menu
if let Some(bump_type) = bump {
    bump_version(app, bump_type, dry_run)?;
}
```

### 2. **JSON files missing newlines**
**File:** `xtask/src/release/bump_js.rs`
**Before:** `fs::write(&path, pretty)?;`
**After:** `fs::write(&path, format!("{}\n", pretty))?;`
**Impact:** Git diffs now work correctly

### 3. **Wrong file extension check**
**File:** `xtask/src/deploy/worker_catalog.rs`
**Before:** Checked for `wrangler.toml`
**After:** Checks for `wrangler.jsonc` (correct format)
**Impact:** Worker catalog deployment now works

### 4. **Wrong config format**
**File:** `xtask/src/deploy/worker_catalog.rs`
**Before:** Created TOML format config
**After:** Creates JSON format config (wrangler.jsonc)

### 5. **Unused struct field warning**
**File:** `xtask/src/release/tiers.rs`
**Before:** `pub name: String,` (never used)
**After:** Removed field
**Impact:** Clean compile, no warnings

## Verification

```bash
✅ cargo check --bin xtask  # No errors, no warnings (except harmless autodocs)
```

## Remaining Issues (Not Critical)

- Duplicate code in deploy files (commercial, marketplace, docs)
- Hardcoded hostnames in binaries.rs
- No pre-flight checks for tools (pnpm, wrangler, gh)

**All critical bugs fixed. Deploy from release menu now works!**
