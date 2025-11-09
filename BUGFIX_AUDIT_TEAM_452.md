# TEAM-452: Complete Bugfix Audit

## Critical Bugs Found

### 🔴 BUG 1: `bump_js.rs` - Missing newline at end of file
**File:** `xtask/src/release/bump_js.rs:75-76`
**Issue:** Writing JSON without trailing newline breaks git diffs
```rust
let pretty = serde_json::to_string_pretty(&json)?;
fs::write(&package_json_path, pretty)?;  // ❌ No newline!
```
**Fix:** Add newline to match standard file format
```rust
let pretty = serde_json::to_string_pretty(&json)?;
fs::write(&package_json_path, format!("{}\n", pretty))?;  // ✅ With newline
```

### 🔴 BUG 2: `worker_catalog.rs` - Wrong file extension check
**File:** `xtask/src/deploy/worker_catalog.rs:14-18`
**Issue:** Checking for `wrangler.toml` but should check `wrangler.jsonc` (actual file format)
```rust
let wrangler_path = format!("{}/wrangler.toml", worker_dir);  // ❌ Wrong extension
```
**Evidence:** Worker catalog uses `wrangler.jsonc` not `wrangler.toml`
**Fix:** Check for correct file extension

### 🔴 BUG 3: `deploy/mod.rs` - Missing version bump for deploy
**File:** `xtask/src/deploy/mod.rs:87-98`
**Issue:** When deploying from release menu, `bump` is `None`, so version doesn't get bumped!
```rust
pub fn run(app: &str, bump: Option<&str>, dry_run: bool) -> Result<()> {
    let bump_type = bump.ok_or_else(|| {  // ❌ This errors when called from release!
        anyhow::anyhow!("Version bump is REQUIRED...")
    })?;
```
**Context:** In `release/cli.rs:173`, we call `crate::deploy::run(app, None, false)` 
**Fix:** Make bump optional and skip version bump if already done by release

### 🟡 BUG 4: `bump_rust.rs` - Silent failure for workspace versions
**File:** `xtask/src/release/bump_rust.rs:61-72`
**Issue:** Returns same version twice for workspace crates (no bump), but doesn't tell user
```rust
if version_value.is_table() || version_value.as_str() == Some("workspace") {
    // ...
    return Ok((version.clone(), version));  // ❌ Silent no-op
}
```
**Fix:** Either skip these crates entirely or warn user they weren't bumped

### 🟡 BUG 5: All deploy files - Duplicate code
**Files:** `commercial.rs`, `docs.rs`, `marketplace.rs`
**Issue:** All three have IDENTICAL structure - violates Rule Zero
- Same `create_env_file()` pattern
- Same build/deploy flow
- Only difference is URLs and project names
**Fix:** Create shared `deploy_pages_app()` function

### 🟡 BUG 6: `binaries.rs` - Hardcoded "mac" and "blep" hostnames
**File:** `xtask/src/deploy/binaries.rs:108-122`
**Issue:** SSH to "mac" and "blep" won't work for other users
```rust
Command::new("ssh")
    .args(&["mac", &format!("cd ~/Projects/rbee && ...")])  // ❌ Hardcoded
```
**Fix:** Use environment variables or config file for build hosts

### 🟡 BUG 7: `tiers.rs` - Unused `name` field warning
**File:** `xtask/src/release/tiers.rs:10`
**Issue:** Compiler warns about unused field
```rust
pub struct TierConfig {
    pub name: String,  // ❌ Never read
```
**Fix:** Either use it or remove it

## Medium Issues

### 🟠 ISSUE 1: Inconsistent error handling
- `bump_js.rs:26` - Prints warning and continues
- `bump_rust.rs:36` - Prints warning and continues
- `deploy/gates.rs` - Fails immediately on error
**Recommendation:** Be consistent - either fail fast everywhere or collect errors

### 🟠 ISSUE 2: No rollback on partial failure
**Scenario:** If deploying "all" frontend apps and commercial fails, the others still deploy
**File:** `release/cli.rs:163-173`
**Fix:** Either rollback on failure or make it transactional

### 🟠 ISSUE 3: Missing validation
- No check if `pnpm` is installed before deploying
- No check if `wrangler` is installed
- No check if `gh` CLI is installed for binaries
**Fix:** Add pre-flight checks

## Minor Issues

### 🔵 MINOR 1: Confusing variable names
**File:** `release/cli.rs:150`
```rust
let app = app_choice.split_whitespace().next().unwrap();
```
Reuses `app` variable name - confusing

### 🔵 MINOR 2: Magic numbers
**File:** `bump_js.rs:97`
```rust
.max_depth(10)  // ❌ Why 10?
```
Should be a constant with explanation

### 🔵 MINOR 3: Inconsistent dry-run messages
- Some say "would execute:"
- Some say "Dry run - would execute:"
- Some say "🔍 Dry run - would execute:"
**Fix:** Standardize format

## Summary

### Critical (Must Fix):
1. ✅ Missing newline in JSON files
2. ✅ Wrong file extension check
3. ✅ **BLOCKER:** Deploy from release menu fails (bump=None)

### Important (Should Fix):
4. Duplicate code in deploy files
5. Hardcoded hostnames
6. Unused struct field

### Nice to Have:
7. Consistent error handling
8. Rollback on failure
9. Pre-flight checks
10. Better variable names

## Recommended Fix Order

1. **FIX BUG 3 FIRST** - Deploy from release menu is broken!
2. Fix BUG 1 - JSON newlines
3. Fix BUG 2 - File extension
4. Fix BUG 5 - Deduplicate deploy code
5. Fix remaining issues

---

**TEAM-452: Found 7 bugs, 3 critical, 4 medium priority**
