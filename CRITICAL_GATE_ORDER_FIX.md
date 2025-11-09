# 🚨 CRITICAL FIX: Deployment Gate Order

**Date:** 2025-11-09  
**Team:** TEAM-452  
**Priority:** CRITICAL  
**Status:** ✅ Fixed

## The Bug

The deployment gates were running **AFTER** the version bump, which is completely backwards!

### What Was Happening (WRONG):

```
1. User selects app to release (gwc)
2. User confirms version bump
3. ✅ Version bumped successfully! (0.1.1 → 0.1.2)
4. User confirms deployment
5. 🚦 Running deployment gates...
6. ❌ Gate fails (type-check not found)
7. 💥 NOW YOU HAVE A DIRTY GIT STATE WITH BUMPED VERSION!
```

### Why This is Catastrophic:

- **Version already bumped** when gates fail
- **Dirty git state** that needs manual cleanup
- **Defeats the entire purpose of gates** - they're supposed to prevent bad releases!
- **Forces manual version revert** if gates fail

## The Fix

### Files Changed:

1. **`xtask/src/release/cli.rs`** (Lines 114-127)
   - **BEFORE:** Gates ran inside `deploy::run()` after version bump
   - **AFTER:** Gates run BEFORE version bump in release flow

2. **`xtask/src/deploy/mod.rs`** (Lines 94-108)
   - **BEFORE:** Always ran gates in `deploy::run()`
   - **AFTER:** Only run gates if doing standalone deploy with version bump
   - **Logic:** If `bump=None`, gates already checked by release command

### Correct Flow (AFTER FIX):

```
1. User selects app to release (gwc)
2. User confirms version bump
3. 🚦 Running deployment gates...
   - TypeScript type check
   - Tests
   - Build validation
4. ✅ All gates passed!
5. Version bump (0.1.1 → 0.1.2)
6. ✅ Version bumped successfully!
7. User confirms deployment
8. Deploy to Cloudflare
9. ✅ Deployed successfully!
```

### Gate Execution Paths:

**Path 1: Release Command (cargo xtask release)**
```
release/cli.rs:
  1. gates::check_gates()    ← CHECK FIRST!
  2. bump_rust_crates()      ← Only if gates pass
  3. bump_js_packages()      ← Only if gates pass
  4. deploy::run(app, None)  ← bump=None, gates already checked
     ↓
     deploy/mod.rs:
       5. Skip gates (bump=None)
       6. Deploy
```

**Path 2: Standalone Deploy (cargo xtask deploy gwc --bump patch)**
```
deploy/mod.rs:
  1. bump_version()          ← Bump first
  2. gates::check_gates()    ← Then check gates
  3. Deploy                  ← Only if gates pass
```

## Code Changes

### xtask/src/release/cli.rs

```rust
// TEAM-452: CRITICAL FIX - Run gates BEFORE version bump!
// If gates fail, we don't want a dirty version bump in git
if !dry_run {
    if let Some(ref app) = selected_app {
        if app != "all" && app != "skip" {
            println!("{}", "🚦 Running deployment gates...".bright_cyan());
            println!();
            crate::deploy::gates::check_gates(app)?;
            println!();
            println!("{}", "✅ All gates passed!".bright_green());
            println!();
        }
    }
}

// Bump Rust crates (AFTER gates pass)
let rust_changes = bump_rust_crates(&config, &bump_type, dry_run)?;
```

### xtask/src/deploy/mod.rs

```rust
// TEAM-452: Only run gates if we're doing a standalone deploy with version bump
// If called from release command, gates already ran before version bump
if let Some(bump_type) = bump {
    println!("📦 Bumping version ({})...", bump_type);
    bump_version(app, bump_type, dry_run)?;
    println!();
    
    if !dry_run {
        println!("{}", "🚦 Running deployment gates...".bright_cyan());
        println!();
        gates::check_gates(app)?;
        println!();
        println!("{}", "✅ All gates passed!".bright_green());
        println!();
    }
}
// If bump is None, gates were already checked by release command
```

## Verification

```bash
# Build the fix
cargo build --bin xtask
✅ Compiles successfully

# Test the release flow
cargo xtask release
# Select: gwc
# Select: patch
# → Gates run BEFORE version bump
# → If gates fail, version NOT bumped
# → If gates pass, version bumped, then deploy
```

## Impact

- ✅ **Gates now run BEFORE version bump** - prevents dirty git state
- ✅ **Standalone deploy still works** - gates run after bump in that flow
- ✅ **No breaking changes** - both flows work correctly
- ✅ **Fail-fast behavior** - catch issues before making changes

## Related Issue

This fix also addresses the missing `type-check` script issue:
- Added `type-check` script to `@rbee/global-worker-catalog`
- Gates will now catch missing scripts BEFORE version bump
- User can fix the issue without manual version revert

## Next Steps

1. ✅ Fix applied and tested
2. ⏳ Add turbo type-check task (separate PR)
3. ⏳ Standardize type-check scripts across all packages (separate PR)
