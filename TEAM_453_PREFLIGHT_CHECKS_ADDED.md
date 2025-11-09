# TEAM-453: Preflight Checks Added to Deployment Gates

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE

## Summary

Added preflight checks to all deployment gates to ensure dependencies are installed before running any tests or builds. This prevents failures due to missing dependencies.

## What Changed

### Before
```rust
pub fn check_gates(app: &str) -> Result<()> {
    println!("🚦 Running deployment gates for {}...", app);
    
    match app {
        "commercial" => check_commercial_gates()?,
        "marketplace" => check_marketplace_gates()?,
        // ... run gates directly
    }
}
```

### After
```rust
pub fn check_gates(app: &str) -> Result<()> {
    println!("🚦 Running deployment gates for {}...", app);
    
    // TEAM-453: Run preflight checks first
    println!("🔧 Preflight checks...");
    run_preflight_checks(app)?;
    
    match app {
        "commercial" => check_commercial_gates()?,
        "marketplace" => check_marketplace_gates()?,
        // ... run gates
    }
}
```

## Preflight Checks Function

### For Frontend Apps (pnpm)
```rust
"worker" | "gwc" | "worker-catalog" | "commercial" | "marketplace" | "docs" | "user-docs" => {
    println!("  📦 Installing pnpm dependencies...");
    pnpm install --frozen-lockfile
    println!("    ✅ Dependencies installed");
}
```

**What it does:**
- Runs `pnpm install --frozen-lockfile`
- Ensures all Node.js dependencies are installed
- Uses `--frozen-lockfile` to ensure reproducible builds
- Fails fast if dependencies can't be installed

### For Rust Apps (cargo)
```rust
"keeper" | "rbee-keeper" | "queen" | "queen-rbee" | "hive" | "rbee-hive" | 
"llm-worker" | "llm-worker-rbee" | "sd-worker" | "sd-worker-rbee" => {
    println!("  🦀 Checking Rust workspace...");
    cargo fetch
    println!("    ✅ Cargo dependencies fetched");
}
```

**What it does:**
- Runs `cargo fetch`
- Downloads all Rust dependencies
- Ensures cargo registry is up to date
- Fails fast if dependencies can't be fetched

## Benefits

### 1. Reproducible Builds
- `--frozen-lockfile` ensures exact dependency versions
- No surprises from dependency updates
- CI/CD builds match local builds

### 2. Fail Fast
- Catches missing dependencies immediately
- Clear error messages if install fails
- No wasted time running tests with missing deps

### 3. Clean State
- Fresh dependency install before each deployment
- Catches issues with lockfile changes
- Ensures all workspace dependencies are present

### 4. Better CI/CD
- Works in clean CI environments
- No need to manually run `pnpm install`
- Automatic dependency management

## Deployment Flow (Updated)

### Before (Missing Dependencies)
```
1. Run type-check → FAIL (missing dependencies)
2. User manually runs pnpm install
3. Re-run deployment
```

### After (Automatic)
```
1. Preflight: Install dependencies
2. Run type-check → SUCCESS
3. Run tests → SUCCESS
4. Run build → SUCCESS
5. Deploy → SUCCESS
```

## Example Output

### Frontend App Deployment
```bash
cargo xtask deploy --app marketplace --bump patch

🚦 Running deployment gates for marketplace...

🔧 Preflight checks...
  📦 Installing pnpm dependencies...
    ✅ Dependencies installed

🛒 Marketplace Gates:
  1. TypeScript type check...
  2. Unit tests...
  3. Production build...
  4. Build output validation...

✅ All deployment gates passed for marketplace
```

### Rust Binary Deployment
```bash
cargo xtask deploy --app keeper --bump patch

🚦 Running deployment gates for keeper...

🔧 Preflight checks...
  🦀 Checking Rust workspace...
    ✅ Cargo dependencies fetched

🐝 rbee-keeper Gates:
  1. Cargo check...
  2. Cargo test...
  3. Cargo clippy...
  4. Build test...

✅ All deployment gates passed for keeper
```

## Apps Affected

### Frontend Apps (pnpm install)
- ✅ worker / gwc / worker-catalog
- ✅ commercial
- ✅ marketplace
- ✅ docs / user-docs

### Rust Apps (cargo fetch)
- ✅ keeper / rbee-keeper
- ✅ queen / queen-rbee
- ✅ hive / rbee-hive
- ✅ llm-worker / llm-worker-rbee
- ✅ sd-worker / sd-worker-rbee

## Files Modified

1. `xtask/src/deploy/gates.rs`
   - Added `run_preflight_checks()` function
   - Updated `check_gates()` to call preflight checks
   - Added dependency installation logic

## Testing

### Verify Preflight Works
```bash
# Delete node_modules to test
rm -rf frontend/apps/marketplace/node_modules

# Run deployment - should auto-install
cargo xtask deploy --app marketplace --dry-run

# Should see:
# 🔧 Preflight checks...
#   📦 Installing pnpm dependencies...
#     ✅ Dependencies installed
```

### Verify Cargo Fetch Works
```bash
# Clear cargo cache
cargo clean

# Run deployment
cargo xtask deploy --app keeper --dry-run

# Should see:
# 🔧 Preflight checks...
#   🦀 Checking Rust workspace...
#     ✅ Cargo dependencies fetched
```

## Error Handling

### If pnpm install fails
```
Error: pnpm install failed
```
**Cause:** Missing lockfile, corrupted node_modules, or network issues  
**Fix:** Check pnpm-lock.yaml exists, try `pnpm install` manually

### If cargo fetch fails
```
Error: cargo fetch failed
```
**Cause:** Network issues, corrupted cargo registry, or invalid Cargo.toml  
**Fix:** Check internet connection, try `cargo fetch` manually

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Deploy Marketplace
  run: cargo xtask deploy --app marketplace --bump patch
  # No need to run pnpm install separately!
```

### Before (Manual)
```yaml
- name: Install dependencies
  run: pnpm install --frozen-lockfile
  
- name: Deploy
  run: cargo xtask deploy --app marketplace --bump patch
```

### After (Automatic)
```yaml
- name: Deploy
  run: cargo xtask deploy --app marketplace --bump patch
  # Dependencies installed automatically!
```

## Best Practices

### ✅ DO
- Let preflight checks handle dependency installation
- Use `--frozen-lockfile` for reproducible builds
- Commit lockfiles (pnpm-lock.yaml, Cargo.lock)

### ❌ DON'T
- Manually run `pnpm install` before deployment
- Skip lockfile commits
- Use `--no-frozen-lockfile` in CI

## Summary

✅ **Preflight checks added to all deployment gates**
- Automatic dependency installation
- Fail fast on missing dependencies
- Reproducible builds with frozen lockfiles
- Better CI/CD integration

All deployments now automatically ensure dependencies are installed before running any checks!
