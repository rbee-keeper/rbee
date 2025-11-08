# Deployment Workflow Fix

**Problem:** Deployment and version bumping are separate commands  
**Why it's wrong:** Deployment should ALWAYS include version bumping  
**Status:** Needs to be fixed

---

## ❌ Current (Broken) Workflow

```bash
# Step 1: Bump version manually
cargo xtask release --tier frontend --type patch

# Step 2: Commit
git add .
git commit -m "chore: bump version"
git push

# Step 3: Deploy
cargo xtask deploy --app worker
```

**Problems:**
- ❌ Easy to forget version bump
- ❌ Easy to deploy without bumping
- ❌ Version mismatch between code and deployment
- ❌ Two separate commands for one logical operation

---

## ✅ Correct Workflow (Should Be)

```bash
# One command does everything
cargo xtask deploy --app worker --bump patch

# Behind the scenes:
# 1. Bump version in package.json
# 2. Run deployment gates
# 3. Build
# 4. Deploy to Cloudflare
# 5. Create git tag
# 6. Push tag
```

**Benefits:**
- ✅ One command
- ✅ Can't forget version bump
- ✅ Can't deploy without bumping
- ✅ Atomic operation
- ✅ Git tag created automatically

---

## 🔧 Implementation Plan

### 1. Update Deploy Command

```rust
// xtask/src/cli.rs
#[command(name = "deploy")]
Deploy {
    /// App to deploy
    #[arg(long)]
    app: String,
    
    /// Version bump type (patch, minor, major)
    /// If not provided, deploys current version (no bump)
    #[arg(long)]
    bump: Option<String>,
    
    /// Dry run
    #[arg(long)]
    dry_run: bool,
},
```

### 2. Update Deploy Logic

```rust
// xtask/src/deploy/mod.rs
pub fn run(app: &str, bump: Option<&str>, dry_run: bool) -> Result<()> {
    // Step 1: Bump version if requested
    if let Some(bump_type) = bump {
        println!("📦 Bumping version ({})...", bump_type);
        bump_version(app, bump_type)?;
    } else {
        println!("⚠️  Deploying current version (no bump)");
    }
    
    // Step 2: Run deployment gates
    if !dry_run {
        gates::check_gates(app)?;
    }
    
    // Step 3: Deploy
    match app {
        "worker" => worker_catalog::deploy(dry_run)?,
        // ... other apps
    }
    
    // Step 4: Create git tag (if bumped)
    if let Some(_) = bump {
        create_git_tag(app)?;
    }
    
    Ok(())
}
```

### 3. Separate `release` Command for Rust Binaries

```bash
# For Rust binaries (different workflow - build on multiple machines)
cargo xtask release --tier main --type minor

# This:
# 1. Bumps version in Cargo.toml
# 2. Commits
# 3. Tells you to build on mac and blep
# 4. Tells you to create GitHub release
```

---

## 📊 Comparison

| Operation | Current | Should Be |
|-----------|---------|-----------|
| **Deploy frontend app** | `release` + `deploy` | `deploy --bump patch` |
| **Deploy without bump** | `deploy` only | `deploy` (no --bump) |
| **Release Rust binary** | `release` + manual steps | `release` (unchanged) |

---

## 🎯 Usage Examples

### Deploy with version bump (normal)
```bash
cargo xtask deploy --app worker --bump patch
# Bumps 0.1.0 → 0.1.1, deploys, creates tag v0.1.1
```

### Deploy without bump (hotfix on same version)
```bash
cargo xtask deploy --app worker
# Deploys current version, no bump, no tag
```

### Deploy with dry run
```bash
cargo xtask deploy --app worker --bump minor --dry-run
# Shows what would happen, doesn't actually deploy
```

### Release Rust binaries (different workflow)
```bash
cargo xtask release --tier main --type minor
# Bumps version, commits, tells you next steps
```

---

## 🚀 Migration Path

### Phase 1: Add --bump flag (optional)
- Add `--bump` parameter to deploy command
- If provided, bump version before deploying
- If not provided, deploy current version
- Keep `release` command for Rust binaries

### Phase 2: Make --bump required for production
- Warn if deploying without --bump
- Eventually require --bump for production deploys

### Phase 3: Deprecate standalone version bumping for frontend
- `cargo xtask release` only for Rust binaries
- Frontend apps always use `cargo xtask deploy --bump`

---

## ✅ Benefits

**For developers:**
- ✅ One command to deploy
- ✅ Can't forget version bump
- ✅ Clear intent (--bump or no --bump)

**For CI/CD:**
- ✅ Atomic operation
- ✅ Git tags created automatically
- ✅ Version always matches deployment

**For debugging:**
- ✅ Git tag shows exact deployed version
- ✅ Can rollback to specific tag
- ✅ Clear deployment history

---

## 📝 Next Steps

1. Update `xtask/src/cli.rs` - Add `--bump` parameter
2. Update `xtask/src/deploy/mod.rs` - Implement version bumping
3. Create `bump_version()` function for frontend apps
4. Create `create_git_tag()` function
5. Update documentation
6. Test with dry-run
7. Deploy!

---

**Status:** Documented, ready to implement  
**Priority:** High - This is a fundamental workflow issue
