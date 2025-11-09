# TEAM-452: Tier System REMOVED

## What Was Removed

### ❌ Tier Selection Menu
**Before:**
```
? Which tier to release?
  > commercial
    frontend
    llm-worker
    main
```

**After:**
```
? Which app to release?
  > gwc - Worker Catalog
    commercial - Commercial Site
    marketplace - Marketplace
    docs - Documentation
    keeper - rbee-keeper
    queen - queen-rbee
    hive - rbee-hive
```

### ❌ Tier in Output
**Before:**
```
📋 Preview:
Tier: frontend
Bump: Patch
```

**After:**
```
📋 Preview:
App: gwc
Bump: Patch
```

### ❌ Tier in Git Commits
**Before:**
```
git commit -m "chore: release frontend v0.3.1"
```

**After:**
```
git commit -m "chore: release gwc v0.3.1"
```

### ❌ Tier CLI Argument
**Before:**
```bash
cargo xtask release --tier frontend --type patch
```

**After:**
```bash
cargo xtask release --type patch
# (Uses interactive menu to select app)
```

## What Stayed (Internal Only)

Tiers still exist internally in `.version-tiers/` for config loading, but users NEVER see them.

The code maps apps to tiers internally:
- gwc/commercial/marketplace/docs → frontend tier
- keeper/queen/hive → main tier

**Users don't know or care about tiers anymore!**

## Files Changed

- `xtask/src/release/cli.rs` - Removed tier menu, tier output, tier in commits
- `xtask/src/cli.rs` - Hidden tier CLI argument

## Verification

```bash
cargo xtask release --help
# No mention of tiers!

cargo xtask release
# Shows app list directly!
```

---

**TEAM-452: Tiers are dead. Long live apps!**
