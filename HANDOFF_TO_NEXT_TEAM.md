# Handoff to Next Team

## User's Frustrations (LISTEN TO THESE)

1. **User wants to deploy GWC (worker catalog) easily**
   - Current: Must use `cargo xtask deploy --app worker --bump patch`
   - User expectation: Should be simpler, more obvious
   - Problem: Too many steps, confusing commands

2. **Tier system is confusing**
   - GWC is bundled with commercial, marketplace, docs in "frontend" tier
   - User wants each app to version independently
   - User doesn't understand why they're grouped
   - User didn't ask for bundling

3. **Too many half-assed commits**
   - Previous AI kept committing documentation instead of fixing code
   - User wants CODE CHANGES, not explanations
   - User wants WORKING SOLUTIONS, not plans

4. **Deploy workflow is confusing**
   - User ran `cargo xtask release` expecting to deploy
   - `release` is for version bumping (shows tiers)
   - `deploy` is for actual deployment
   - This split makes no sense to user

## What User Actually Wants

**Simple deployment command:**
```bash
# User wants something like this:
cargo xtask deploy gwc

# Or even simpler:
cargo xtask gwc

# Not this confusing mess:
cargo xtask deploy --app worker --bump patch
```

## Current State (What Works)

### Deploy Commands
```bash
# Worker catalog
cargo xtask deploy --app worker --bump patch

# Commercial site
cargo xtask deploy --app commercial --bump patch

# Marketplace
cargo xtask deploy --app marketplace --bump patch

# Docs
cargo xtask deploy --app docs --bump patch
```

### What Happens
1. Bumps version in package.json
2. Runs deployment gates
3. Deploys to Cloudflare

### Files Modified
- `xtask/src/cli.rs` - Added --bump parameter
- `xtask/src/deploy/mod.rs` - Added bump_version() function
- `xtask/src/main.rs` - Pass bump parameter

## Problems to Fix

### 1. Command is Too Long
```bash
# Current (BAD)
cargo xtask deploy --app worker --bump patch

# Should be (GOOD)
cargo xtask deploy worker
# or
cargo xtask worker
```

### 2. Tier System is Confusing
- User doesn't care about "frontend" tier
- User wants to deploy ONE app at a time
- Remove tier system or make it invisible

### 3. App Names are Inconsistent
- Worker catalog has 3 aliases: worker, gwc, worker-catalog
- Just pick ONE name and use it everywhere

### 4. Release vs Deploy Split
- User doesn't understand why these are separate
- Consider merging or renaming

## What Next Team Should Do

### Priority 1: Simplify Deploy Command
Make it dead simple:
```bash
cargo xtask deploy gwc
```

This should:
1. Auto-detect current version
2. Auto-bump patch version
3. Deploy

No --app, no --bump, just works.

### Priority 2: Remove Tier Confusion
- Delete tier system for frontend apps
- Each app versions independently
- User never sees "frontend" tier

### Priority 3: Better Naming
Pick ONE name per app:
- gwc (not worker, not worker-catalog)
- commercial
- marketplace
- docs

### Priority 4: Better Help
```bash
cargo xtask deploy --help

Should show:
  cargo xtask deploy gwc          Deploy worker catalog
  cargo xtask deploy commercial   Deploy commercial site
  cargo xtask deploy marketplace  Deploy marketplace
  cargo xtask deploy docs         Deploy documentation
```

## Code Locations

### Deploy Logic
- `xtask/src/deploy/mod.rs` - Main deploy logic
- `xtask/src/deploy/worker_catalog.rs` - GWC deployment
- `xtask/src/deploy/gates.rs` - Deployment gates

### CLI
- `xtask/src/cli.rs` - Command definitions
- `xtask/src/main.rs` - Command routing

### Config
- `.version-tiers/frontend.toml` - Frontend tier config (REMOVE THIS)
- `bin/80-hono-worker-catalog/package.json` - GWC version
- `bin/80-hono-worker-catalog/wrangler.jsonc` - GWC Cloudflare config

## User's Workflow (What They Want)

1. Make changes to worker catalog
2. Run: `cargo xtask deploy gwc`
3. Done

That's it. No tiers, no --bump, no confusion.

## Important Notes

- User HATES documentation commits
- User wants CODE that WORKS
- User wants SIMPLE commands
- User doesn't care about "best practices" if they're confusing
- User wants to deploy ONE app at a time
- User didn't ask for tier system

## What NOT to Do

1. ❌ Don't commit documentation without code changes
2. ❌ Don't explain why current system is "correct"
3. ❌ Don't tell user to "use it differently"
4. ❌ Don't create more complexity
5. ❌ Don't bundle apps together without asking

## What TO Do

1. ✅ Simplify commands
2. ✅ Make it obvious
3. ✅ Fix the code, not the docs
4. ✅ Test before committing
5. ✅ Listen to user frustration

## Summary

User wants to deploy worker catalog. Current system is too complex. Simplify it.

**End goal:**
```bash
cargo xtask deploy gwc
```

That's all they should need to type.
