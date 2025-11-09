# TEAM-463: Production Deployment Support

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE

## Problem

The `cargo xtask deploy` command only deployed to **preview** environments (branch=main), not production. This meant all deployments went to preview URLs like `https://main.rbee-commercial.pages.dev` instead of production URLs.

## Solution

Added `--production` flag to the deploy command to support production deployments.

### Changes Made

1. **CLI Flag** (`xtask/src/cli.rs`)
   - Added `production: bool` flag to `Deploy` command
   - Default: `false` (preview deployment)

2. **Deploy Module** (`xtask/src/deploy/mod.rs`)
   - Updated `run()` function signature to accept `production: bool`
   - Passes production flag to all deployment functions

3. **Deployment Functions** (Updated all 4 Cloudflare apps)
   - `commercial.rs` - Commercial site
   - `marketplace.rs` - Marketplace
   - `docs.rs` - User docs
   - `worker_catalog.rs` - Worker catalog
   
   Each now:
   - Accepts `production: bool` parameter
   - Deploys to `--branch=production` when `production=true`
   - Deploys to `--branch=main` when `production=false` (preview)
   - Shows clear environment indicator in output

4. **Release Integration** (`xtask/src/release/cli.rs`)
   - Release command now passes `production=true` when deploying
   - Ensures releases go to production, not preview

## Usage

### Preview Deployment (Default)
```bash
# Deploy to preview (branch=main)
cargo xtask deploy --app commercial --bump patch

# Preview URL: https://main.rbee-commercial.pages.dev
```

### Production Deployment
```bash
# Deploy to production (branch=production)
cargo xtask deploy --app commercial --bump patch --production

# Production URL: https://rbee-commercial.pages.dev
# Custom domain: https://rbee.dev
```

### Release Command (Auto-Production)
```bash
# Release command automatically deploys to production
cargo xtask release --app commercial --type patch

# After version bump, deploys to production automatically
```

## Deployment Targets

| App | Preview URL | Production URL | Custom Domain |
|-----|-------------|----------------|---------------|
| commercial | `main.rbee-commercial.pages.dev` | `rbee-commercial.pages.dev` | `rbee.dev` |
| marketplace | `main.rbee-marketplace.pages.dev` | `rbee-marketplace.pages.dev` | `marketplace.rbee.dev` |
| docs | `main.rbee-user-docs.pages.dev` | `rbee-user-docs.pages.dev` | `docs.rbee.dev` |
| worker | `main.rbee-gwc.pages.dev` | `rbee-gwc.pages.dev` | `gwc.rbee.dev` |

## Verification

✅ **Build successful** - `cargo check --bin xtask` passes  
✅ **All 4 Cloudflare apps updated**  
✅ **Release command integration complete**  
✅ **Clear environment indicators in output**

## Code Reduction (RULE ZERO)

**Created abstraction to eliminate duplicate logic:**

- **Before:** 3 files × ~100 lines = ~300 lines of duplicate deployment code
- **After:** 1 shared module (120 lines) + 3 config files (20 lines each) = ~180 lines
- **Reduction:** ~120 lines removed (40% less code)

### Abstraction: `nextjs_ssg.rs`

Created shared deployment logic for all Next.js SSG apps:
- Single source of truth for build/deploy process
- Configurable via `NextJsDeployConfig` struct
- Handles: env file creation, build, deploy, output formatting

### Refactored Files

All 3 Next.js apps now use the abstraction:
1. `commercial.rs` - 27 lines (was ~100)
2. `marketplace.rs` - 30 lines (was ~108)
3. `docs.rs` - 23 lines (was ~81)

Each file now just defines config and calls `deploy_nextjs_ssg()`.

## Files Changed

1. `xtask/src/cli.rs` - Added `--production` flag
2. `xtask/src/main.rs` - Pass production flag to deploy::run
3. `xtask/src/deploy/mod.rs` - Accept and pass production flag, added nextjs_ssg module
4. **`xtask/src/deploy/nextjs_ssg.rs`** - NEW: Shared Next.js SSG deployment logic
5. `xtask/src/deploy/commercial.rs` - Refactored to use abstraction (100→27 lines)
6. `xtask/src/deploy/marketplace.rs` - Refactored to use abstraction (108→30 lines)
7. `xtask/src/deploy/docs.rs` - Refactored to use abstraction (81→23 lines)
8. `xtask/src/deploy/worker_catalog.rs` - Use production branch
9. `xtask/src/release/cli.rs` - Deploy to production after release

---

**RULE ZERO APPLIED:** ✅ Created abstraction to eliminate duplicate code  
**ENTROPY ELIMINATED:** ✅ Single source of truth for Next.js deployments
