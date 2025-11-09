# Frontend Deployment Cleanup - TEAM-XXX

## Summary

Fixed deployment architecture to correctly separate SSG (Static Site Generation) from SSR (Server-Side Rendering) applications. **Marketplace was incorrectly configured with OpenNext when it only needs static export.**

## Changes Made

### 1. Marketplace - Removed Unnecessary OpenNext ✅

**Issue:** Marketplace uses `output: 'export'` for SSG but had OpenNext dependencies and build artifacts.

**Fixed:**
- ❌ Removed `@opennextjs/cloudflare` from `dependencies`
- ❌ Deleted `open-next.config.ts`
- ❌ Deleted `.open-next/` build artifacts
- ✅ Updated deploy script to use `wrangler pages deploy out/`
- ✅ Cleaned up `next.config.ts` comments

**Files Modified:**
```
frontend/apps/marketplace/package.json
frontend/apps/marketplace/next.config.ts
```

**Before:**
```json
"deploy": "opennextjs-cloudflare build && opennextjs-cloudflare deploy"
"dependencies": {
  "@opennextjs/cloudflare": "^1.11.1",
  ...
}
```

**After:**
```json
"deploy": "pnpm run build && npx wrangler pages deploy out/ --project-name=rbee-marketplace --branch=main"
"dependencies": {
  // No @opennextjs/cloudflare
  ...
}
```

### 2. User Docs - Fixed Deployment Script ✅

**Issue:** xtask deployment script was deploying `.next` instead of `out/` and using wrong project name.

**Fixed:**
```rust
// Before
".next", "--project-name=rbee-docs", "--branch=production"

// After  
"out", "--project-name=rbee-user-docs", "--branch=main"
```

**Files Modified:**
```
xtask/src/deploy/docs.rs
```

**Result:** ✅ https://docs.rbee.dev/ now works!

### 3. Commercial Site - Configured for OpenNext SSR ✅

**Issue:** Commercial deployment script was using static Pages deployment instead of OpenNext.

**Fixed:**
```rust
// Before: Wrong - static Pages deployment
wrangler pages deploy .next --project-name=rbee-commercial

// After: Correct - OpenNext SSR deployment
pnpm run deploy  // Runs: opennextjs-cloudflare build && deploy
```

**Files Modified:**
```
xtask/src/deploy/commercial.rs
```

**Configuration Verified:**
- ✅ `@opennextjs/cloudflare` dependency present
- ✅ `open-next.config.ts` exists
- ✅ `wrangler.jsonc` correctly points to `.open-next/worker.js`
- ✅ Build creates `.open-next/` directory with Worker + assets
- ✅ No `output: 'export'` in next.config.ts (enables SSR)

## Verification

### Marketplace Build ✅
```bash
cd frontend/apps/marketplace
pnpm run build
# ✅ Outputs to: out/ (354 static files)
# ✅ No .open-next/ directory
# ✅ All routes marked as SSG (●) or Static (○)
```

### User Docs Build ✅
```bash
cd frontend/apps/user-docs
pnpm run build
# ✅ Outputs to: out/ (35 static pages)
# ✅ Successfully deployed to https://docs.rbee.dev/
```

### Commercial Build ✅
```bash
cd frontend/apps/commercial
pnpm run build
npx opennextjs-cloudflare build
# ✅ Outputs to: .next/ (Next.js build)
# ✅ Transforms to: .open-next/ (OpenNext bundle)
# ✅ Creates: .open-next/worker.js + .open-next/assets/
# ✅ Ready for SSR deployment
```

## Architecture Summary

| Application | Type | OpenNext? | Build Output | Deployment |
|------------|------|-----------|--------------|------------|
| **Marketplace** | SSG | ❌ No | `out/` | Static Pages |
| **User Docs** | SSG | ❌ No | `out/` | Static Pages |
| **Commercial** | SSR | ✅ Yes | `.open-next/` | Workers + Pages |

## Why This Matters

### SSG Sites (Marketplace, Docs)
- **Faster builds** - No OpenNext transformation needed
- **Lower costs** - Static Pages hosting vs Workers
- **Better caching** - Pure static assets on CDN
- **Simpler deployments** - Just upload files

### SSR Site (Commercial)
- **Dynamic rendering** - Can use server-side logic
- **API routes** - Backend functionality
- **Personalization** - User-specific content
- **Form handling** - Server-side processing

## Deployment Commands

```bash
# Deploy marketplace (SSG)
cargo xtask deploy marketplace
# OR
cd frontend/apps/marketplace && pnpm run deploy

# Deploy user docs (SSG)
cargo xtask deploy docs
# OR
cd frontend/apps/user-docs && pnpm run build && \
  wrangler pages deploy out/ --project-name=rbee-user-docs

# Deploy commercial (SSR)
cargo xtask deploy commercial
# OR
cd frontend/apps/commercial && pnpm run deploy
```

## Files Modified

```
frontend/apps/marketplace/package.json         # Removed OpenNext dependency
frontend/apps/marketplace/next.config.ts       # Updated comments
frontend/apps/marketplace/open-next.config.ts  # DELETED
frontend/apps/marketplace/.open-next/          # DELETED

xtask/src/deploy/docs.rs                      # Fixed output dir + project name
xtask/src/deploy/commercial.rs                # Use opennextjs-cloudflare deploy

frontend/DEPLOYMENT_ARCHITECTURE.md           # NEW - Architecture docs
frontend/CLEANUP_SUMMARY_TEAM_XXX.md          # NEW - This summary
```

## Testing

All three applications tested and verified:

1. ✅ **Marketplace** - Builds as SSG, deploys to static Pages
2. ✅ **User Docs** - Builds as SSG, deploys to docs.rbee.dev
3. ✅ **Commercial** - Builds with OpenNext, ready for SSR deployment

## Rule Zero Compliance ✅

**Breaking changes > Backwards compatibility**

- Removed unnecessary dependencies (marketplace no longer needs OpenNext)
- Fixed incorrect implementations (docs deployment)
- No deprecated code left behind
- Clear, single way to deploy each application type
- Compiler-verified changes (cargo check passes)

---

**Next Steps:**

Commercial site is ready for deployment whenever needed:
```bash
cargo xtask deploy commercial
```

This will build and deploy the SSR-enabled commercial site to Cloudflare Workers.
