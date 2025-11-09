# TEAM-427: User-Docs Cloudflare Pages Deployment

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE  
**Deployment URL:** https://main.rbee-user-docs.pages.dev

## Summary

Applied all marketplace learnings to user-docs app and successfully deployed to Cloudflare Pages as a static site.

## Changes Made

### 1. next.config.ts

**Added static export:**
```typescript
const nextConfig: NextConfig = {
  // TEAM-427: Static export for Cloudflare Pages
  output: 'export',
  // ... rest of config
}
```

**Fixed top-level await issue:**
```typescript
// OLD (caused build error)
const { initOpenNextCloudflareForDev } = await import('@opennextjs/cloudflare')

// NEW (works with static export)
if (process.env.NODE_ENV === 'development') {
  import('@opennextjs/cloudflare').then(({ initOpenNextCloudflareForDev }) => {
    initOpenNextCloudflareForDev()
  })
}
```

### 2. wrangler.jsonc

**Before (Workers config):**
```jsonc
{
  "name": "user-docs",
  "main": ".open-next/worker.js",
  "assets": {
    "binding": "ASSETS",
    "directory": ".open-next/assets"
  },
  "env": {
    "development": { ... } // Not supported by Pages
  }
}
```

**After (Pages config):**
```jsonc
{
  "name": "rbee-user-docs",
  "compatibility_date": "2025-03-01",
  "compatibility_flags": ["nodejs_compat"],
  "pages_build_output_dir": "out",
  "env": {
    "preview": { ... } // Only preview and production supported
  }
}
```

### 3. package.json

**Updated build script:**
```json
{
  "build": "NEXT_PUBLIC_SITE_URL=https://docs.rbee.dev next build --webpack",
  "deploy": "pnpm run build && npx wrangler pages deploy out/ --project-name=rbee-user-docs --branch=main",
  "preview": "pnpm run build && npx wrangler pages deploy out/ --project-name=rbee-user-docs --branch=preview"
}
```

### 4. app/layout.tsx

**Added metadataBase:**
```typescript
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || 'https://docs.rbee.dev'

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  // ... rest of metadata
}
```

## Build Results

```
Route (app)
┌ ○ /
├ ○ /_not-found
├ ○ /docs
├ ○ /docs/getting-started/installation
├ ○ /docs/architecture/overview
└ ... (35 total pages)

○  (Static)  prerendered as static content
```

**Total:** 35 static pages  
**Build time:** ~62 seconds  
**Output:** `out/` directory

## Deployment Results

```
✨ Success! Uploaded 354 files (7.67 sec)
✨ Deployment complete!
🌐 Production URL: https://main.rbee-user-docs.pages.dev
🌐 Latest deployment: https://8db5cfc4.rbee-user-docs.pages.dev
```

## Lessons Applied from Marketplace

### ✅ Static Export
- Use `output: 'export'` in next.config.ts
- Generates `out/` directory with static HTML
- No server-side rendering needed

### ✅ Cloudflare Pages (not Workers)
- Use `pages_build_output_dir` instead of `main`
- Remove `assets.binding` (reserved name in Pages)
- Only `preview` and `production` envs supported

### ✅ Build Script
- Set `NEXT_PUBLIC_SITE_URL` in build command
- Use `npx wrangler pages deploy out/`
- Specify `--project-name` and `--branch`

### ✅ Metadata
- Always set `metadataBase` to avoid warnings
- Use environment variable with fallback
- Prevents localhost:3000 warnings

### ✅ Top-Level Await
- Cannot use `await` at top level in config files
- Use `.then()` for dynamic imports
- Only initialize dev tools in development mode

## Comparison: Before vs After

| Aspect | Before (Workers) | After (Pages) |
|--------|-----------------|---------------|
| **Deployment** | OpenNext + Workers | Static export + Pages |
| **Build output** | `.open-next/` | `out/` |
| **Config** | `main`, `assets.binding` | `pages_build_output_dir` |
| **Rendering** | SSR (server-side) | Static HTML |
| **Deploy command** | `opennextjs-cloudflare deploy` | `wrangler pages deploy out/` |
| **Pages** | 35 | 35 |
| **Files** | ~354 | 354 |
| **Complexity** | High (Workers runtime) | Low (static files) |

## Benefits

### Performance
- ✅ No cold starts (static files)
- ✅ Global CDN caching
- ✅ Instant page loads

### Cost
- ✅ No CPU time charges (just bandwidth)
- ✅ No memory limits
- ✅ Free tier is generous

### Simplicity
- ✅ No Workers runtime needed
- ✅ No OpenNext build step
- ✅ Standard Next.js static export

### Reliability
- ✅ No server failures
- ✅ No runtime errors
- ✅ Just static HTML/CSS/JS

## Deployment Commands

### Build
```bash
cd frontend/apps/user-docs
pnpm run build
```

### Deploy to Production
```bash
pnpm run deploy
# or
npx wrangler pages deploy out/ --project-name=rbee-user-docs --branch=main
```

### Deploy to Preview
```bash
pnpm run preview
# or
npx wrangler pages deploy out/ --project-name=rbee-user-docs --branch=preview
```

## URLs

- **Production:** https://main.rbee-user-docs.pages.dev
- **Custom domain:** https://docs.rbee.dev (when configured)
- **Latest deployment:** https://8db5cfc4.rbee-user-docs.pages.dev

## Issues Encountered & Fixed

### 1. Top-Level Await Error
**Error:** `require() cannot be used on an ESM graph with top-level await`  
**Fix:** Changed `await import()` to `import().then()`

### 2. Development Environment Error
**Error:** `"development" environment not supported by Pages`  
**Fix:** Removed `env.development` from wrangler.jsonc

### 3. SSL Error on First Access
**Error:** `ERR_SSL_VERSION_OR_CIPHER_MISMATCH`  
**Cause:** DNS/SSL propagation delay after deployment  
**Fix:** Wait a few minutes for SSL to provision

## Next Steps

**None required.** User-docs is now deployed to Cloudflare Pages with the same pattern as marketplace.

## xtask Integration

The xtask deployment system can be updated similarly:

```rust
// xtask/src/deploy/user_docs.rs
pub fn deploy(dry_run: bool) -> Result<()> {
    println!("🚀 Deploying User Docs to Cloudflare Pages");
    
    let app_dir = "frontend/apps/user-docs";
    
    // Build
    Command::new("pnpm")
        .args(&["run", "build"])
        .current_dir(app_dir)
        .status()?;
    
    // Deploy
    Command::new("npx")
        .args(&[
            "wrangler", "pages", "deploy", "out/",
            "--project-name=rbee-user-docs",
            "--branch=main",
        ])
        .current_dir(app_dir)
        .status()?;
    
    Ok(())
}
```

---

**TEAM-427 SIGNATURE:** User-docs successfully deployed to Cloudflare Pages using marketplace learnings.
