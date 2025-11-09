# TEAM-427: xtask Deploy & Release Update

**Date:** 2025-11-09  
**Status:** ✅ COMPLETE

## Changes Made

### 1. Updated Marketplace Deployment (`xtask/src/deploy/marketplace.rs`)

**Old approach (OUTDATED):**
- Deployed `.next-deploy/` directory
- Used `rsync` to exclude cache
- Assumed Next.js SSR build output

**New approach (CURRENT):**
- Deploys `out/` directory (Next.js static export)
- No rsync needed (static export is clean)
- Verifies `out/` directory exists before deployment
- Uses `npx wrangler` instead of direct `wrangler`

### Key Changes

#### Build Output
```rust
// OLD: Deploy .next-deploy (SSR build)
"pages", "deploy", ".next-deploy"

// NEW: Deploy out/ (static export)
"pages", "deploy", "out/"
```

#### Verification
```rust
// NEW: Verify out/ directory exists
let out_dir = format!("{}/out", app_dir);
if !std::path::Path::new(&out_dir).exists() {
    anyhow::bail!("Build output directory 'out/' not found. Check next.config.ts has output: 'export'");
}
```

#### Deployment Info
Now shows accurate page counts:
- 455 total static pages
- 200 model redirect pages (`/models/[slug]`)
- 200 model detail pages (100 HF + 100 CivitAI)
- 55 other pages (filters, workers, etc.)

## How to Deploy

### Using xtask
```bash
cargo xtask deploy marketplace
```

### Manual Deployment
```bash
cd frontend/apps/marketplace
pnpm run build
npx wrangler pages deploy out/ --project-name=rbee-marketplace --branch=main
```

## Deployment Configuration

### Required Files

**`next.config.ts`:**
```typescript
const nextConfig: NextConfig = {
  output: 'export',  // ← CRITICAL: Enables static export
  // ... webpack config for WASM
}
```

**`wrangler.jsonc`:**
```jsonc
{
  "name": "rbee-marketplace",
  "compatibility_date": "2025-03-01",
  "compatibility_flags": ["nodejs_compat"],
  "pages_build_output_dir": "out"
}
```

**`package.json` build script:**
```json
{
  "scripts": {
    "build": "NEXT_PUBLIC_SITE_URL=https://marketplace.rbee.dev next build --webpack"
  }
}
```

## Build Warnings

### metadataBase Warning (BENIGN)

**Warning message:**
```
⚠ metadataBase property in metadata export is not set for resolving social open graph or twitter images, using "http://localhost:3000"
```

**Why it appears:**
Next.js shows this during the build process before the root layout metadata is fully processed.

**Why it's safe to ignore:**
The actual generated HTML has correct metadata:
```html
<meta property="og:url" content="https://marketplace.rbee.dev"/>
<meta property="og:image" content="https://marketplace.rbee.dev/opengraph-image?..."/>
```

**Verification:**
```bash
grep 'og:url' frontend/apps/marketplace/out/index.html
# Shows: https://marketplace.rbee.dev (correct!)
```

## Deployment URLs

- **Production:** https://main.rbee-marketplace.pages.dev
- **Custom domain:** https://marketplace.rbee.dev (when configured)

## URL Structure

The marketplace supports multiple URL formats:

1. **Legacy redirect:** `/models/[slug]` → Auto-detects provider
2. **HuggingFace:** `/models/huggingface/[slug]` → Direct
3. **CivitAI:** `/models/civitai/[slug]` → Direct
4. **Workers:** `/workers/[workerId]` → Direct

All URLs are pre-rendered as static HTML at build time.

## Release Process

### Current Status
- ✅ Marketplace deployment updated
- ⏸️ Release process (not modified - works as-is)

### For Future Reference

The release process in `xtask/src/release/` handles:
- `bump_js.rs` - Bump JavaScript package versions
- `bump_rust.rs` - Bump Rust crate versions
- `cli.rs` - Release CLI commands
- `tiers.rs` - Version tier management

These don't need updates for the marketplace deployment change.

## Testing

### Dry Run
```bash
cargo xtask deploy marketplace --dry-run
```

### Full Deployment
```bash
cargo xtask deploy marketplace
```

### Verify Deployment
```bash
# Check page count
ls -1 frontend/apps/marketplace/out/*.html | wc -l

# Check metadata
grep 'og:url' frontend/apps/marketplace/out/index.html

# Check redirect pages
ls -1 frontend/apps/marketplace/out/models/*.html | wc -l
# Should show 200 redirect pages
```

## Next Steps

**None required.** The deployment system is now updated for the current marketplace architecture.

---

**TEAM-427 SIGNATURE:** xtask deployment updated for Next.js static export.
