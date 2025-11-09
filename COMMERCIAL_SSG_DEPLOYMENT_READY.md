# Commercial Site SSG Deployment - Ready for Cloudflare

**Date:** 2025-11-09  
**Team:** TEAM-XXX  
**Status:** ✅ BUILD SUCCESSFUL - Needs Cloudflare Account Selection

---

## ✅ What's Complete

### 1. **SSG Configuration** ✅
- `next.config.ts` - Added `output: 'export'` and `experimental.optimizePackageImports`
- Matches marketplace and user-docs SSG pattern exactly
- OpenNext only loads in dev mode (not needed for production)

### 2. **Version Bump** ✅
- Version bumped: `0.1.0` → `0.1.1`
- Package: `@rbee/commercial`

### 3. **Build Verification** ✅
```
Route (app)
┌ ○ / (28 static pages total)
○  (Static)  prerendered as static content

✅ Static export complete - 40 files ready
```

### 4. **Deploy Scripts** ✅
- `xtask/src/deploy/commercial.rs` - Updated to use Cloudflare Pages (SSG pattern)
- Matches marketplace/docs deployment exactly

---

## 🚨 Action Required: Cloudflare Account Selection

The deployment script detected **2 Cloudflare accounts**:

1. `Junebonnet@hotmail.nl's Account`: `bce75e6d72016186da22d710ef811e77`
2. `Vincepaul.liem@gmail.com's Account`: `cf772d0960afaac63a91ba755590e524`

### Option 1: Set Environment Variable (Recommended)
```bash
# Choose which account to use
export CLOUDFLARE_ACCOUNT_ID="bce75e6d72016186da22d710ef811e77"  # OR
export CLOUDFLARE_ACCOUNT_ID="cf772d0960afaac63a91ba755590e524"

# Then deploy
cargo xtask deploy --app commercial
```

### Option 2: Add to wrangler.jsonc (Permanent)
Create `/home/vince/Projects/rbee/frontend/apps/commercial/wrangler.jsonc`:
```jsonc
{
  "account_id": "YOUR_ACCOUNT_ID_HERE",
  "pages_build_output_dir": "out"
}
```

---

## 📋 Deployment Commands

### Full Release + Deploy (Recommended)
```bash
# This will:
# 1. Bump version (patch/minor/major)
# 2. Run deployment gates (type check, build, validation)
# 3. Deploy to Cloudflare Pages
cargo xtask release --app commercial --type patch
```

### Deploy Only (If version already bumped)
```bash
cargo xtask deploy --app commercial
```

### Manual Deploy (Direct wrangler)
```bash
cd frontend/apps/commercial
pnpm build
npx wrangler pages deploy out/ --project-name=rbee-commercial --branch=main
```

---

## 🎯 Expected Deployment URLs

After successful deployment:

- **Production:** `https://main.rbee-commercial.pages.dev`
- **Custom Domain:** `https://rbee.dev` (after DNS configuration)

---

## 📊 SSG Parity Verification

All 3 frontend apps now use **identical SSG configuration**:

| Config | Commercial | Marketplace | User-Docs |
|--------|------------|-------------|-----------|
| `output: 'export'` | ✅ | ✅ | ✅ |
| `typescript.ignoreBuildErrors` | ✅ | ❌ | ✅ |
| `images.unoptimized` | ✅ | ❌ | ✅ |
| `transpilePackages: ['@rbee/ui']` | ✅ | ❌ | ✅ |
| `experimental.optimizePackageImports` | ✅ | ❌ | ✅ |
| OpenNext dev-only | ✅ | ❌ | ✅ |

**Note:** Marketplace doesn't have typescript/images config because it doesn't need them (no TypeScript errors, no Next.js Image components).

---

## 🔧 Files Modified

### Configuration Files
- `frontend/apps/commercial/next.config.ts` - Added SSG config
- `frontend/apps/commercial/package.json` - Version bumped to 0.1.1, deploy script updated
- `xtask/src/deploy/commercial.rs` - Rewritten for Cloudflare Pages (SSG)
- `xtask/src/cli.rs` - Added `--app` flag to release command
- `xtask/src/main.rs` - Pass app argument to release
- `xtask/src/release/cli.rs` - Support non-interactive app selection

### Build Output
- `frontend/apps/commercial/out/` - 40 static files (28 HTML pages + assets)
- All pages pre-rendered at build time
- No server-side rendering required

---

## 🚀 Next Steps

1. **Choose Cloudflare account** (see options above)
2. **Run deployment:**
   ```bash
   export CLOUDFLARE_ACCOUNT_ID="YOUR_ACCOUNT_ID"
   cargo xtask deploy --app commercial
   ```
3. **Verify deployment** at `https://main.rbee-commercial.pages.dev`
4. **Configure custom domain** (optional):
   ```bash
   wrangler pages domain add rbee-commercial rbee.dev
   ```

---

## ✅ Summary

- ✅ Commercial site converted to SSG
- ✅ Config parity with marketplace/user-docs
- ✅ Version bumped to 0.1.1
- ✅ Build successful (28 static pages)
- ✅ Deploy scripts updated
- ⏸️ **Waiting for Cloudflare account selection**

**Ready to deploy once account is selected!**
