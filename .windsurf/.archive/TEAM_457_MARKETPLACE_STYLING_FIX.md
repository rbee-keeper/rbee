# TEAM-457: Marketplace Styling Fix

**Status:** 🔧 REQUIRES RESTART

## Problem

Marketplace app (http://localhost:7823) has no styling - completely unstyled HTML.

## Root Cause

After running `pnpm install` to add `@rbee/env-config`, Next.js dev server's CSS cache is stale. The browser is requesting:
```
http://localhost:7823/_next/static/css/app/layout.css
```

But this file doesn't exist (returns "Not Found"). The actual CSS files exist in `.next/static/css/` but with different hashes.

## Solution

**Clear Next.js cache and restart the marketplace dev server:**

```bash
# Option 1: Clear cache and restart (recommended)
cd frontend/apps/marketplace
rm -rf .next
pnpm dev

# Option 2: If using turbo dev, restart the entire turbo process
# Ctrl+C to stop turbo dev
pnpm turbo dev --concurrency 30
```

## Why This Happened

1. `pnpm install` added new workspace packages (`@rbee/env-config`)
2. Next.js dev server was already running during the install
3. Next.js cached the old CSS bundle references
4. Browser requests old CSS file that no longer exists
5. Result: No styles loaded

## Verification

After restart, check:
1. ✅ Navigate to http://localhost:7823
2. ✅ Page should have full styling (dark theme, proper fonts, layout)
3. ✅ Navigation should be styled
4. ✅ Check browser DevTools > Network > CSS files should load successfully

## Files Affected

- `.next/` cache directory (needs to be cleared)
- No code changes required

---

**TEAM-457 signature:** Next.js cache issue after workspace package changes
