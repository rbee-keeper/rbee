# TEAM-457: Dev Server Restart Required

## Issue Found

The `.env.local` file had **production URLs** instead of development URLs!

### What Was Wrong

```bash
# These were set to production:
NEXT_PUBLIC_MARKETPLACE_URL=https://marketplace.rbee.dev
NEXT_PUBLIC_SITE_URL=https://rbee.dev

# Development URLs were commented out:
# NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:3001
# NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

### What Was Fixed

Uncommented the development URLs in `.env.local`:

```bash
NEXT_PUBLIC_MARKETPLACE_URL=http://localhost:3001
NEXT_PUBLIC_SITE_URL=http://localhost:3000
```

## Action Required

**You MUST restart the dev server for changes to take effect:**

```bash
# Stop current dev server (Ctrl+C)
cd frontend/apps/commercial
pnpm dev
```

## Why This Happened

The `.env.local` file was created with production values by default. For local development, you need to uncomment the development overrides.

## Verification

After restarting:

1. Visit `http://localhost:7822/`
2. Click "Marketplace" in navigation
3. Click "LLM Models"
4. Should go to `http://localhost:3001/models` (NOT `https://marketplace.rbee.dev/models`)

Or visit the debug page:
```
http://localhost:7822/debug-env
```

Should show:
- `env.marketplaceUrl`: `http://localhost:3001`
- `env.siteUrl`: `http://localhost:3000`

## UX Issue: Click Closes Dropdown

You mentioned clicking the Marketplace button closes the dropdown. This is a separate issue from the URL bug.

**Current behavior:**
- Hover → Opens dropdown
- Click → Toggles dropdown (closes if open)

**Expected behavior:**
- Hover → Opens dropdown  
- Click → Keeps dropdown open (doesn't toggle)

This requires a Navigation component fix (separate from env vars).
