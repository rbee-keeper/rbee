# TEAM-457: env-config Package Fix

**Status:** ✅ FIXED (Restart Required)

## Problem

Build error on commercial frontend: `Module not found: Can't resolve '@rbee/env-config'`

## Root Causes

1. **Missing from workspace:** `frontend/packages/env-config` not listed in `pnpm-workspace.yaml`
2. **Missing dependency:** `@rbee/env-config` not declared in commercial app's `package.json`
3. **Wrong package names:** env-config used `@rbee/eslint-config` and `@rbee/typescript-config` instead of `@repo/` prefix

## Fixes Applied

### 1. Added to pnpm-workspace.yaml
```yaml
packages:
  # ... other packages ...
  - frontend/packages/env-config  # ← ADDED
  # ... other packages ...
```

### 2. Added to commercial app dependencies
```json
{
  "dependencies": {
    "@rbee/env-config": "workspace:*",  // ← ADDED
    // ... other dependencies ...
  }
}
```

### 3. Fixed env-config devDependencies
```json
{
  "devDependencies": {
    "@types/node": "^22.10.1",
    "@repo/eslint-config": "workspace:*",     // ← Fixed from @rbee/
    "@repo/typescript-config": "workspace:*",  // ← Fixed from @rbee/
    "typescript": "^5.7.2"
  }
}
```

### 4. Ran pnpm install
```bash
pnpm install  # ✅ SUCCESS - packages linked
```

## Next Step: Restart Dev Server

The dev server is still showing the error because it was running during the package installation. **You need to restart the turbo dev process.**

### Restart Command:
1. Stop current turbo dev (Ctrl+C in the terminal running `turbo dev`)
2. Restart: `pnpm turbo dev --concurrency 30`

## Files Modified

- `/home/vince/Projects/llama-orch/pnpm-workspace.yaml` (+1 line)
- `/home/vince/Projects/llama-orch/frontend/apps/commercial/package.json` (+1 dependency)
- `/home/vince/Projects/llama-orch/frontend/packages/env-config/package.json` (fixed 2 package names)

## Verification

After restart, the commercial frontend should:
- ✅ Build successfully
- ✅ Import `@rbee/env-config` without errors
- ✅ Navigation to Marketplace > Models should work

## What env-config Provides

- Automatic dev/prod URL detection based on `NODE_ENV`
- Consistent port configuration from `@rbee/shared-config`
- URL helpers: `urls.marketplace.llmModels`, etc.
- Environment overrides via `NEXT_PUBLIC_*` env vars
- CORS origins configuration

---

**TEAM-457 signature:** Package configuration fixed, dev server restart required
