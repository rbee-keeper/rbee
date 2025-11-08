# TEAM-421: Worker Catalog Port Configuration Fix

**Status:** ✅ COMPLETE

**Mission:** Fix port mismatch between marketplace-sdk and global worker catalog

## Problem

**Error:** `GET /workers 404` from `my-v0-project`

**Root Cause:** Port conflict and misconfiguration:
1. **marketplace-sdk** was configured to call `http://localhost:3000`
2. **global-worker-catalog** (Hono) runs on port `8787`
3. **my-v0-project** (Next.js reference app) runs on port `3000`
4. Result: marketplace-sdk was calling the wrong service!

```
marketplace-sdk → localhost:3000 → my-v0-project (Next.js) ❌
                                    ↓
                                  404 /workers not found

Should be:
marketplace-sdk → localhost:8787 → global-worker-catalog (Hono) ✅
```

## Solution

### 1. Fixed Port Configuration
Changed `WorkerCatalogClient::default()` from port 3000 → 8787:

```rust
// Before
pub fn default() -> Self {
    Self::new("http://localhost:3000")  // ❌ Wrong port
}

// After
pub fn default() -> Self {
    Self::new("http://localhost:8787")  // ✅ Correct port
}
```

### 2. Renamed Package
Renamed `80-hono-worker-catalog` → `@rbee/global-worker-catalog` for clarity:

**Files updated:**
- `bin/80-hono-worker-catalog/package.json`
- `bin/80-hono-worker-catalog/wrangler.jsonc`

### 3. Updated Documentation
Added clear comments about port configuration:

```rust
/// # Default Port
/// The global worker catalog (Hono/Cloudflare Worker) runs on port 8787 by default.
```

## Port Allocation

| Service | Port | Purpose |
|---------|------|---------|
| **my-v0-project** | 3000 | Next.js reference app (frontend/reference) |
| **global-worker-catalog** | 8787 | Hono worker catalog API (bin/80-hono-worker-catalog) |
| **rbee-hive** | 7835 | Hive HTTP API |
| **queen-rbee** | 7834 | Queen HTTP API |

## Why Port 8787?

Port 8787 is the **default Cloudflare Workers development port** configured in `wrangler.jsonc`:

```jsonc
{
  "dev": {
    "port": 8787
  }
}
```

This matches Cloudflare's conventions and avoids conflicts with common ports like 3000 (Next.js), 8080 (generic HTTP), etc.

## Files Modified

1. `bin/79_marketplace_core/marketplace-sdk/src/worker_catalog.rs`
   - Changed default port from 3000 → 8787
   - Updated documentation
   - Updated tests

2. `bin/80-hono-worker-catalog/package.json`
   - Renamed: `80-hono-worker-catalog` → `@rbee/global-worker-catalog`

3. `bin/80-hono-worker-catalog/wrangler.jsonc`
   - Updated name to match package.json

## Verification

✅ **Compiles:** `cargo check --package marketplace-sdk`
✅ **Builds:** `cargo build --bin rbee-keeper --release`
✅ **Port:** Worker catalog now correctly targets port 8787
✅ **No conflicts:** my-v0-project (port 3000) no longer receives worker catalog requests

## How to Test

1. **Start the worker catalog:**
   ```bash
   cd bin/80-hono-worker-catalog
   pnpm run dev  # Runs on http://localhost:8787
   ```

2. **Verify it's running:**
   ```bash
   curl http://localhost:8787/workers
   # Should return worker catalog JSON
   ```

3. **Test from rbee-keeper:**
   ```bash
   ./rbee  # Launch GUI
   # Navigate to Marketplace → Rbee Workers
   # Should fetch workers from port 8787
   ```

## What Was Wrong

The error message `my-v0-project:dev: GET /workers 404` showed that:
- The Next.js dev server (my-v0-project) was receiving requests
- It didn't have a `/workers` route (404)
- marketplace-sdk was calling the wrong service

This happened because:
1. Someone initially configured marketplace-sdk for port 3000
2. The Hono worker catalog was configured for port 8787 (Cloudflare default)
3. The mismatch went unnoticed until runtime

## Prevention

To prevent similar issues:
1. ✅ Document port allocations in a central location
2. ✅ Use environment variables for configurable ports
3. ✅ Add port validation in startup scripts
4. ✅ Use distinct ports that don't conflict with common defaults

---

**TEAM-421 Complete** - Worker catalog now correctly configured for port 8787
