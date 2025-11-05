# macOS Tauri White Screen Fix

## Problem
Running `./rbee` on macOS opened a Tauri window but showed only a white screen instead of loading the UI.

## Root Cause
**macOS WKWebView (Tauri's webview engine) has issues resolving `localhost` in some configurations.**

On macOS, `localhost` can resolve to either IPv4 (`127.0.0.1`) or IPv6 (`::1`), and WKWebView sometimes fails to connect when using the hostname `localhost` directly.

## Solution
Changed all `localhost` references to explicit `127.0.0.1` IP address.

### Files Changed

#### 1. `bin/00_rbee_keeper/tauri.conf.json`
```diff
  "build": {
-   "devUrl": "http://localhost:5173",
+   "devUrl": "http://127.0.0.1:5173",
  }
```

#### 2. `bin/00_rbee_keeper/ui/vite.config.ts`
```diff
  server: {
+   host: '127.0.0.1', // TEAM-XXX: mac compat - Bind to 127.0.0.1 for Tauri WKWebView
    port: 5173,
    strictPort: true,
  },
```

#### 3. `bin/00_rbee_keeper/tauri.conf.json` (Permissions)
```diff
  "permissions": [
    "core:default",
    "core:event:allow-listen",
    "core:event:allow-emit",
+   "core:window:allow-create",
+   "core:webview:allow-create-webview",
+   "core:webview:allow-internal-toggle-devtools"
  ]
```

## Why This Works

### localhost vs 127.0.0.1
- **`localhost`**: Hostname that resolves via DNS/hosts file (can be IPv4 or IPv6)
- **`127.0.0.1`**: Explicit IPv4 loopback address (no DNS resolution needed)

### macOS WKWebView Behavior
WKWebView on macOS sometimes has issues with:
1. **DNS resolution delays** - `localhost` lookup can timeout
2. **IPv6 preference** - macOS prefers IPv6, but Vite binds to IPv4 by default
3. **Security policies** - ATS (App Transport Security) can block localhost in some cases

Using `127.0.0.1` bypasses all these issues by:
- ✅ No DNS resolution needed
- ✅ Explicit IPv4 (matches Vite's default bind)
- ✅ Loopback is always allowed by ATS

## Testing

### Before Fix
```bash
./rbee
# Result: White screen, no UI loaded
```

### After Fix
```bash
# 1. Start dev server
turbo dev --concurrency 30

# 2. Run rbee
./rbee
# Result: ✅ Tauri window opens with UI loaded from http://127.0.0.1:5173
```

## Verification

```bash
# Check dev server is bound to 127.0.0.1
lsof -i :5173 | grep LISTEN
# Should show: TCP 127.0.0.1:5173 (LISTEN)

# Test direct access
curl http://127.0.0.1:5173
# Should return HTML

# Test in browser
open http://127.0.0.1:5173
# Should load UI
```

## Related Issues

This is a known issue with Tauri v2 on macOS:
- [tauri-apps/tauri#5084](https://github.com/tauri-apps/tauri/issues/5084)
- [tauri-apps/tauri#6442](https://github.com/tauri-apps/tauri/issues/6442)

## Impact

### ✅ Fixed
- Tauri GUI now loads correctly on macOS
- Dev server accessible from WKWebView
- Hot reload works

### ⚠️ Side Effects
- Dev server now only accessible on `127.0.0.1` (not `0.0.0.0`)
- Cannot access dev server from other devices on network
- This is acceptable for local development

### 🔄 Future Consideration
For production builds, this doesn't matter because Tauri serves from `dist/` folder, not a dev server.

## Additional macOS Compatibility Notes

### Other Potential Issues (Not Encountered)
1. **CORS** - Vite's CORS is permissive by default
2. **CSP** - Set to `null` (permissive) in tauri.conf.json
3. **Firewall** - macOS firewall doesn't block localhost by default

### macOS-Specific Tauri Features
- Uses WKWebView (Safari's rendering engine)
- Requires code signing for distribution
- `.icns` icon format for macOS bundles

## Summary

**Problem:** White screen in Tauri on macOS  
**Cause:** WKWebView can't resolve `localhost`  
**Fix:** Use `127.0.0.1` instead  
**Status:** ✅ RESOLVED

---

**Last Updated:** 2025-11-05  
**Tested On:** macOS (Darwin)  
**Tauri Version:** 2.x
