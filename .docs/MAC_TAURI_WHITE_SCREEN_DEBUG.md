# Tauri White Screen Issue on macOS - Debug Guide

## Problem
Running `./rbee` opens a Tauri window but shows a white screen instead of loading the UI from `http://localhost:5173`.

## Verified Working
- ✅ Dev server is running on port 5173
- ✅ `curl http://localhost:5173` returns valid HTML
- ✅ Binary builds successfully
- ✅ Tauri window opens with correct size

## Potential Causes

### 1. macOS Webview Permissions
macOS webview might need additional permissions to load localhost content.

**Fix Applied:**
- Added `core:webview:allow-create-webview` permission
- Added `core:webview:allow-internal-toggle-devtools` permission

### 2. CSP (Content Security Policy)
The CSP might be blocking localhost resources.

**Current Config:**
```json
"security": {
  "csp": null
}
```

### 3. Vite Dev Server CORS
Vite might not be allowing connections from the Tauri webview.

**Check:**
```bash
# Vite config should allow all origins in dev mode
```

### 4. macOS Network Security
macOS might be blocking localhost connections from the webview.

**Check:**
- System Preferences → Security & Privacy → Firewall
- Allow incoming connections for the app

### 5. Tauri devUrl Configuration
The `devUrl` might not be correctly configured for macOS.

**Current Config:**
```json
"build": {
  "devUrl": "http://localhost:5173"
}
```

## Debugging Steps

### Step 1: Enable Tauri DevTools
Press `Cmd+Option+I` when the white screen appears to open DevTools.

**Look for:**
- Console errors
- Network tab - is the request to localhost:5173 failing?
- Security errors

### Step 2: Check Tauri Logs
```bash
./rbee 2>&1 | tee tauri-debug.log
```

**Look for:**
- Webview creation errors
- URL loading errors
- Permission errors

### Step 3: Test with Production Build
```bash
cd bin/00_rbee_keeper/ui
pnpm build
cd ../..
cargo build --release --bin rbee-keeper
./rbee
```

If production build works but dev doesn't, it's a dev server connectivity issue.

### Step 4: Test Direct Browser Access
```bash
open http://localhost:5173
```

If this works in Safari but not in Tauri, it's a Tauri-specific issue.

### Step 5: Check macOS Webview Version
Tauri v2 uses WKWebView on macOS. Check if it's up to date.

```bash
sw_vers  # macOS version
```

## Known Issues

### Tauri v2 + macOS + localhost
Some versions of macOS have issues with WKWebView loading localhost URLs.

**Workarounds:**
1. Use `127.0.0.1` instead of `localhost`
2. Use a different port
3. Disable ATS (App Transport Security) - NOT RECOMMENDED

### Vite + Tauri + macOS
Vite's HMR (Hot Module Replacement) might not work correctly with Tauri on macOS.

**Workarounds:**
1. Disable HMR in vite.config.ts
2. Use production build for testing

## Next Steps

1. **Enable DevTools** - Press `Cmd+Option+I` in the white screen
2. **Check Console** - Look for JavaScript errors
3. **Check Network** - Is localhost:5173 being requested?
4. **Check Tauri Logs** - Run `./rbee 2>&1 | grep -i error`

## Possible Fixes

### Fix 1: Change localhost to 127.0.0.1
```json
// tauri.conf.json
"build": {
  "devUrl": "http://127.0.0.1:5173"
}
```

```ts
// vite.config.ts
server: {
  host: '127.0.0.1',
  port: 5173
}
```

### Fix 2: Disable Vite strictPort
```ts
// vite.config.ts
server: {
  port: 5173,
  strictPort: false  // Allow fallback to another port
}
```

### Fix 3: Add Vite CORS Headers
```ts
// vite.config.ts
server: {
  port: 5173,
  cors: true,
  headers: {
    'Access-Control-Allow-Origin': '*'
  }
}
```

### Fix 4: Enable Tauri DevTools by Default
```json
// tauri.conf.json
"app": {
  "windows": [{
    "devtools": true  // Enable devtools in dev mode
  }]
}
```

## References
- [Tauri v2 Security Docs](https://v2.tauri.app/concept/security/)
- [Tauri v2 Window Config](https://v2.tauri.app/reference/config/#windowconfig)
- [Vite Server Options](https://vitejs.dev/config/server-options.html)
