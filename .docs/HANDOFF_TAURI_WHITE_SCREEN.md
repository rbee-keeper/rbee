# HANDOFF: Tauri White Screen on macOS - URGENT

**Status:** BLOCKED - White screen, no errors, need deep debugging  
**Priority:** CRITICAL - First macOS test, must work  
**Team:** Next team must fix this NOW

---

## THE PROBLEM

Running `./rbee` on macOS:
- ✅ Binary compiles (exit code 0)
- ✅ Binary runs (exit code 0, no errors)
- ✅ Window opens (correct size: 960x1080)
- ❌ **Screen is WHITE (should show UI)**
- ❌ **NO error messages anywhere**

---

## VERIFIED FACTS

### Dev Server is WORKING
```bash
# Vite is running
lsof -i :5173
# node 79122 - vite on port 5173

# Server responds
curl -I http://127.0.0.1:5173
# HTTP/1.1 200 OK

# Browser works
open http://127.0.0.1:5173
# Shows UI correctly in Safari
```

### Binary is WORKING
```bash
cargo build --bin rbee-keeper
# Compiles successfully

./rbee
# Exit code: 0 (no Rust errors)
# Window opens
# Screen is white
```

### Configuration is CORRECT
```json
// bin/00_rbee_keeper/tauri.conf.json
{
  "build": {
    "devUrl": "http://127.0.0.1:5173"
  },
  "app": {
    "windows": [{
      "label": "main",
      "url": "http://127.0.0.1:5173"
    }]
  }
}
```

```ts
// bin/00_rbee_keeper/ui/vite.config.ts
server: {
  host: '127.0.0.1',
  port: 5173,
  strictPort: true
}
```

---

## WHAT WE TRIED (ALL FAILED)

1. ❌ Changed `localhost` → `127.0.0.1` (both Tauri and Vite)
2. ❌ Added window label `"main"`
3. ❌ Added explicit `"url"` in window config
4. ❌ Added webview permissions to capabilities
5. ❌ Added devtools auto-open code (doesn't show errors)
6. ❌ Added logging (shows window exists, no errors)

---

## THE MYSTERY

**Why is there NO error message?**

- Tauri starts successfully
- Window is created
- But webview either:
  1. Not loading the URL at all
  2. Loading but failing silently
  3. Loading but rendering nothing

**We have ZERO visibility into what's happening inside the webview.**

---

## CRITICAL FILES

### Main Entry Point
```
bin/00_rbee_keeper/src/main.rs
- Line 83-176: launch_gui() function
- Line 133-174: .setup() with debug code
```

### Tauri Config
```
bin/00_rbee_keeper/tauri.conf.json
- Line 8: devUrl = "http://127.0.0.1:5173"
- Line 44-56: Window config with label and url
```

### Vite Config
```
bin/00_rbee_keeper/ui/vite.config.ts
- Line 17-21: Server config binding to 127.0.0.1
```

### UI Entry Point
```
bin/00_rbee_keeper/ui/src/main.tsx
- React app entry point
- Uses TauriProvider, ThemeProvider, QueryProvider
```

---

## WHAT THE NEXT TEAM MUST DO

### 1. VERIFY WEBVIEW EXISTS (5 min)
The test HTML injection code is already in `main.rs` line 148-164.

**Run:**
```bash
cargo build --bin rbee-keeper
./rbee
```

**Check terminal output:**
- Does it say "✅ Injected test HTML"?
- Does it say "❌ Failed to inject HTML"?
- Does the screen turn RED or stay WHITE?

**If RED:** Webview works, problem is URL loading  
**If WHITE:** Webview broken, macOS issue  
**If error:** Read the error message

### 2. CHECK MACOS WEBVIEW LOGS (10 min)
macOS Console.app shows WKWebView errors.

**Steps:**
```bash
# Open Console.app
open /System/Applications/Utilities/Console.app

# Filter for "rbee" or "WKWebView"
# Run ./rbee
# Check for errors in Console.app
```

**Look for:**
- Network errors (ERR_CONNECTION_REFUSED)
- Security errors (CORS, CSP)
- WKWebView crashes
- Permission denials

### 3. TRY PRODUCTION BUILD (15 min)
Eliminate dev server from equation.

**Steps:**
```bash
# Build frontend
cd bin/00_rbee_keeper/ui
pnpm build
cd ../..

# Build Tauri in release
cargo build --release --bin rbee-keeper

# Run
./rbee
```

**If this works:** Dev server connection issue  
**If this fails:** Webview or Tauri issue

### 4. USE CARGO TAURI DEV (10 min)
Tauri's official dev command might show more errors.

**Steps:**
```bash
# Install tauri-cli if needed
cargo install tauri-cli --version "^2.0.0"

# Run from Tauri directory
cd bin/00_rbee_keeper
cargo tauri dev
```

**This might show:**
- Build errors
- Webview creation errors
- URL loading errors

### 5. CHECK MACOS PERMISSIONS (5 min)
macOS might be blocking network access.

**Check:**
- System Preferences → Security & Privacy → Privacy
- Look for "rbee-keeper" or "rbee"
- Grant all permissions

### 6. TEST WITH MINIMAL TAURI APP (30 min)
Create a fresh Tauri app to see if it works.

**Steps:**
```bash
cd /tmp
npm create tauri-app@latest test-app
cd test-app
npm install
npm run tauri dev
```

**If this works:** Our config is wrong  
**If this fails:** macOS Tauri is broken

### 7. CHECK MACOS VERSION (2 min)
Some macOS versions have WKWebView bugs.

**Run:**
```bash
sw_vers
system_profiler SPSoftwareDataType
```

**Known issues:**
- macOS < 10.15: WKWebView might not work
- macOS 14+: New security restrictions

---

## DEBUGGING TOOLS

### Enable All Tauri Logging
```bash
# Set environment variable
RUST_LOG=tauri=trace,wry=trace ./rbee
```

### Check Network Requests
```bash
# Install mitmproxy
brew install mitmproxy

# Run proxy
mitmproxy -p 8888

# Configure macOS to use proxy
# Run ./rbee
# See if requests appear in mitmproxy
```

### Inspect WKWebView Process
```bash
# While ./rbee is running
ps aux | grep -i webview
ps aux | grep -i wkwebview

# Check if webview process exists
```

---

## POSSIBLE ROOT CAUSES

### 1. macOS WKWebView Bug
- WKWebView can't load localhost/127.0.0.1
- **Fix:** Use file:// protocol or embed HTML
- **Test:** Production build

### 2. Tauri v2 macOS Bug
- Tauri v2 has known macOS issues
- **Fix:** Downgrade to Tauri v1 or wait for fix
- **Test:** Minimal Tauri app

### 3. Security Policy Blocking
- macOS App Transport Security blocking HTTP
- **Fix:** Add ATS exception or use HTTPS
- **Test:** Check Console.app for ATS errors

### 4. Missing Webview Initialization
- Tauri not creating webview on macOS
- **Fix:** Debug Tauri source or file bug
- **Test:** Check process list for webview

### 5. React App Error
- React app crashes immediately
- **Fix:** Check browser console (if we can access it)
- **Test:** Inject simple HTML (already tried)

---

## SUCCESS CRITERIA

When fixed, running `./rbee` should:
1. Open window
2. Load UI from http://127.0.0.1:5173
3. Show the rbee-keeper interface
4. No white screen

---

## RESOURCES

### Tauri Docs
- https://v2.tauri.app/develop/debug/
- https://v2.tauri.app/reference/config/

### Known Issues
- https://github.com/tauri-apps/tauri/issues?q=white+screen+macos
- https://github.com/tauri-apps/tauri/issues?q=wkwebview

### macOS WKWebView
- https://developer.apple.com/documentation/webkit/wkwebview

---

## CONTACT INFO

**Previous Team:** Left debug code in main.rs  
**Files Modified:**
- `bin/00_rbee_keeper/tauri.conf.json` (added label, url, changed to 127.0.0.1)
- `bin/00_rbee_keeper/ui/vite.config.ts` (bind to 127.0.0.1)
- `bin/00_rbee_keeper/src/main.rs` (added debug logging, test HTML injection)

**Time Spent:** 2+ hours  
**Status:** BLOCKED - Need macOS expertise or Tauri expertise

---

## IMMEDIATE NEXT STEP

**RUN THIS FIRST:**
```bash
./rbee 2>&1 | tee rbee-debug.log
```

**Check the output for:**
- "✅ Injected test HTML" or "❌ Failed to inject HTML"
- Any error messages
- Window labels

**Then open macOS Console.app and filter for "rbee" while running.**

**This will give you the FIRST clue about what's actually failing.**

---

**GOOD LUCK. THIS MUST WORK.**
