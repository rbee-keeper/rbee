# TAURI WHITE SCREEN FIX - TESTING INSTRUCTIONS

**Status:** ✅ FIX IMPLEMENTED - NEEDS TESTING  
**Date:** 2025-11-05

---

## QUICK TEST (5 MINUTES)

### Terminal 1: Start Vite Dev Server
```bash
cd bin/00_rbee_keeper/ui
pnpm dev
```

**Expected output:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
```

### Terminal 2: Build and Run Tauri
```bash
# From project root
cargo build --bin rbee-keeper
./rbee
```

**Expected output:**
```
🚀 Launching Tauri GUI in DEBUG mode
   Expected dev URL: http://localhost:5173
🔍 Available windows: ["main"]
🔍 Found window: main
✅ Injected test HTML
🔍 DevTools opened
```

### Expected Result

**✅ SUCCESS:**
- Window opens (960x1080)
- UI loads (NOT white screen)
- DevTools open automatically
- UI is fully interactive

**❌ FAILURE (Still White):**
See "Troubleshooting" section below

---

## WHAT WAS CHANGED

### Root Cause
Tauri v2 treats `http://127.0.0.1:5173` as a "remote domain" requiring explicit permissions. IP addresses don't work with Tauri's remote domain configuration - only domain names.

### The Fix
1. **Changed from `127.0.0.1` to `localhost`** in 3 files:
   - `tauri.conf.json` (devUrl and window.url)
   - `ui/vite.config.ts` (server.host)
   - `src/main.rs` (debug messages)

2. **Added remote URL permissions** to capability:
   ```json
   "remote": {
     "urls": ["http://localhost:5173"]
   }
   ```

---

## VERIFICATION CHECKLIST

After running the test above, verify:

### Basic Functionality
- [ ] Window opens
- [ ] **NO WHITE SCREEN** (this is the key fix)
- [ ] UI displays correctly
- [ ] No error messages in terminal

### Advanced Testing (Optional)
- [ ] Right-click in window → "Inspect" opens DevTools
- [ ] Console shows no errors
- [ ] Tauri commands work (test any UI button)
- [ ] macOS Console.app shows no WKWebView errors

---

## TROUBLESHOOTING

### Still Showing White Screen

**1. Check Dev Server is Running**
```bash
lsof -i :5173
# Should show: node ... vite
```

**2. Check localhost Resolves**
```bash
ping localhost
# Should resolve to 127.0.0.1 or ::1
```

**3. Check Vite is Binding to localhost**
```bash
curl -I http://localhost:5173
# Should return: HTTP/1.1 200 OK
```

**4. Check macOS Console Logs**
```bash
# Open Console.app
open /System/Applications/Utilities/Console.app

# Filter for "rbee" or "WKWebView"
# Run ./rbee
# Look for errors
```

### Port Already in Use

```bash
# Kill process on port 5173
lsof -ti :5173 | xargs kill -9

# Restart dev server
cd bin/00_rbee_keeper/ui
pnpm dev
```

### Compilation Errors

```bash
# Clean and rebuild
cargo clean
cargo build --bin rbee-keeper
```

---

## IF FIX DOESN'T WORK

The fix should work based on extensive research of Tauri v2 issues. If it doesn't:

### 1. Verify Tauri Version
```bash
cd bin/00_rbee_keeper
cargo tree | grep "^tauri "
# Should show: tauri v2.x.x
```

### 2. Check for macOS-Specific Issues
- macOS version: `sw_vers`
- WKWebView version: Check Console.app
- Security settings: System Preferences → Security & Privacy

### 3. Try Production Build
```bash
# Build frontend
cd bin/00_rbee_keeper/ui
pnpm build

# Build Tauri in release mode
cd ../..
cargo build --release --bin rbee-keeper

# Run
./target/release/rbee-keeper
```

If production build works but dev doesn't, it's a dev server connectivity issue.

---

## DOCUMENTATION

**Full Investigation:**
`.docs/TAURI_WHITE_SCREEN_FIX_COMPLETE.md`

**Original Handoff:**
`.docs/HANDOFF_TAURI_WHITE_SCREEN.md`

**Related Issues:**
- [Tauri #7009](https://github.com/tauri-apps/tauri/issues/7009) - IP addresses don't work
- [Tauri #11934](https://github.com/tauri-apps/tauri/issues/11934) - Remote API access

---

## SUCCESS CRITERIA

When the fix works, you should see:

```
┌──────────────────────────────────────┐
│  rbee Keeper                    ⚫ ⚫ ⚫│
├──────────────────────────────────────┤
│                                       │
│  ✨ ACTUAL UI IS VISIBLE ✨          │
│                                       │
│  NOT a white screen!                  │
│                                       │
│  UI is fully functional               │
│                                       │
└──────────────────────────────────────┘
```

**NOT:**
```
┌──────────────────────────────────────┐
│  rbee Keeper                    ⚫ ⚫ ⚫│
├──────────────────────────────────────┤
│                                       │
│                                       │
│      (white screen - nothing)         │
│                                       │
│                                       │
└──────────────────────────────────────┘
```

---

**Next Steps After Successful Test:**
1. Update handoff doc with "✅ FIXED" status
2. Run production build test
3. Consider removing debug HTML injection code from main.rs
4. Archive old investigation docs

**Team Signature:** TEAM-XXX  
**Fix Implementation:** Complete  
**Testing Required:** Yes (on macOS)
