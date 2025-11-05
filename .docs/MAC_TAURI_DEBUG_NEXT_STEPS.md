# Tauri White Screen - Debug Next Steps

## What We've Done

### 1. Fixed localhost → 127.0.0.1
- ✅ Changed `tauri.conf.json` devUrl to `http://127.0.0.1:5173`
- ✅ Changed `vite.config.ts` to bind to `127.0.0.1`
- ✅ Added webview permissions

### 2. Added Debug Logging
- ✅ Added startup logging to show mode (DEBUG/RELEASE)
- ✅ Added expected URL logging
- ✅ Auto-open DevTools in debug mode
- ✅ Log available windows

## Why You're Not Seeing Errors

The binary is running successfully (exit code 0), which means:
- ✅ Tauri is starting correctly
- ✅ No Rust panics
- ✅ Window is being created

The issue is **inside the webview** - JavaScript errors, network failures, or loading issues.

## Next Steps - RUN THIS

### Step 1: Rebuild (DONE)
```bash
cargo build --bin rbee-keeper
```

### Step 2: Make Sure Dev Server is Running
```bash
# In another terminal
turbo dev --concurrency 30

# Or just the keeper UI
cd bin/00_rbee_keeper/ui
pnpm dev
```

### Step 3: Run rbee and CHECK THE OUTPUT
```bash
./rbee
```

**You should now see:**
```
🚀 Launching Tauri GUI in DEBUG mode
   Expected dev URL: http://127.0.0.1:5173
🔍 Available windows: ["main"]
🔍 DevTools opened - check for console errors
```

### Step 4: Look at the DevTools Window

A **second window** should open automatically - this is the DevTools inspector.

**Check the Console tab for:**
- ❌ Red errors (JavaScript errors)
- ⚠️ Yellow warnings
- 🔴 Failed network requests

**Check the Network tab for:**
- Is `http://127.0.0.1:5173` being requested?
- Is it returning 200 OK or failing?
- Are there CORS errors?

## Common Issues You Might See

### Issue 1: "Failed to load resource: net::ERR_CONNECTION_REFUSED"
**Cause:** Dev server not running on port 5173  
**Fix:** Start the dev server first

### Issue 2: "Failed to load resource: net::ERR_NAME_NOT_RESOLVED"
**Cause:** Still trying to use `localhost` instead of `127.0.0.1`  
**Fix:** Check tauri.conf.json is using `127.0.0.1`

### Issue 3: JavaScript errors in console
**Cause:** Frontend code has errors  
**Fix:** Check the specific error message

### Issue 4: CORS errors
**Cause:** Vite not allowing Tauri webview  
**Fix:** Add CORS config to vite.config.ts

### Issue 5: "Uncaught ReferenceError: process is not defined"
**Cause:** Missing process polyfill  
**Fix:** Already added in vite.config.ts (`define: { 'process.env': {} }`)

## If DevTools Don't Open

### Manual Method
When the white screen appears, press:
- **macOS:** `Cmd + Option + I`
- **Linux:** `Ctrl + Shift + I`
- **Windows:** `Ctrl + Shift + I`

### Check Build Mode
```bash
# Make sure you're using debug build
cargo build --bin rbee-keeper  # NOT --release
```

## Debugging Checklist

- [ ] Dev server is running on port 5173
- [ ] Can access http://127.0.0.1:5173 in browser
- [ ] Rebuilt rbee-keeper with debug logging
- [ ] Ran `./rbee` and saw startup logs
- [ ] DevTools window opened (or opened manually)
- [ ] Checked Console tab for errors
- [ ] Checked Network tab for failed requests
- [ ] Identified the specific error message

## Report Back

Once you run `./rbee`, please share:

1. **Terminal output** - What logs do you see?
2. **DevTools Console** - Any red errors?
3. **DevTools Network** - Is 127.0.0.1:5173 being requested?
4. **Screenshot** - If possible, screenshot of the DevTools

This will tell us exactly what's failing!

---

**Last Updated:** 2025-11-05  
**Status:** Waiting for debug output
