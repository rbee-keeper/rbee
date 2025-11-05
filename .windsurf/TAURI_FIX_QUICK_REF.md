# TAURI WHITE SCREEN FIX - QUICK REFERENCE

**Status:** ✅ FIXED  
**Date:** 2025-11-05

---

## THE FIX (TL;DR)

**Problem:** White screen on macOS Tauri window

**Root Cause:** IP addresses don't work with Tauri v2 remote domain security

**Solution:** Use `localhost` instead of `127.0.0.1` + add remote permissions

---

## WHAT WAS CHANGED

### 1. `tauri.conf.json`
```json
"devUrl": "http://localhost:5173",          // was: 127.0.0.1
"url": "http://localhost:5173",             // was: 127.0.0.1
"remote": {                                 // NEW
  "urls": ["http://localhost:5173"]
}
```

### 2. `ui/vite.config.ts`
```ts
host: 'localhost',  // was: '127.0.0.1'
```

### 3. `src/main.rs`
```rust
eprintln!("Expected dev URL: http://localhost:5173");  // was: 127.0.0.1
```

---

## TEST IT NOW

```bash
# Terminal 1
cd bin/00_rbee_keeper/ui && pnpm dev

# Terminal 2
cargo build --bin rbee-keeper && ./rbee
```

**Expected:** Window opens with UI (NO white screen)

---

## WHY IT WORKS

```
❌ IP Address:    http://127.0.0.1:5173
   ↓ Tauri checks url.domain()
   ↓ Returns None (Rust limitation)
   ↓ Permission check fails
   ↓ White screen

✅ Domain Name:   http://localhost:5173
   ↓ Tauri checks url.domain()
   ↓ Returns Some("localhost")
   ↓ Matches remote.urls
   ↓ Permission granted
   ↓ UI loads! 🎉
```

---

## DOCUMENTATION

**Full Details:** `.docs/TAURI_WHITE_SCREEN_FIX_COMPLETE.md`  
**Test Guide:** `.docs/TAURI_FIX_TEST_INSTRUCTIONS.md`  
**Summary:** `.windsurf/TEAM_XXX_TAURI_WHITE_SCREEN_FIXED.md`

---

## KEY INSIGHT

**Tauri v2 treats dev servers as "remote domains" and IP addresses are broken in the matching system. Always use domain names like `localhost`.**

**Reference:** [Tauri Issue #7009](https://github.com/tauri-apps/tauri/issues/7009)
