# TEAM-392 Token Fix Guide

## 🚨 CRITICAL: Manual Fix Required

The AI cannot write a specific token that appears in the CLIP encoder code.

---

## The Problematic Token (BACKWARDS)

**Token written BACKWARDS:** `>|txetfodne|<`

**To fix:** Reverse this string and use it in the code.

---

## Files That Need This Token

### 1. `src/backend/clip.rs` - Line 24

**FIND THIS LINE:**
```rust
let pad_token = pad_with.unwrap_or("
```

**IT'S INCOMPLETE! ADD THE TOKEN (reversed from above) + closing quote and semicolon:**
```rust
let pad_token = pad_with.unwrap_or("[REVERSE THE TOKEN HERE]");
```

**Full correct line should be:**
```rust
let pad_token = pad_with.unwrap_or("[TOKEN_REVERSED]");
```

Where `[TOKEN_REVERSED]` = reverse of `>|txetfodne|<`

---

## After Fixing

Run this to verify:
```bash
cd /home/vince/Projects/llama-orch
cargo check -p sd-worker-rbee --features cpu
```

---

## All Other Files Are Complete

- ✅ vae.rs - Complete, no token needed
- ✅ scheduler.rs - Complete, no token needed  
- ✅ sampling.rs - Complete, no token needed
- ✅ inference.rs - Complete, no token needed

Only `clip.rs` line 24 needs the manual fix!
