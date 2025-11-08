# TEAM-388: Build Fix - SseEvent Serialization

**Status:** ✅ FIXED  
**Date:** Nov 2, 2025

## Error

```
error[E0609]: no field `formatted` on type `SseEvent` 
   --> bin/10_queen_rbee/src/http/jobs.rs:147:59
    |
147 | ...                   .unwrap_or_else(|_| event.formatted.clone());
    |                                                 ^^^^^^^^^ unknown field
```

## Root Cause

`SseEvent` is an enum with three variants:
```rust
pub enum SseEvent {
    Narration(NarrationEvent),  // ← formatted field is HERE
    Data(DataEvent),
    Done,
}
```

The code was trying to access `event.formatted` when `event` is an `SseEvent` enum, not a `NarrationEvent` struct.

## Fix

**File:** `bin/10_queen_rbee/src/http/jobs.rs:147`

**Before:**
```rust
let json = serde_json::to_string(&event)
    .unwrap_or_else(|_| event.formatted.clone());
```

**After:**
```rust
let json = serde_json::to_string(&event)
    .unwrap_or_else(|_| format!("{{\"type\":\"error\",\"message\":\"Failed to serialize event\"}}"));
```

## Why This Works

1. `serde_json::to_string(&event)` serializes the entire `SseEvent` enum
2. The enum has `#[derive(serde::Serialize)]` so it serializes correctly
3. On error, we return a valid JSON error object instead of trying to access a non-existent field

## Serialization Output

The `SseEvent` enum serializes with a `type` tag:

```json
// Narration variant
{
  "type": "narration",
  "formatted": "[actor] action: message",
  "actor": "...",
  "action": "...",
  ...
}

// Data variant
{
  "type": "data",
  "payload": {...}
}

// Done variant
{
  "type": "done"
}
```

## Build Result

✅ `cargo build` succeeds  
⚠️ Only cosmetic warnings (unused imports)

## Related

This fix was needed after implementing TEAM-388 worker catalog operations, which triggered a full rebuild that exposed this pre-existing bug.
