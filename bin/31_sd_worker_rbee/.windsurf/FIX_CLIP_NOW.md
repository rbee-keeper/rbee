# Fix clip.rs - ONE CHANGE ONLY

## File: `src/backend/clip.rs` Line 27

**Current (BACKWARDS):**
```rust
let pad_token = pad_with.unwrap_or(">|txetfodne|<");
```

**Change to (REVERSED):**
```rust
let pad_token = pad_with.unwrap_or("
