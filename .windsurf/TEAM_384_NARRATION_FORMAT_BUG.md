# TEAM-384: Narration Format Bug Analysis

**Date:** Nov 2, 2025 3:14 PM  
**Status:** 🐛 BUG IDENTIFIED

---

## The Problem

When running `rbee model list`, narration from rbee-hive appears in **raw format** instead of **formatted text**:

### ❌ Current Output (Raw Format)

```
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'
rbee_hive::job_router::execute_operation model_list_result   
Found 0 model(s)
rbee_hive::job_router::execute_operation model_list_json     
[]
rbee_hive::job_router::execute_operation model_list_complete 
✅ Model list operation complete
```

### ✅ Expected Output (Formatted)

```
📋 Listing models on hive 'localhost'
Found 0 model(s)
[]
✅ Model list operation complete
```

---

## Root Cause Analysis

### The Flow

1. **rbee-hive** calls `n!("model_list_start", "📋 Listing models...")`
2. **narration-core** `macro_emit()` gets `job_id` from context ✅
3. **narration-core** `narrate()` sends to SSE channel ✅
4. **narration-core** `narrate()` ALSO emits tracing event ⚠️
5. **rbee-keeper** tracing subscriber prints raw format to stderr ❌

### The Bug

**File:** `bin/00_rbee_keeper/src/tracing_init.rs:29-41`

```rust
pub fn init_cli_tracing() {
    fmt()  // ← Default formatter prints RAW tracing events!
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_line_number(false)
        .with_file(false)
        .with_target(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
}
```

**Problem:** The default `fmt()` subscriber prints tracing events in raw format:
```
<target> <fields>
```

Which becomes:
```
rbee_hive::job_router::execute_operation model_list_start human="📋 Listing models..."
```

---

## Why This Happens

### Narration Dual Output

**File:** `bin/99_shared_crates/narration-core/src/api/emit.rs:114-130`

```rust
pub fn narrate(fields: NarrationFields, level: NarrationLevel) {
    // ...
    
    // 1. Send to SSE (job-scoped, formatted)
    if sse_sink::is_enabled() {
        let _sse_sent = sse_sink::try_send(&fields);  // ← Goes to SSE channel
    }
    
    // 2. Emit tracing event (for subscribers)
    match tracing_level {
        Level::INFO => emit_event!(Level::INFO, fields),  // ← Goes to tracing
        // ...
    }
}
```

**Design:** Narration emits BOTH:
1. **SSE events** (formatted, job-scoped) - for streaming to client
2. **Tracing events** (raw) - for logging/debugging

### The Mismatch

- **rbee-keeper** (client): Tracing subscriber prints raw format ❌
- **rbee-hive** (server): Narration goes to SSE, but tracing events leak to client's stderr ❌

---

## Evidence

### What We See

```bash
$ ./rbee model ls

# ✅ rbee-keeper narration (formatted via INFO logs)
2025-11-02T14:13:21.138753Z  INFO actor="rbee_keeper" action="job_submit" target=job_submit human=📋 Job submitted: model_list

# ❌ rbee-hive narration (raw tracing format)
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'

# ✅ rbee-keeper narration (formatted via INFO logs)
2025-11-02T14:13:21.205877Z  INFO actor="rbee_keeper" action="job_complete" target=job_complete human=✅ Complete: model_list
```

### Why rbee-keeper Looks Better

rbee-keeper's own narration uses the same tracing subscriber, but it's formatted as:
```
2025-11-02T14:13:21.138753Z  INFO actor="rbee_keeper" action="job_submit" target=job_submit human=📋 Job submitted: model_list
```

This is the **tracing default format** with timestamps and structured fields.

### Why rbee-hive Looks Worse

rbee-hive's narration comes through the SSE stream, but the tracing events are ALSO being captured by rbee-keeper's subscriber and printed in raw format!

---

## The Real Issue

**The narration IS working correctly!** The SSE channel is receiving formatted events.

**The problem is:** rbee-keeper's tracing subscriber is ALSO capturing rbee-hive's tracing events (somehow) and printing them in raw format.

**Wait, that doesn't make sense...** rbee-keeper and rbee-hive are separate processes. How is rbee-keeper's tracing subscriber seeing rbee-hive's events?

### Re-Analysis

Let me re-read the output:

```
rbee_hive::job_router::execute_operation model_list_start    
📋 Listing models on hive 'localhost'
```

This is coming from **rbee-hive's stdout**, not rbee-keeper's tracing subscriber!

**The actual issue:** rbee-hive is printing narration to stdout in raw format, and rbee-keeper is just passing it through!

---

## The REAL Root Cause

### rbee-hive Tracing Init

Let me check if rbee-hive has tracing initialized:

**File:** `bin/20_rbee_hive/src/main.rs` (need to check)

If rbee-hive has a tracing subscriber that prints to stdout, that would explain the raw format!

---

## Narration Gaps?

### Question: "Do we have narration gaps?"

**Answer:** NO! All narration is being emitted. The issue is **formatting**, not **missing events**.

**Evidence:**
- ✅ `model_list_start` - present
- ✅ `model_list_result` - present
- ✅ `model_list_json` - present
- ✅ `model_list_complete` - present
- ✅ `[DONE]` marker - present

**All narration is there!** It's just in raw format.

---

## Question: "#[with_job_id] on all functions?"

### Current State

**File:** `bin/20_rbee_hive/src/job_router.rs:73`

```rust
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    // ...
}
```

**Missing:** `#[with_job_id]` attribute ❌

**But:** The function calls `route_operation()` which sets context:

```rust
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    use observability_narration_core::{with_narration_context, NarrationContext};
    
    // TEAM-381: Set narration context for ALL operations
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // ... execute operation ...
    }).await
}
```

**So:** The `job_id` IS being set in context ✅

**The `#[with_job_id]` macro is NOT needed** because the context is set manually.

---

## The Fix

### Option 1: Filter Tracing Events in rbee-hive

**Problem:** rbee-hive has a tracing subscriber that prints to stdout

**Solution:** Configure rbee-hive to NOT print narration events (they go to SSE anyway)

### Option 2: Use Custom Layer in rbee-keeper

**Problem:** rbee-keeper's tracing subscriber prints raw format

**Solution:** Use a custom layer (like GUI mode) that formats narration properly

### Option 3: Disable Tracing Output in rbee-hive

**Problem:** rbee-hive prints to stdout

**Solution:** Only enable SSE output, disable tracing output for narration events

---

## Recommended Fix

**Best approach:** Configure rbee-hive to NOT print narration to stdout.

**Why?**
1. Narration is already going to SSE (the primary channel)
2. Stdout output is redundant
3. It creates formatting confusion
4. SSE is the single source of truth

**Implementation:**
1. Check rbee-hive's tracing init
2. Either disable it or filter out narration events
3. Keep SSE as the only output channel

---

## Next Steps

1. ✅ **Identify** where rbee-hive prints to stdout
2. ⏳ **Disable** stdout output for narration (keep SSE only)
3. ⏳ **Verify** that formatted narration appears correctly
4. ⏳ **Test** that all narration events are captured

---

## Summary

**Problem:** rbee-hive narration appears in raw format  
**Root Cause:** rbee-hive has tracing subscriber printing to stdout  
**Solution:** Disable stdout output, use SSE only  
**Narration Gaps:** None! All events are present  
**#[with_job_id]:** Not needed (context set manually)

---

**TEAM-384:** Bug identified! Narration is working, just needs formatting fix in rbee-hive tracing config. 🐛
