# TEAM-384: Dual-Channel SSE Architecture

**Status:** ✅ COMPLETE

**Date:** Nov 2, 2025

## Problem

The SSE stream was mixing two different types of information:
1. **Narration** - Human-readable messages for CLI/UI logs
2. **Data** - Structured JSON for UI state updates

This caused issues:
- ❌ UI had to parse narration lines looking for JSON
- ❌ CLI saw JSON garbage mixed with human messages
- ❌ No clear separation of concerns
- ❌ Fragile parsing logic (searching backwards for `[{`)

## Solution: Dual-Channel SSE

**Same SSE connection, two event types:**

```rust
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SseEvent {
    /// Human-readable narration message
    Narration(NarrationEvent),
    /// Structured data for UI consumption
    Data(DataEvent),
    /// Job completion marker
    Done,
}
```

## Architecture

### SSE Stream Format

**Before (mixed):**
```text
📋 Listing models on hive 'localhost'
Found 1 model(s)
id │ name │ size
───┼──────┼─────
... table ...
[{"id":"model1","name":"..."}]  ← JSON mixed with narration!
✅ Model list operation complete
[DONE]
```

**After (dual-channel):**
```json
{
  "type": "narration",
  "formatted": "📋 Listing models on hive 'localhost'"
}

{
  "type": "narration",
  "formatted": "Found 1 model(s)"
}

{
  "type": "narration",
  "formatted": "id │ name │ size\n───┼──────┼─────\n..."
}

{
  "type": "data",
  "action": "model_list",
  "payload": {
    "models": [
      {"id": "model1", "name": "Model 1", "size": 1024}
    ]
  }
}

{
  "type": "narration",
  "formatted": "✅ Model list operation complete"
}

{
  "type": "done"
}
```

### Backend API

**Emit narration (for CLI/UI logs):**
```rust
use observability_narration_core::n;

n!("model_list_start", "📋 Listing models...");
n!("model_list_table", table: &models_json);
```

**Emit data (for UI state):**
```rust
use observability_narration_core::sse_sink;

sse_sink::emit_data(
    &job_id,
    "model_list",
    serde_json::json!({"models": models_json})
);
```

### Frontend Consumption

**Narration events (for logs):**
```typescript
if (event.type === 'narration') {
  console.log(event.formatted);  // Display in log panel
}
```

**Data events (for state updates):**
```typescript
if (event.type === 'data') {
  if (event.action === 'model_list') {
    setModels(event.payload.models);  // Update React state
  }
}
```

## Implementation

### 1. New Types (narration-core/src/output/sse_sink.rs)

```rust
/// SSE event types - supports both narration and structured data
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SseEvent {
    Narration(NarrationEvent),
    Data(DataEvent),
    Done,
}

/// Structured data event for UI state updates
#[derive(Debug, Clone, serde::Serialize)]
pub struct DataEvent {
    /// Action this data relates to (e.g., "model_list", "worker_list")
    pub action: String,
    /// Structured JSON payload
    pub payload: serde_json::Value,
}
```

### 2. Updated Channel Types

```rust
pub struct SseChannelRegistry {
    /// TEAM-384: Changed to SseEvent to support both narration and data
    senders: Arc<Mutex<HashMap<String, mpsc::Sender<SseEvent>>>>,
    receivers: Arc<Mutex<HashMap<String, mpsc::Receiver<SseEvent>>>>,
}
```

### 3. Public API

```rust
/// Emit structured data to a job's SSE stream
pub fn emit_data(job_id: &str, action: impl Into<String>, payload: serde_json::Value) {
    let data = DataEvent {
        action: action.into(),
        payload,
    };
    SSE_CHANNEL_REGISTRY.send_data_to_job(job_id, data);
}
```

### 4. Backend Usage (job_router.rs)

```rust
// Emit table for CLI
n!("model_list_table", table: &models_json);

// Emit data for UI (separate channel)
observability_narration_core::sse_sink::emit_data(
    &job_id,
    "model_list",
    serde_json::json!({"models": models_json})
);
```

## Benefits

### ✅ Clean Separation

- **Narration** = Human-readable (CLI stdout, UI logs)
- **Data** = Machine-readable (UI state updates)
- No mixing, no confusion

### ✅ CLI Stays Clean

CLI only sees narration events:
```bash
$ ./rbee model ls
📋 Listing models on hive 'localhost'
Found 1 model(s)
id │ name │ size
───┼──────┼─────
...
✅ Model list operation complete
```

No JSON garbage!

### ✅ UI Gets Structured Data

UI receives typed events:
```typescript
interface DataEvent {
  type: 'data';
  action: string;
  payload: any;
}
```

No fragile JSON parsing from narration lines!

### ✅ Same Pipeline

Both channels flow through the same SSE connection:
- Same job isolation
- Same security model
- Same cleanup logic
- Same backpressure handling

### ✅ Backward Compatible

Existing narration code continues to work:
```rust
n!("action", "message");  // Still works!
```

Just add data emission where needed:
```rust
sse_sink::emit_data(&job_id, "action", json!({...}));
```

## Migration Pattern

**For any operation that needs UI state updates:**

```rust
// 1. Emit narration for CLI/logs
n!("operation_start", "Starting operation...");

// 2. Do the work
let results = do_work();

// 3. Emit table for CLI (optional)
if !results.is_empty() {
    let json_data = results.iter().map(|r| json!({...})).collect();
    n!("operation_table", table: &json_data);
}

// 4. Emit data for UI
sse_sink::emit_data(
    &job_id,
    "operation_name",
    json!({"results": results})
);

// 5. Emit completion
n!("operation_complete", "✅ Operation complete");
```

## Frontend Migration

**Old (fragile):**
```typescript
const lines: string[] = [];
await client.submitAndStream(op, (line: string) => {
  if (line !== '[DONE]') {
    lines.push(line);
  }
});

// Search backwards for JSON line
const jsonLine = lines.reverse().find(line => 
  line.trim().startsWith('[') || line.trim().startsWith('{')
);
return jsonLine ? JSON.parse(jsonLine) : [];
```

**New (robust):**
```typescript
let data: any = null;
await client.submitAndStream(op, (event: SseEvent) => {
  if (event.type === 'narration') {
    console.log(event.formatted);  // Show in logs
  } else if (event.type === 'data') {
    data = event.payload;  // Capture structured data
  }
});
return data;
```

## Files Changed

### narration-core (SSE infrastructure)

- `bin/99_shared_crates/narration-core/src/output/sse_sink.rs` (+80 LOC)
  - Added `SseEvent` enum
  - Added `DataEvent` struct
  - Updated channel types to `SseEvent`
  - Added `emit_data()` public API
  - Updated `send_to_job()` to wrap in `SseEvent::Narration`

- `bin/99_shared_crates/narration-core/src/output/mod.rs` (+1 LOC)
  - Export `SseEvent` and `DataEvent`

- `bin/99_shared_crates/narration-core/src/lib.rs` (+1 LOC)
  - Re-export new types

### rbee-hive (backend usage)

- `bin/20_rbee_hive/src/job_router.rs` (+6 LOC, -3 LOC)
  - Use `sse_sink::emit_data()` instead of narrating JSON
  - Keep table emission for CLI
  - Emit structured data for UI

## Next Steps

### Backend

Apply this pattern to other operations:
- ✅ `model_list` - DONE
- ⏳ `worker_list` - TODO
- ⏳ `hive_list` - TODO
- ⏳ `worker_ps` - TODO

### Frontend

Update UI to consume `SseEvent`:
1. Update `JobClient` to parse `SseEvent` enum
2. Update `useModels` hook to listen for `data` events
3. Remove fragile JSON parsing logic
4. Add type safety with TypeScript interfaces

### SSE Endpoint

Update SSE streaming endpoint to send proper event types:
```rust
while let Some(event) = rx.recv().await {
    match event {
        SseEvent::Narration(n) => {
            // Send as SSE event
            send_sse("narration", &serde_json::to_string(&n)?).await?;
        }
        SseEvent::Data(d) => {
            // Send as SSE event
            send_sse("data", &serde_json::to_string(&d)?).await?;
        }
        SseEvent::Done => {
            send_sse("done", "{}").await?;
            break;
        }
    }
}
```

## Lessons Learned

1. **Separate concerns early** - Don't mix human-readable and machine-readable data
2. **Use type systems** - Enums make intent clear and prevent mistakes
3. **Same pipeline, different channels** - Reuse infrastructure, add structure
4. **CLI and UI have different needs** - Design for both from the start
5. **Backward compatibility matters** - Existing code should keep working

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Backend (job_router.rs)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  n!("action", "message")  ──────────┐                      │
│                                      │                      │
│  sse_sink::emit_data(...)  ─────────┤                      │
│                                      ▼                      │
│                            ┌──────────────────┐            │
│                            │  SseEvent Enum   │            │
│                            ├──────────────────┤            │
│                            │ • Narration      │            │
│                            │ • Data           │            │
│                            │ • Done           │            │
│                            └────────┬─────────┘            │
│                                     │                      │
└─────────────────────────────────────┼──────────────────────┘
                                      │
                                      │ SSE Stream
                                      │
┌─────────────────────────────────────┼──────────────────────┐
│ Frontend (UI)                       │                      │
├─────────────────────────────────────┼──────────────────────┤
│                                     ▼                      │
│                          ┌────────────────────┐           │
│                          │  Event Handler     │           │
│                          └────────┬───────────┘           │
│                                   │                       │
│                    ┌──────────────┴──────────────┐       │
│                    │                             │       │
│                    ▼                             ▼       │
│          ┌──────────────────┐        ┌──────────────────┐│
│          │ Narration Event  │        │   Data Event     ││
│          ├──────────────────┤        ├──────────────────┤│
│          │ → Console logs   │        │ → State updates  ││
│          │ → UI log panel   │        │ → React setState ││
│          └──────────────────┘        └──────────────────┘│
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Summary

We've transformed the SSE system from a single-channel (narration only) to a dual-channel architecture (narration + data) while keeping the same underlying infrastructure. This provides:

- **Clean separation** between human-readable and machine-readable data
- **Better UX** for both CLI and UI users
- **Type safety** with Rust enums and TypeScript interfaces
- **Same security model** (job isolation, fail-fast)
- **Backward compatibility** (existing narration code works)

The UI can now receive structured data through a proper channel instead of parsing it from narration lines!
