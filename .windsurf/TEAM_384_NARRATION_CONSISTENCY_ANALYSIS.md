# TEAM-384: Narration Consistency Analysis

**Status:** 📋 ANALYSIS COMPLETE  
**Date:** Nov 1, 2025

## Problem Statement

Two different narration implementations exist:

1. **rbee-keeper** (mature): Structured narration with Zustand store, proper parsing, fn_name grouping
2. **rbee-hive** (basic): Raw string array, no parsing, no structure

**User Request:** Make rbee-hive's worker installation narration consistent with rbee-keeper's mature implementation.

---

## Current Implementations

### 1. rbee-keeper (MATURE) ✅

**Architecture:**
```
Backend (Rust) → SSE Stream → iframe-bridge → narrationListener.ts → Zustand Store → NarrationPanel.tsx
```

**Key Files:**
- `bin/00_rbee_keeper/ui/src/utils/narrationListener.ts` - Parses structured events
- `bin/00_rbee_keeper/ui/src/store/narrationStore.ts` - Zustand store with persistence
- `bin/00_rbee_keeper/ui/src/components/NarrationPanel.tsx` - Rich UI with grouping

**Data Structure:**
```typescript
interface NarrationEvent {
  level: string           // "info", "warn", "error", "debug"
  message: string         // Human-readable message
  timestamp: string       // ISO 8601
  actor: string | null    // Who performed the action
  action: string | null   // What action was performed
  context: string | null  // job_id for context
  human: string          // Same as message
  fn_name: string | null // Function name (extracted from ANSI codes)
  target: string | null  // Target module
}
```

**Features:**
- ✅ **Structured parsing** - Extracts fn_name from ANSI escape codes
- ✅ **Level badges** - Color-coded (error=red, warn=yellow, info=blue, debug=gray)
- ✅ **Function grouping** - Groups messages by fn_name with timestamp headers
- ✅ **Persistence** - Keeps last 100 entries in localStorage
- ✅ **Newest first** - Prepends new messages to top (shell-like)
- ✅ **Timestamps** - HH:MM:SS format
- ✅ **Clear button** - Reset all entries
- ✅ **Test button** - Verify narration pipeline

**UI Pattern:**
```
┌─────────────────────────────────────────────┐
│ Narration                              [X]  │
├─────────────────────────────────────────────┤
│ 23:44:15                                    │
│ lifecycle_local::rebuild::rebuild_daemon    │ ← fn_name header
│                                             │
│ rebuild_start         INFO                  │ ← action + level
│ 🔄 Rebuilding rbee-hive locally             │ ← message
│                                             │
│ rebuild_build         INFO                  │
│ 🔨 Building rbee-hive locally               │
│                                             │
│ ...                                         │
├─────────────────────────────────────────────┤
│ 42 entries          [Test] [Clear]         │
└─────────────────────────────────────────────┘
```

---

### 2. rbee-hive (BASIC) ❌

**Architecture:**
```
Backend (Rust) → SSE Stream → HiveClient.submitAndStream() → useState<string[]> → map() in JSX
```

**Key Files:**
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useWorkerOperations.ts` - Raw string collection
- `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx` - Basic display

**Data Structure:**
```typescript
const [progressMessages, setProgressMessages] = useState<string[]>([])
// Just raw strings: ["🔄 Starting...", "✅ Complete"]
```

**Features:**
- ❌ **No parsing** - Just raw strings from SSE
- ❌ **No structure** - No level, timestamp, fn_name extraction
- ❌ **No grouping** - All messages in flat list
- ❌ **No persistence** - Lost on component unmount
- ❌ **Basic styling** - Just colored background based on state
- ❌ **No timestamps** - Can't tell when things happened
- ❌ **No clear button** - Can't reset (only via onResetInstall prop)

**UI Pattern:**
```
┌─────────────────────────────────────────────┐
│ Installing Worker...                   [X]  │
├─────────────────────────────────────────────┤
│ 🔄 Starting...                              │ ← Just raw strings
│ 🔨 Building...                              │
│ ✅ Complete                                 │
└─────────────────────────────────────────────┘
```

---

## Key Differences

| Feature | rbee-keeper | rbee-hive |
|---------|-------------|-----------|
| **Data Structure** | Structured NarrationEvent | Raw string[] |
| **Parsing** | Extracts fn_name, level, action | None |
| **Grouping** | By fn_name with headers | Flat list |
| **Timestamps** | HH:MM:SS format | None |
| **Level Badges** | Color-coded (4 levels) | None |
| **Persistence** | Zustand + localStorage | Component state only |
| **Order** | Newest first (prepend) | Oldest first (append) |
| **Clear** | Dedicated button | Via parent prop |
| **Test** | Built-in test button | None |
| **Store** | Centralized Zustand | Local useState |

---

## Root Cause Analysis

### Why rbee-hive is Basic

1. **Different SSE format**: rbee-hive receives raw strings, rbee-keeper receives structured JSON
2. **No shared narration client**: Each implements its own parsing
3. **Different use case**: rbee-hive shows progress for ONE operation, rbee-keeper shows ALL operations
4. **Time pressure**: rbee-hive was built quickly, rbee-keeper evolved over time

### SSE Format Comparison

**rbee-keeper receives:**
```json
{
  "type": "NARRATION_EVENT",
  "payload": {
    "level": "info",
    "human": "🔄 Rebuilding rbee-hive locally",
    "timestamp": "2025-11-01T23:44:15Z",
    "actor": "lifecycle_local",
    "action": "rebuild_start",
    "job_id": "job-123",
    "formatted": "\x1b[1mlifecycle_local::rebuild::rebuild_daemon\x1b[0m \x1b[2mrebuild_start\x1b[0m\n🔄 Rebuilding rbee-hive locally"
  }
}
```

**rbee-hive receives:**
```
lifecycle_local::rebuild::rebuild_daemon rebuild_start
🔄 Rebuilding rbee-hive locally
```

**Problem:** rbee-hive gets the `formatted` field as raw text, not the structured JSON!

---

## Solution Options

### Option 1: Shared Narration Store (RECOMMENDED) ✅

**Create:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/stores/narrationStore.ts`

**Copy from rbee-keeper:**
- Zustand store with immer middleware
- NarrationEvent interface
- addEntry/clearEntries actions
- Persistence (last 100 entries)

**Create:** `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/utils/narrationParser.ts`

**Parse SSE lines:**
```typescript
// Input: "lifecycle_local::rebuild::rebuild_daemon rebuild_start\n🔄 Rebuilding..."
// Output: { fn_name: "lifecycle_local::rebuild::rebuild_daemon", action: "rebuild_start", message: "🔄 Rebuilding..." }
```

**Update:** `useWorkerOperations.ts`
```typescript
// OLD:
setProgressMessages(prev => [...prev, line])

// NEW:
const parsed = parseNarrationLine(line)
useNarrationStore.getState().addEntry(parsed)
```

**Create:** `bin/20_rbee_hive/ui/app/src/components/NarrationPanel.tsx`

**Copy from rbee-keeper:**
- Same UI layout
- Same grouping logic
- Same level badges
- Same timestamp formatting

**Benefits:**
- ✅ Consistent UX across rbee-keeper and rbee-hive
- ✅ Reusable narration panel component
- ✅ Proper structure and parsing
- ✅ Persistence and history

**Effort:** 4-6 hours

---

### Option 2: Backend JSON Output (ALTERNATIVE)

**Change backend to output JSON:**
```rust
// OLD:
n!("rebuild_start", "🔄 Rebuilding {} locally", daemon_name);

// NEW:
let event = NarrationEvent {
    level: "info",
    human: format!("🔄 Rebuilding {} locally", daemon_name),
    action: Some("rebuild_start"),
    fn_name: Some("lifecycle_local::rebuild::rebuild_daemon"),
    timestamp: Some(Utc::now()),
    ...
};
println!("{}", serde_json::to_string(&event)?);
```

**Benefits:**
- ✅ No parsing needed in frontend
- ✅ Structured from the start

**Drawbacks:**
- ❌ Breaks existing CLI output (not human-readable)
- ❌ Requires changes to narration-core crate
- ❌ Affects ALL services (queen, hive, worker)

**Effort:** 8-12 hours + testing

---

### Option 3: Dual Output Format (COMPLEX)

**Add JSON mode to narration-core:**
```rust
// CLI mode: human-readable
NARRATE.set_output_format(OutputFormat::Human);

// API mode: JSON
NARRATE.set_output_format(OutputFormat::Json);
```

**Benefits:**
- ✅ Best of both worlds
- ✅ CLI stays human-readable
- ✅ API gets structured data

**Drawbacks:**
- ❌ Most complex solution
- ❌ Requires narration-core refactor
- ❌ Need to detect context (CLI vs API)

**Effort:** 12-16 hours

---

## Recommendation: Option 1 (Shared Narration Store)

**Why:**
1. **Fastest** - No backend changes, pure frontend work
2. **Safest** - Doesn't affect CLI or other services
3. **Reusable** - Can be used for model operations too
4. **Proven** - rbee-keeper already validates the pattern

**Implementation Plan:**

### Phase 1: Parser (2 hours)
1. Create `narrationParser.ts` in rbee-hive-react
2. Parse SSE lines into structured events
3. Extract fn_name, action, message, emoji
4. Add unit tests

### Phase 2: Store (1 hour)
1. Copy narrationStore.ts from rbee-keeper
2. Adapt for rbee-hive (same interface)
3. Add to rbee-hive-react package

### Phase 3: Panel Component (2 hours)
1. Copy NarrationPanel.tsx from rbee-keeper
2. Adapt styling for rbee-hive theme
3. Add to rbee-hive app

### Phase 4: Integration (1 hour)
1. Update useWorkerOperations to use store
2. Replace WorkerCatalogView progress display
3. Add NarrationPanel to layout

### Phase 5: Testing (1 hour)
1. Test worker installation flow
2. Verify grouping and timestamps
3. Test persistence across page reloads

**Total Effort:** 6-7 hours

---

## Parser Implementation Details

### Input Format Analysis

**SSE Line Format:**
```
<fn_name> <action>
<message>
```

**Example:**
```
lifecycle_local::rebuild::rebuild_daemon rebuild_start
🔄 Rebuilding rbee-hive locally
```

**Parser Logic:**
```typescript
function parseNarrationLine(line: string): NarrationEvent {
  const lines = line.split('\n')
  
  if (lines.length < 2) {
    // Malformed line, return as-is
    return {
      level: 'info',
      message: line,
      timestamp: new Date().toISOString(),
      action: null,
      fn_name: null,
      ...
    }
  }
  
  const [header, ...messageLines] = lines
  const [fn_name, action] = header.split(' ')
  const message = messageLines.join('\n')
  
  // Extract level from emoji or action
  const level = detectLevel(message, action)
  
  return {
    level,
    message,
    timestamp: new Date().toISOString(),
    action,
    fn_name,
    actor: fn_name?.split('::')[0] || null,
    context: null, // TODO: Extract job_id if present
    human: message,
    target: null,
  }
}

function detectLevel(message: string, action: string | null): string {
  // Error indicators
  if (message.includes('❌') || message.includes('Error') || action?.includes('error')) {
    return 'error'
  }
  
  // Warning indicators
  if (message.includes('⚠️') || message.includes('Warning') || action?.includes('warn')) {
    return 'warn'
  }
  
  // Debug indicators
  if (message.includes('🔍') || action?.includes('debug')) {
    return 'debug'
  }
  
  // Default to info
  return 'info'
}
```

---

## Next Steps

1. **Create parser** - `narrationParser.ts` with tests
2. **Copy store** - `narrationStore.ts` from rbee-keeper
3. **Copy panel** - `NarrationPanel.tsx` from rbee-keeper
4. **Update hook** - `useWorkerOperations.ts` to use store
5. **Update view** - `WorkerCatalogView.tsx` to use NarrationPanel
6. **Test** - Full worker installation flow

---

## Files to Create/Modify

### Create:
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/utils/narrationParser.ts` (150 LOC)
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/stores/narrationStore.ts` (70 LOC)
- `bin/20_rbee_hive/ui/app/src/components/NarrationPanel.tsx` (180 LOC)

### Modify:
- `bin/20_rbee_hive/ui/packages/rbee-hive-react/src/hooks/useWorkerOperations.ts` (-10 LOC, +5 LOC)
- `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/WorkerCatalogView.tsx` (-80 LOC, +10 LOC)
- `bin/20_rbee_hive/ui/app/src/components/WorkerManagement/index.tsx` (+5 LOC)

**Total:** ~400 LOC added, ~90 LOC removed

---

## Expected Outcome

**Before:**
```
Installing Worker...
🔄 Starting...
🔨 Building...
✅ Complete
```

**After:**
```
┌─────────────────────────────────────────────┐
│ Narration                              [X]  │
├─────────────────────────────────────────────┤
│ 23:44:15                                    │
│ lifecycle_local::rebuild::rebuild_daemon    │
│                                             │
│ rebuild_start         INFO                  │
│ 🔄 Rebuilding rbee-hive locally             │
│                                             │
│ rebuild_build         INFO                  │
│ 🔨 Building rbee-hive locally               │
│                                             │
│ 23:44:20                                    │
│ lifecycle_shared::build::build_daemon       │
│                                             │
│ build_start           INFO                  │
│ 🔨 Building rbee-hive from source...        │
│                                             │
│ ...                                         │
├─────────────────────────────────────────────┤
│ 42 entries          [Test] [Clear]         │
└─────────────────────────────────────────────┘
```

---

**TEAM-384 Signature:** Analysis complete, ready for implementation.
