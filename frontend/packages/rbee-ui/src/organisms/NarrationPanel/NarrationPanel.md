# NarrationPanel

**TEAM-384**: Shared narration panel component for displaying real-time backend events.

## Overview

`NarrationPanel` is a reusable organism that displays structured narration events from Rust backend services. It's used by both `rbee-keeper` and `rbee-hive` to show real-time operation progress.

## Features

- ✅ **Structured Events** - Parses both JSON and raw text SSE formats
- ✅ **Function Grouping** - Groups messages by function name with timestamps
- ✅ **Level Badges** - Color-coded badges (error/warn/info/debug)
- ✅ **Persistence** - Keeps last 100 entries in localStorage
- ✅ **Newest First** - Shell-like reading order (newest at top)
- ✅ **Clear Button** - Reset all entries
- ✅ **Test Button** - Optional test narration pipeline

## Usage

### Basic Usage

```tsx
import { NarrationPanel, useNarrationStore, parseNarrationLine } from '@rbee/ui/organisms'

function MyComponent() {
  const addEntry = useNarrationStore((state) => state.addEntry)
  
  // Listen to SSE stream
  useEffect(() => {
    const eventSource = new EventSource('/api/events')
    eventSource.onmessage = (event) => {
      const parsed = parseNarrationLine(event.data)
      addEntry(parsed)
    }
    return () => eventSource.close()
  }, [addEntry])
  
  return <NarrationPanel onClose={() => setShowPanel(false)} />
}
```

### With Test Button

```tsx
<NarrationPanel
  onClose={() => setShowPanel(false)}
  showTestButton={true}
  onTest={async () => {
    // Test narration pipeline
    await invoke('test_narration')
  }}
/>
```

### Custom Title

```tsx
<NarrationPanel
  title="Build Progress"
  onClose={() => setShowPanel(false)}
/>
```

## API

### NarrationPanel Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `onClose` | `() => void` | - | Callback when panel is closed |
| `title` | `string` | `"Narration"` | Panel title |
| `showTestButton` | `boolean` | `false` | Show test button |
| `onTest` | `() => void` | - | Test button callback |

### useNarrationStore

Zustand store for managing narration entries.

```tsx
const entries = useNarrationStore((state) => state.entries)
const addEntry = useNarrationStore((state) => state.addEntry)
const clearEntries = useNarrationStore((state) => state.clearEntries)
const showNarration = useNarrationStore((state) => state.showNarration)
const setShowNarration = useNarrationStore((state) => state.setShowNarration)
```

### parseNarrationLine

Parse raw SSE line into structured `NarrationEvent`.

```tsx
const event = parseNarrationLine(line)
// Returns: { level, message, timestamp, action, fn_name, ... }
```

Handles two formats:

1. **Raw text** (rbee-hive):
   ```
   lifecycle_local::rebuild::rebuild_daemon rebuild_start
   🔄 Rebuilding rbee-hive locally
   ```

2. **JSON** (rbee-keeper):
   ```json
   {
     "level": "info",
     "human": "🔄 Rebuilding rbee-hive locally",
     "action": "rebuild_start",
     "fn_name": "lifecycle_local::rebuild::rebuild_daemon"
   }
   ```

## Types

### NarrationEvent

```typescript
interface NarrationEvent {
  level: 'error' | 'warn' | 'info' | 'debug'
  message: string
  timestamp: string // ISO 8601
  actor: string | null
  action: string | null
  context: string | null // Usually job_id
  human: string
  fn_name: string | null
  target: string | null
}
```

### NarrationEntry

```typescript
interface NarrationEntry extends NarrationEvent {
  id: number // Unique ID for React keys
}
```

## Integration Examples

### rbee-keeper (Tauri + iframe-bridge)

```tsx
import { setupNarrationListener } from './utils/narrationListener'
import { NarrationPanel } from '@rbee/ui/organisms'

function App() {
  useEffect(() => {
    const cleanup = setupNarrationListener()
    return cleanup
  }, [])
  
  return <NarrationPanel onClose={() => {}} />
}
```

### rbee-hive (Direct SSE)

```tsx
import { useNarrationStore, parseNarrationLine, NarrationPanel } from '@rbee/ui/organisms'

function WorkerInstall() {
  const addEntry = useNarrationStore((state) => state.addEntry)
  
  const installWorker = async (workerId: string) => {
    await client.submitAndStream(op, (line: string) => {
      if (line !== '[DONE]') {
        const parsed = parseNarrationLine(line)
        addEntry(parsed)
      }
    })
  }
  
  return <NarrationPanel />
}
```

## Styling

Uses Tailwind CSS with shadcn/ui design tokens:

- `bg-background` - Panel background
- `border-border` - Borders
- `text-foreground` - Text
- `text-muted-foreground` - Secondary text
- `bg-muted` - Muted backgrounds
- Level colors: `text-red-500`, `text-yellow-500`, `text-blue-500`, `text-gray-500`

## Persistence

Entries are persisted to localStorage under key `rbee-narration-store`:

- Last 100 entries saved
- Panel visibility state saved
- Survives page reloads

## Level Detection

Automatically detects log level from message content:

- **Error**: ❌ emoji, "error", "failed" keywords
- **Warn**: ⚠️ emoji, "warning", "warn" keywords
- **Debug**: 🔍 emoji, "debug" keyword
- **Info**: Everything else (default)

## Function Grouping

Messages are grouped by `fn_name` with timestamp headers:

```
23:44:15
lifecycle_local::rebuild::rebuild_daemon

rebuild_start         INFO
🔄 Rebuilding rbee-hive locally

rebuild_build         INFO
🔨 Building rbee-hive locally
```

## Dependencies

- `zustand` - State management
- `zustand/middleware` - Persistence and immer
- `lucide-react` - Icons
- `@rbee/ui/atoms/ScrollArea` - Scrollable container

## Architecture

```
Backend (Rust)
    ↓ SSE Stream
Parser (parseNarrationLine)
    ↓ NarrationEvent
Store (useNarrationStore)
    ↓ NarrationEntry[]
Component (NarrationPanel)
    ↓ UI
```

## TEAM-384 Signature

All files in this directory are created by TEAM-384 for narration consistency across rbee-keeper and rbee-hive.
