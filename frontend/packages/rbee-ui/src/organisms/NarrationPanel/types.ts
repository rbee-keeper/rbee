// TEAM-384: Narration types - shared across rbee-keeper and rbee-hive
// Structured narration events from backend SSE streams

/**
 * Structured narration event
 *
 * This represents a single narration message from the backend,
 * parsed from either:
 * - Structured JSON (rbee-keeper via iframe-bridge)
 * - Raw SSE text (rbee-hive direct connection)
 */
export interface NarrationEvent {
  /** Log level: error, warn, info, debug */
  level: 'error' | 'warn' | 'info' | 'debug'

  /** Human-readable message (the actual content) */
  message: string

  /** ISO 8601 timestamp */
  timestamp: string

  /** Actor/module that generated the event (e.g., "lifecycle_local") */
  actor: string | null

  /** Action being performed (e.g., "rebuild_start") */
  action: string | null

  /** Context identifier (usually job_id) */
  context: string | null

  /** Human-readable message (alias for message) */
  human: string

  /** Full function name (e.g., "lifecycle_local::rebuild::rebuild_daemon") */
  fn_name: string | null

  /** Target module (optional) */
  target: string | null
}

/**
 * Narration entry with unique ID
 * Used in the store to track individual entries
 */
export interface NarrationEntry extends NarrationEvent {
  /** Unique ID for React keys */
  id: number
}

/**
 * Props for NarrationPanel component
 */
export interface NarrationPanelProps {
  /** Callback when panel is closed */
  onClose?: () => void

  /** Optional title override (default: "Narration") */
  title?: string

  /** Optional test callback (for testing narration pipeline) */
  onTest?: () => void

  /** Show test button (default: false) */
  showTestButton?: boolean
}
