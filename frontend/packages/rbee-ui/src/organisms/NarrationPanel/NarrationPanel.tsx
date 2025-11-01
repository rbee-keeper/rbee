// TEAM-384: Narration panel - displays real-time narration events from Rust backend
// Shared component for rbee-keeper and rbee-hive
// Listens to narration events and displays them with grouping and formatting

'use client'

import { ScrollArea } from '../../atoms/ScrollArea'
import { X } from 'lucide-react'
import { useNarrationStore } from './useNarrationStore'
import type { NarrationPanelProps } from './types'

/**
 * NarrationPanel - Real-time narration event display
 * 
 * Features:
 * - Newest-first ordering (shell-like reading)
 * - Function name grouping with timestamps
 * - Level badges (error/warn/info/debug)
 * - Persistence (last 100 entries)
 * - Clear and test buttons
 * 
 * @example
 * ```tsx
 * <NarrationPanel
 *   onClose={() => setShowPanel(false)}
 *   showTestButton={true}
 *   onTest={async () => {
 *     // Test narration pipeline
 *     await invoke('test_narration')
 *   }}
 * />
 * ```
 */
export function NarrationPanel({ 
  onClose, 
  title = 'Narration',
  onTest,
  showTestButton = false 
}: NarrationPanelProps) {
  // Get entries from Zustand store (persisted even when panel closed)
  const entries = useNarrationStore((state) => state.entries)
  const clearEntries = useNarrationStore((state) => state.clearEntries)

  // Clear all entries
  const handleClear = () => {
    clearEntries()
  }

  // Format timestamp to HH:MM:SS
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  // Get level badge style
  const getLevelBadge = (level: 'error' | 'warn' | 'info' | 'debug') => {
    const baseClasses = 'px-1.5 py-0.5 rounded text-xs font-mono font-semibold'
    switch (level) {
      case 'error':
        return `${baseClasses} bg-red-500/10 text-red-500`
      case 'warn':
        return `${baseClasses} bg-yellow-500/10 text-yellow-500`
      case 'info':
        return `${baseClasses} bg-blue-500/10 text-blue-500`
      case 'debug':
        return `${baseClasses} bg-gray-500/10 text-gray-500`
    }
  }

  return (
    <div className="w-full h-full border-l border-border bg-background flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-background">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-muted transition-colors text-foreground"
            aria-label="Close narration panel"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Entries list */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-2 overflow-x-hidden">
            {entries.length === 0 ? (
              <div className="text-center text-sm text-muted-foreground py-8">
                Waiting for events...
              </div>
            ) : (
              entries.map((entry, index) => {
                // Check if fn_name changed from previous entry
                const prevEntry = index > 0 ? entries[index - 1] : null
                const fnNameChanged = !prevEntry || prevEntry.fn_name !== entry.fn_name

                return (
                  <div key={entry.id}>
                    {/* Show fn_name title with timestamp when it changes */}
                    {fnNameChanged && entry.fn_name && (
                      <div className="px-3 py-1.5 text-xs font-medium bg-muted border-l-4 border-blue-500 mb-1 overflow-hidden">
                        <div className="space-y-1">
                          <div className="text-muted-foreground font-mono text-[10px]">
                            {formatTime(entry.timestamp)}
                          </div>
                          <div className="text-foreground break-words overflow-wrap-anywhere">
                            {entry.fn_name}
                          </div>
                        </div>
                      </div>
                    )}

                    <div className="p-2 rounded-md bg-muted/30 hover:bg-muted/50 transition-colors text-xs space-y-1 overflow-hidden mb-1">
                      {/* Action and level */}
                      <div className="flex items-center justify-between gap-2 min-w-0">
                        <span className="text-muted-foreground font-mono truncate">
                          {entry.action || '—'}
                        </span>
                        <span className={`${getLevelBadge(entry.level)} shrink-0`}>
                          {entry.level.toUpperCase()}
                        </span>
                      </div>

                      {/* Message */}
                      <div className="text-foreground break-words overflow-wrap-anywhere font-mono leading-relaxed">
                        {entry.message}
                      </div>
                    </div>
                  </div>
                )
              })
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Footer stats */}
      <div className="flex items-center justify-between px-4 py-2 border-t border-border text-xs text-muted-foreground">
        <span>
          {entries.length} {entries.length === 1 ? 'entry' : 'entries'}
        </span>
        <div className="flex gap-2">
          {showTestButton && onTest && (
            <button
              onClick={onTest}
              className="text-xs text-blue-500 hover:text-blue-600 transition-colors"
              title="Test narration events"
            >
              Test
            </button>
          )}
          <button
            onClick={handleClear}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            title="Clear all entries"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  )
}
