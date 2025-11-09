// TEAM-384: Narration store - Zustand store for narration events
// Persists narration events even when panel is closed
// Shared across rbee-keeper and rbee-hive

'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import type { NarrationEntry, NarrationEvent } from './types'

interface NarrationState {
  entries: NarrationEntry[]
  idCounter: number
  showNarration: boolean

  // Actions
  addEntry: (event: NarrationEvent) => void
  clearEntries: () => void
  setShowNarration: (show: boolean) => void
}

/**
 * Zustand store for narration events
 *
 * Features:
 * - Persists last 100 entries to localStorage
 * - Prepends new entries (newest first)
 * - Tracks panel visibility state
 * - Uses immer for immutable updates
 *
 * @example
 * ```tsx
 * const entries = useNarrationStore((state) => state.entries)
 * const addEntry = useNarrationStore((state) => state.addEntry)
 *
 * // Add new entry
 * addEntry({
 *   level: 'info',
 *   message: '🔄 Starting...',
 *   timestamp: new Date().toISOString(),
 *   ...
 * })
 * ```
 */
export const useNarrationStore = create<NarrationState>()(
  persist(
    immer((set) => ({
      entries: [],
      idCounter: 0,
      showNarration: true,

      addEntry: (event: NarrationEvent) => {
        set((state) => {
          // Prepend new entry to top (newest first)
          state.entries.unshift({
            ...event,
            id: state.idCounter++,
          })
        })
      },

      clearEntries: () => {
        set((state) => {
          state.entries = []
          state.idCounter = 0
        })
      },

      setShowNarration: (show: boolean) => {
        set((state) => {
          state.showNarration = show
        })
      },
    })),
    {
      name: 'rbee-narration-store',
      partialize: (state) => ({
        entries: state.entries.slice(0, 100), // Keep last 100 entries
        idCounter: state.idCounter,
        showNarration: state.showNarration, // Persist panel state
      }),
    },
  ),
)
