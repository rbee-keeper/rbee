// TEAM-413: Download tracking store
// Manages active downloads for models and workers
// Wired into narration system for progress updates

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Download } from '../components/DownloadPanel'

interface DownloadStore {
  downloads: Download[]

  // Core actions
  startDownload: (jobId: string, name: string, type: 'model' | 'worker') => void
  updateDownload: (id: string, update: Partial<Download>) => void
  completeDownload: (id: string) => void
  failDownload: (id: string, error: string) => void
  removeDownload: (id: string) => void
  cancelDownload: (id: string) => void
  retryDownload: (id: string) => void
  clearCompleted: () => void

  // Narration integration
  updateFromNarration: (jobId: string, message: string) => void
}

// Helper to parse download progress from narration messages
function parseDownloadProgress(message: string): Partial<Download> | null {
  // Match: "📊 Progress: 45.2% (2.5 MB / 5.5 MB)" or similar
  const progressMatch = message.match(/(\d+\.?\d*)%/)
  const sizeMatch = message.match(/([\d.]+)\s*(B|KB|MB|GB|TB)\s*\/\s*([\d.]+)\s*(B|KB|MB|GB|TB)/i)

  if (progressMatch || sizeMatch) {
    const update: Partial<Download> = {}

    if (progressMatch && progressMatch[1]) {
      update.percentage = parseFloat(progressMatch[1])
    }

    if (sizeMatch && sizeMatch[1] && sizeMatch[2] && sizeMatch[3] && sizeMatch[4]) {
      const downloaded = parseFloat(sizeMatch[1])
      const downloadedUnit = sizeMatch[2].toUpperCase()
      const total = parseFloat(sizeMatch[3])
      const totalUnit = sizeMatch[4].toUpperCase()

      // Convert to bytes
      const units: Record<string, number> = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }
      update.bytesDownloaded = downloaded * (units[downloadedUnit] || 1)
      update.totalSize = total * (units[totalUnit] || 1)
    }

    return update
  }

  return null
}

export const useDownloadStore = create<DownloadStore>()(
  persist(
    (set) => ({
      downloads: [],

      startDownload: (jobId, name, type) =>
        set((state) => ({
          downloads: [
            ...state.downloads,
            {
              id: jobId,
              name,
              type,
              status: 'downloading' as const,
              bytesDownloaded: 0,
              totalSize: null,
              percentage: null,
              speed: null,
              eta: null,
            },
          ],
        })),

      updateDownload: (id, update) =>
        set((state) => ({
          downloads: state.downloads.map((d) => (d.id === id ? { ...d, ...update } : d)),
        })),

      completeDownload: (id) =>
        set((state) => ({
          downloads: state.downloads.map((d) =>
            d.id === id ? { ...d, status: 'complete' as const, percentage: 100 } : d,
          ),
        })),

      failDownload: (id, error) =>
        set((state) => ({
          downloads: state.downloads.map((d) => (d.id === id ? { ...d, status: 'failed' as const, error } : d)),
        })),

      removeDownload: (id) =>
        set((state) => ({
          downloads: state.downloads.filter((d) => d.id !== id),
        })),

      cancelDownload: (id) =>
        set((state) => ({
          downloads: state.downloads.map((d) => (d.id === id ? { ...d, status: 'cancelled' as const } : d)),
        })),

      retryDownload: (id) =>
        set((state) => ({
          downloads: state.downloads.map((d) =>
            d.id === id
              ? { ...d, status: 'downloading' as const, bytesDownloaded: 0, percentage: null }
              : d,
          ),
        })),

      clearCompleted: () =>
        set((state) => ({
          downloads: state.downloads.filter((d) => d.status === 'downloading'),
        })),

      // TEAM-413: Parse narration messages and update download progress
      updateFromNarration: (jobId, message) =>
        set((state) => {
          const download = state.downloads.find((d) => d.id === jobId)
          if (!download) return state

          // Check for completion
          if (message.includes('✅') || message.includes('Complete') || message.includes('[DONE]')) {
            return {
              downloads: state.downloads.map((d) =>
                d.id === jobId ? { ...d, status: 'complete' as const, percentage: 100 } : d,
              ),
            }
          }

          // Check for failure
          if (message.includes('❌') || message.includes('Failed') || message.includes('Error')) {
            return {
              downloads: state.downloads.map((d) =>
                d.id === jobId ? { ...d, status: 'failed' as const, error: message } : d,
              ),
            }
          }

          // Parse progress
          const progress = parseDownloadProgress(message)
          if (progress) {
            return {
              downloads: state.downloads.map((d) => (d.id === jobId ? { ...d, ...progress } : d)),
            }
          }

          return state
        }),
    }),
    {
      name: 'download-store',
      partialize: (state) => ({
        downloads: state.downloads.filter((d) => d.status !== 'complete').slice(0, 10),
      }),
    },
  ),
)
