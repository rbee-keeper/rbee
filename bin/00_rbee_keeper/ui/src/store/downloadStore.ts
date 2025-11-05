// TEAM-413: Download tracking store
// Manages active downloads for models and workers

import { create } from 'zustand'
import type { Download } from '../components/DownloadPanel'

interface DownloadStore {
  downloads: Download[]
  addDownload: (download: Omit<Download, 'status' | 'bytesDownloaded' | 'percentage' | 'speed' | 'eta'>) => void
  updateDownload: (id: string, update: Partial<Download>) => void
  removeDownload: (id: string) => void
  cancelDownload: (id: string) => void
  retryDownload: (id: string) => void
  clearCompleted: () => void
}

export const useDownloadStore = create<DownloadStore>((set) => ({
  downloads: [],

  addDownload: (download) => set((state) => ({
    downloads: [
      ...state.downloads,
      {
        ...download,
        status: 'downloading' as const,
        bytesDownloaded: 0,
        percentage: null,
        speed: null,
        eta: null,
      },
    ],
  })),

  updateDownload: (id, update) => set((state) => ({
    downloads: state.downloads.map((d) =>
      d.id === id ? { ...d, ...update } : d
    ),
  })),

  removeDownload: (id) => set((state) => ({
    downloads: state.downloads.filter((d) => d.id !== id),
  })),

  cancelDownload: (id) => set((state) => ({
    downloads: state.downloads.map((d) =>
      d.id === id ? { ...d, status: 'cancelled' as const } : d
    ),
  })),

  retryDownload: (id) => set((state) => ({
    downloads: state.downloads.map((d) =>
      d.id === id
        ? { ...d, status: 'downloading' as const, bytesDownloaded: 0, percentage: null, error: undefined }
        : d
    ),
  })),

  clearCompleted: () => set((state) => ({
    downloads: state.downloads.filter((d) => d.status === 'downloading'),
  })),
}))
