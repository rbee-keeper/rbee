// Worker filter definitions - SSG-compatible filtering system

import type { WorkerCatalogEntry } from '../../../../../bin/80-global-worker-catalog/src/types'
import type { FilterConfig, FilterGroup } from '@/lib/filters/types'

// Worker filter state
export interface WorkerFilters {
  category: 'all' | 'llm' | 'image'
  backend: 'all' | 'cpu' | 'cuda' | 'metal' | 'rocm'
  platform: 'all' | 'linux' | 'macos' | 'windows'
}

// Filter group definitions
export const WORKER_FILTER_GROUPS: FilterGroup[] = [
  {
    id: 'category',
    label: 'Worker Category',
    options: [
      { label: 'All Workers', value: 'all' },
      { label: 'LLM Workers', value: 'llm' },
      { label: 'Image Workers', value: 'image' },
    ],
  },
  {
    id: 'backend',
    label: 'Backend Type',
    options: [
      { label: 'All Backends', value: 'all' },
      { label: 'CPU', value: 'cpu' },
      { label: 'CUDA (NVIDIA)', value: 'cuda' },
      { label: 'Metal (Apple)', value: 'metal' },
      { label: 'ROCm (AMD)', value: 'rocm' },
    ],
  },
  {
    id: 'platform',
    label: 'Platform',
    options: [
      { label: 'All Platforms', value: 'all' },
      { label: 'Linux', value: 'linux' },
      { label: 'macOS', value: 'macos' },
      { label: 'Windows', value: 'windows' },
    ],
  },
]

// Pre-generated filter combinations for SSG
export const PREGENERATED_WORKER_FILTERS: FilterConfig<WorkerFilters>[] = [
  // Default view
  { filters: { category: 'all', backend: 'all', platform: 'all' }, path: '' },

  // By category
  { filters: { category: 'llm', backend: 'all', platform: 'all' }, path: 'filter/llm' },
  { filters: { category: 'image', backend: 'all', platform: 'all' }, path: 'filter/image' },

  // By backend
  { filters: { category: 'all', backend: 'cpu', platform: 'all' }, path: 'filter/cpu' },
  { filters: { category: 'all', backend: 'cuda', platform: 'all' }, path: 'filter/cuda' },
  { filters: { category: 'all', backend: 'metal', platform: 'all' }, path: 'filter/metal' },
  { filters: { category: 'all', backend: 'rocm', platform: 'all' }, path: 'filter/rocm' },

  // By platform
  { filters: { category: 'all', backend: 'all', platform: 'linux' }, path: 'filter/linux' },
  { filters: { category: 'all', backend: 'all', platform: 'macos' }, path: 'filter/macos' },
  { filters: { category: 'all', backend: 'all', platform: 'windows' }, path: 'filter/windows' },

  // Popular combinations - LLM
  { filters: { category: 'llm', backend: 'cpu', platform: 'all' }, path: 'filter/llm/cpu' },
  { filters: { category: 'llm', backend: 'cuda', platform: 'linux' }, path: 'filter/llm/cuda/linux' },
  { filters: { category: 'llm', backend: 'metal', platform: 'macos' }, path: 'filter/llm/metal/macos' },
  { filters: { category: 'llm', backend: 'rocm', platform: 'linux' }, path: 'filter/llm/rocm/linux' },

  // Popular combinations - Image
  { filters: { category: 'image', backend: 'cpu', platform: 'all' }, path: 'filter/image/cpu' },
  { filters: { category: 'image', backend: 'cuda', platform: 'linux' }, path: 'filter/image/cuda/linux' },
  { filters: { category: 'image', backend: 'metal', platform: 'macos' }, path: 'filter/image/metal/macos' },
  { filters: { category: 'image', backend: 'rocm', platform: 'linux' }, path: 'filter/image/rocm/linux' },
]

/**
 * Build URL from worker filter configuration
 */
export function buildWorkerFilterUrl(filters: Partial<WorkerFilters>): string {
  const found = PREGENERATED_WORKER_FILTERS.find(
    (f) =>
      f.filters.category === (filters.category || 'all') &&
      f.filters.backend === (filters.backend || 'all') &&
      f.filters.platform === (filters.platform || 'all'),
  )

  return found?.path ? `/workers/${found.path}` : '/workers'
}

/**
 * Get filter configuration from URL path
 */
export function getWorkerFilterFromPath(path: string): WorkerFilters {
  const found = PREGENERATED_WORKER_FILTERS.find((f) => f.path === path)
  return found?.filters || { category: 'all', backend: 'all', platform: 'all' }
}

/**
 * Filter workers based on current filter state
 */
export function filterWorkers(workers: WorkerCatalogEntry[], filters: WorkerFilters): WorkerCatalogEntry[] {
  return workers.filter((worker) => {
    // Category filter (based on worker ID prefix)
    if (filters.category !== 'all') {
      const isLLM = worker.id.startsWith('llm-')
      const isImage = worker.id.startsWith('sd-')

      if (filters.category === 'llm' && !isLLM) return false
      if (filters.category === 'image' && !isImage) return false
    }

    // Backend filter (workerType)
    if (filters.backend !== 'all' && worker.workerType !== filters.backend) {
      return false
    }

    // Platform filter
    if (filters.platform !== 'all' && !worker.platforms.includes(filters.platform as 'linux' | 'macos' | 'windows')) {
      return false
    }

    return true
  })
}

/**
 * Build filter description for display
 */
export function buildFilterDescription(filters: WorkerFilters): string {
  const parts: string[] = []

  if (filters.category !== 'all') {
    parts.push(filters.category === 'llm' ? 'LLM' : 'Image')
  }
  if (filters.backend !== 'all') {
    parts.push(filters.backend.toUpperCase())
  }
  if (filters.platform !== 'all') {
    parts.push(filters.platform.charAt(0).toUpperCase() + filters.platform.slice(1))
  }

  return parts.length > 0 ? parts.join(' · ') : 'All Workers'
}
