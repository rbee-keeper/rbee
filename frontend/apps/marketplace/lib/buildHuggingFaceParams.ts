// TEAM-502: Canonical mapping from HFFilterSidebar state + GWC workers
// to HuggingFaceListModelsParams. This is the single source of truth for
// building HF queries from worker-driven filters.

import type { GWCWorker, HuggingFaceListModelsParams } from '@rbee/marketplace-core'
import type { HFFilterState } from '@rbee/ui/marketplace/organisms/HFFilterSidebar'

/**
 * Build HuggingFace list-models query parameters from sidebar filter state
 * and available GWC workers.
 */
export function buildHuggingFaceParamsFromFilters(
  filters: HFFilterState,
  searchQuery: string,
  workers: GWCWorker[],
  limit: number = 50,
): HuggingFaceListModelsParams {
  // TEAM_511: Map UI sort options to HuggingFace sort fields.
  // - `trending` is handled via the /models-json endpoint in the adapter.
  // - /api/models accepts: downloads, likes, lastModified, createdAt.
  const sortMap: Record<HFFilterState['sort'], HuggingFaceListModelsParams['sort']> = {
    trending: 'trending',
    downloads: 'downloads',
    likes: 'likes',
    updated: 'lastModified',
    created: 'createdAt',
  }

  const params: HuggingFaceListModelsParams = {
    limit,
    sort: sortMap[filters.sort],
    direction: filters.direction,
  }

  if (searchQuery) {
    params.search = searchQuery
  }

  const selectedWorkers = workers.filter((w) => filters.workers.includes(w.id))

  // Aggregate compatibility from selected workers
  const compatTasks = new Set<string>()
  const compatLibraries = new Set<string>()
  const compatFormats = new Set<string>()

  selectedWorkers.forEach((worker) => {
    const compat = worker.marketplaceCompatibility?.huggingface
    if (!compat) return

    compat.tasks.forEach((t) => void compatTasks.add(t))
    compat.libraries.forEach((l) => void compatLibraries.add(l))
    compat.formats.forEach((f) => void compatFormats.add(f))
  })

  const effectiveTasks =
    filters.tasks.length > 0 ? filters.tasks : selectedWorkers.length > 0 ? Array.from(compatTasks) : []

  const effectiveLibraries =
    filters.libraries.length > 0 ? filters.libraries : selectedWorkers.length > 0 ? Array.from(compatLibraries) : []

  const effectiveFormats =
    filters.formats.length > 0 ? filters.formats : selectedWorkers.length > 0 ? Array.from(compatFormats) : []

  if (effectiveTasks.length > 0) {
    // HF supports a single pipeline_tag; choose the first effective task.
    params.pipeline_tag = effectiveTasks[0] as HuggingFaceListModelsParams['pipeline_tag']
  }

  if (effectiveLibraries.length > 0) {
    // HF supports a single library; choose the first effective library.
    params.library = effectiveLibraries[0] as HuggingFaceListModelsParams['library']
  }

  if (effectiveFormats.length > 0) {
    // HF filter supports comma-separated tags for OR semantics.
    params.filter = effectiveFormats.join(',')
  }

  // Language and license are driven purely by user filter state for now.
  if (filters.languages && filters.languages.length > 0) {
    // HF language param accepts a single string; we use the first selection.
    params.language = filters.languages[0]
  }

  if (filters.licenses && filters.licenses.length > 0) {
    // HF license param accepts a single license; we use the first selection.
    params.license = filters.licenses[0]
  }

  return params
}
