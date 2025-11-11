// TEAM-413: Workers list page with SSG
// TEAM-461: Added filtering system using CategoryFilterBar

import { WorkerCard } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import Link from 'next/link'
import { WORKERS } from '../../../../../bin/80-global-worker-catalog/src/data'
import { buildFilterDescription, filterWorkers, PREGENERATED_WORKER_FILTERS, WORKER_FILTER_GROUPS } from './filters'
import { WorkersFilterBar } from './WorkersFilterBar'

export const metadata: Metadata = {
  title: 'Workers | rbee Marketplace',
  description: 'Browse rbee workers for running language models and image generation on CPU, CUDA, and Metal.',
}

export default async function WorkersPage() {
  // Default filter (all workers)
  const currentFilters = PREGENERATED_WORKER_FILTERS[0].filters
  const filteredWorkers = filterWorkers(WORKERS, currentFilters)
  const filterDescription = buildFilterDescription(currentFilters)

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Workers</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Browse and install rbee workers for LLM inference and image generation
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{filteredWorkers.length} workers available</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-blue-500" />
            <span>CPU, CUDA, and Metal support</span>
          </div>
        </div>
      </div>

      {/* Filter Bar - Client Component */}
      <WorkersFilterBar groups={WORKER_FILTER_GROUPS} currentFilters={currentFilters} />

      {/* Workers Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {filteredWorkers.map((worker) => (
          <Link key={worker.id} href={`/workers/${worker.id}`} className="block">
            <WorkerCard
              worker={{
                id: worker.id,
                name: worker.name,
                description: worker.description,
                version: worker.version,
                platform: worker.platforms,
                architecture: worker.architectures,
                workerType: worker.workerType,
              }}
            />
          </Link>
        ))}
      </div>
    </div>
  )
}
