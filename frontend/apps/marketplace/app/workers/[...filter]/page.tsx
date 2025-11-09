// TEAM-461: Dynamic filtered workers pages (SSG pre-generated)

import { WorkerCard } from '@rbee/ui/marketplace'
import type { Metadata } from 'next'
import Link from 'next/link'
import { WORKERS } from '@/../../../bin/80-hono-worker-catalog/src/data'
import {
  buildFilterDescription,
  filterWorkers,
  getWorkerFilterFromPath,
  PREGENERATED_WORKER_FILTERS,
  WORKER_FILTER_GROUPS,
} from '../filters'
import { WorkersFilterBar } from '../WorkersFilterBar'

interface PageProps {
  params: Promise<{
    filter: string[]
  }>
}

// Pre-generate static pages for all filter combinations
export async function generateStaticParams() {
  return PREGENERATED_WORKER_FILTERS.filter((f) => f.path !== '') // Exclude default (handled by main page)
    .map((f) => ({
      filter: f.path.split('/').filter(Boolean), // Remove empty strings
    }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilters = getWorkerFilterFromPath(filterPath)
  const description = buildFilterDescription(currentFilters)

  return {
    title: `${description} Workers | rbee Marketplace`,
    description: `Browse ${description.toLowerCase()} workers for rbee. Install and run on your hardware.`,
  }
}

export default async function FilteredWorkersPage({ params }: PageProps) {
  const { filter } = await params
  const filterPath = filter.join('/')
  const currentFilters = getWorkerFilterFromPath(filterPath)
  const filteredWorkers = filterWorkers(WORKERS, currentFilters)
  const filterDescription = buildFilterDescription(currentFilters)

  console.log(`[SSG] Showing ${filteredWorkers.length} workers (${filterPath})`)

  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-8 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">Workers</h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            {filterDescription} · Browse and install rbee workers
          </p>
        </div>

        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{filteredWorkers.length} workers</span>
          </div>
          {currentFilters.category !== 'all' && (
            <div className="flex items-center gap-2">
              <div className="size-2 rounded-full bg-blue-500" />
              <span>{currentFilters.category === 'llm' ? 'Language Models' : 'Image Generation'}</span>
            </div>
          )}
        </div>
      </div>

      {/* Filter Bar */}
      <WorkersFilterBar groups={WORKER_FILTER_GROUPS} currentFilters={currentFilters} />

      {/* Workers Grid */}
      {filteredWorkers.length > 0 ? (
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
      ) : (
        <div className="text-center py-12">
          <p className="text-muted-foreground text-lg">No workers match the selected filters.</p>
          <Link href="/workers" className="text-primary hover:underline mt-2 inline-block">
            Clear filters
          </Link>
        </div>
      )}
    </div>
  )
}
