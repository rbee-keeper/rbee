// TEAM-413: Worker detail page with SSG
// TEAM-421: Refactored to use shared WorkerDetailWithInstall component
// TEAM-453: Now fetches workers from gwc.rbee.dev instead of hardcoded data
import { listWorkers, type WorkerCatalogEntry } from '@rbee/marketplace-node'
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { WorkerDetailWithInstall } from '@/components/WorkerDetailWithInstall'

// TEAM-453: Worker interface for UI component
interface Worker {
  id: string
  name: string
  description: string
  type: 'cpu' | 'cuda' | 'metal' | 'rocm'
  platform: string[]
  version: string
  requirements: string[]
  features: string[]
}

// TEAM-453: Convert WorkerCatalogEntry to Worker interface
function convertWorkerCatalogEntry(entry: WorkerCatalogEntry): Worker {
  return {
    id: entry.id,
    name: entry.name,
    description: entry.description,
    type: entry.workerType as 'cpu' | 'cuda' | 'metal' | 'rocm',
    platform: entry.platforms,
    version: entry.version,
    requirements: entry.depends,
    features: entry.supportedFormats.map(format => `Supports ${format}`),
  }
}

// TEAM-453: Fetch workers from gwc.rbee.dev
async function getWorkers(): Promise<Record<string, Worker>> {
  console.log('[SSG] Fetching workers from gwc.rbee.dev')
  const catalogEntries = await listWorkers()
  const workersMap: Record<string, Worker> = {}
  
  for (const entry of catalogEntries) {
    workersMap[entry.id] = convertWorkerCatalogEntry(entry)
  }
  
  console.log(`[SSG] Converted ${Object.keys(workersMap).length} workers`)
  return workersMap
}

interface PageProps {
  params: Promise<{ workerId: string }>
}

export async function generateStaticParams() {
  console.log('[SSG] Fetching workers from gwc.rbee.dev')
  const workers = await getWorkers()
  console.log(`[SSG] Pre-building ${Object.keys(workers).length} worker pages`)
  
  return Object.keys(workers).map((workerId) => ({
    workerId,
  }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { workerId } = await params
  const workers = await getWorkers()
  const worker = workers[workerId]

  if (!worker) {
    return {
      title: 'Worker Not Found',
    }
  }

  return {
    title: `${worker.name} | rbee Marketplace`,
    description: worker.description,
  }
}

export default async function WorkerDetailPage({ params }: PageProps) {
  const { workerId } = await params
  const workers = await getWorkers()
  const worker = workers[workerId]

  if (!worker) {
    notFound()
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <WorkerDetailWithInstall worker={worker} />
    </div>
  )
}
