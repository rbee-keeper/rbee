// TEAM-453: Worker detail page with SSG
// TEAM-421: Refactored to use shared WorkerDetailWithInstall component
// TEAM-453: Now fetches workers from gwc.rbee.dev instead of hardcoded data
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

// Hardcoded worker data for now
const WORKERS: Worker[] = [
  {
    id: 'rbee-cuda-v0.1.0',
    name: 'rbee CUDA Worker',
    description: 'GPU-accelerated worker for NVIDIA CUDA systems',
    type: 'cuda',
    platform: ['linux'],
    version: '0.1.0',
    requirements: ['CUDA 12.0+', 'Linux', '8GB+ RAM'],
    features: ['GPU acceleration', 'Multi-model support', 'SSH deployment'],
  },
  {
    id: 'rbee-cpu-v0.1.0',
    name: 'rbee CPU Worker',
    description: 'CPU-only worker for inference and small models',
    type: 'cpu',
    platform: ['linux', 'macos', 'windows'],
    version: '0.1.0',
    requirements: ['x86_64', '4GB+ RAM'],
    features: ['CPU inference', 'Cross-platform', 'SSH deployment'],
  },
]

// TEAM-453: Get workers from hardcoded data
async function getWorkers(): Promise<Record<string, Worker>> {
  console.log('[SSG] Using hardcoded worker data')
  const workersMap: Record<string, Worker> = {}

  for (const worker of WORKERS) {
    workersMap[worker.id] = worker
  }

  console.log(`[SSG] Loaded ${Object.keys(workersMap).length} workers`)
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
