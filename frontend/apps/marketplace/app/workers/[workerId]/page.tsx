// TEAM-413: Worker detail page with SSG
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'

// TEAM-413: Worker type definitions
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

// TEAM-413: Static worker catalog
const WORKERS: Record<string, Worker> = {
  'cpu-llm': {
    id: 'cpu-llm',
    name: 'CPU LLM Worker',
    description: 'Run language models on CPU. Works on any system without GPU requirements.',
    type: 'cpu',
    platform: ['linux', 'macos', 'windows'],
    version: '0.1.0',
    requirements: [
      'x86_64 or ARM64 processor',
      '8GB RAM minimum (16GB recommended)',
      'No GPU required',
    ],
    features: [
      'Universal compatibility',
      'No special drivers needed',
      'Optimized for CPU inference',
      'Supports quantized models',
    ],
  },
  'cuda-llm': {
    id: 'cuda-llm',
    name: 'CUDA LLM Worker',
    description: 'GPU-accelerated language model inference using NVIDIA CUDA.',
    type: 'cuda',
    platform: ['linux', 'windows'],
    version: '0.1.0',
    requirements: [
      'NVIDIA GPU (Compute Capability 7.0+)',
      'CUDA 11.8 or later',
      '8GB VRAM minimum',
    ],
    features: [
      '10-100x faster than CPU',
      'Supports larger models',
      'Flash Attention 2 support',
      'Multi-GPU support',
    ],
  },
  'metal-llm': {
    id: 'metal-llm',
    name: 'Metal LLM Worker',
    description: 'GPU-accelerated language model inference using Apple Metal.',
    type: 'metal',
    platform: ['macos'],
    version: '0.1.0',
    requirements: [
      'Apple Silicon (M1/M2/M3) or AMD GPU',
      'macOS 13.0 or later',
      '8GB unified memory minimum',
    ],
    features: [
      'Native Apple Silicon acceleration',
      'Unified memory architecture',
      'Low power consumption',
      'Optimized for Mac hardware',
    ],
  },
  'rocm-llm': {
    id: 'rocm-llm',
    name: 'ROCm LLM Worker',
    description: 'GPU-accelerated language model inference using AMD ROCm.',
    type: 'rocm',
    platform: ['linux'],
    version: '0.1.0',
    requirements: [
      'AMD GPU (GCN 4th gen or later)',
      'ROCm 5.7 or later',
      '8GB VRAM minimum',
    ],
    features: [
      'AMD GPU acceleration',
      'Open-source stack',
      'Competitive with CUDA',
      'Growing ecosystem',
    ],
  },
}

interface PageProps {
  params: Promise<{ workerId: string }>
}

export async function generateStaticParams() {
  return Object.keys(WORKERS).map((workerId) => ({
    workerId,
  }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { workerId } = await params
  const worker = WORKERS[workerId]
  
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
  const worker = WORKERS[workerId]

  if (!worker) {
    notFound()
  }

  return (
    <div className="container mx-auto px-4 py-12 max-w-5xl">
      {/* Header */}
      <div className="mb-8 space-y-4">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center rounded-full bg-primary/10 px-4 py-1.5 text-sm font-medium text-primary">
            {worker.type.toUpperCase()}
          </span>
          <span className="text-sm text-muted-foreground">
            v{worker.version}
          </span>
        </div>

        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          {worker.name}
        </h1>

        <p className="text-lg text-muted-foreground max-w-3xl">
          {worker.description}
        </p>
      </div>

      {/* Install Button Placeholder */}
      <div className="mb-12">
        <div className="rounded-lg border border-border bg-card p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-semibold">Install Worker</h3>
              <p className="text-sm text-muted-foreground">
                One-click installation with rbee Keeper
              </p>
            </div>
            <a
              href={`rbee://worker/${worker.id}`}
              className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Install with rbee
            </a>
          </div>
        </div>
      </div>

      {/* Details Grid */}
      <div className="grid gap-8 md:grid-cols-2">
        {/* Platforms */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Supported Platforms</h2>
          <div className="space-y-2">
            {worker.platform.map((platform) => (
              <div
                key={platform}
                className="flex items-center gap-3 rounded-lg border border-border bg-card p-4"
              >
                <div className="size-2 rounded-full bg-primary" />
                <span className="font-medium capitalize">{platform}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Requirements */}
        <div className="space-y-4">
          <h2 className="text-2xl font-semibold">Requirements</h2>
          <ul className="space-y-2">
            {worker.requirements.map((req, index) => (
              <li
                key={index}
                className="flex items-start gap-3 rounded-lg border border-border bg-card p-4"
              >
                <span className="text-primary">•</span>
                <span className="text-sm">{req}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Features */}
      <div className="mt-8 space-y-4">
        <h2 className="text-2xl font-semibold">Features</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {worker.features.map((feature, index) => (
            <div
              key={index}
              className="flex items-start gap-3 rounded-lg border border-border bg-card p-4"
            >
              <div className="mt-1 size-5 rounded-full bg-primary/10 flex items-center justify-center">
                <span className="text-xs text-primary">✓</span>
              </div>
              <span className="text-sm">{feature}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
