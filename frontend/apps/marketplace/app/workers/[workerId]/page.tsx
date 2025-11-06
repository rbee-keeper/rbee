// TEAM-413: Worker detail page with SSG
// TEAM-421: Refactored to use shared WorkerDetailWithInstall component
import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { WorkerDetailWithInstall } from '@/components/WorkerDetailWithInstall'

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
    <div className="container mx-auto px-4 py-8 max-w-7xl">
      <WorkerDetailWithInstall worker={worker} />
    </div>
  )
}
