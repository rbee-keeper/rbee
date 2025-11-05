// TEAM-413: Workers list page with SSG
import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Workers | rbee Marketplace',
  description: 'Browse rbee workers for running AI models on different hardware.',
}

// TEAM-413: Worker type definitions
interface Worker {
  id: string
  name: string
  description: string
  type: 'cpu' | 'cuda' | 'metal' | 'rocm'
  platform: string[]
  version: string
}

// TEAM-413: Static worker catalog (until Worker Catalog API is ready)
const WORKERS: Worker[] = [
  {
    id: 'cpu-llm',
    name: 'CPU LLM Worker',
    description: 'Run language models on CPU. Works on any system without GPU requirements.',
    type: 'cpu',
    platform: ['linux', 'macos', 'windows'],
    version: '0.1.0',
  },
  {
    id: 'cuda-llm',
    name: 'CUDA LLM Worker',
    description: 'GPU-accelerated language model inference using NVIDIA CUDA.',
    type: 'cuda',
    platform: ['linux', 'windows'],
    version: '0.1.0',
  },
  {
    id: 'metal-llm',
    name: 'Metal LLM Worker',
    description: 'GPU-accelerated language model inference using Apple Metal.',
    type: 'metal',
    platform: ['macos'],
    version: '0.1.0',
  },
  {
    id: 'rocm-llm',
    name: 'ROCm LLM Worker',
    description: 'GPU-accelerated language model inference using AMD ROCm.',
    type: 'rocm',
    platform: ['linux'],
    version: '0.1.0',
  },
]

export default async function WorkersPage() {
  return (
    <div className="container mx-auto px-4 py-12 max-w-7xl">
      {/* Header Section */}
      <div className="mb-12 space-y-4">
        <div className="space-y-2">
          <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
            Workers
          </h1>
          <p className="text-muted-foreground text-lg md:text-xl max-w-3xl">
            Browse and install rbee workers for running AI models on different hardware
          </p>
        </div>
        
        {/* Stats */}
        <div className="flex items-center gap-6 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-primary" />
            <span>{WORKERS.length} workers available</span>
          </div>
        </div>
      </div>

      {/* Workers Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {WORKERS.map((worker) => (
          <a
            key={worker.id}
            href={`/workers/${worker.id}`}
            className="group rounded-lg border border-border bg-card p-6 transition-all hover:border-primary hover:shadow-lg"
          >
            <div className="space-y-4">
              {/* Worker Type Badge */}
              <div className="flex items-center justify-between">
                <span className="inline-flex items-center rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                  {worker.type.toUpperCase()}
                </span>
                <span className="text-xs text-muted-foreground">
                  v{worker.version}
                </span>
              </div>

              {/* Worker Info */}
              <div className="space-y-2">
                <h3 className="text-xl font-semibold group-hover:text-primary transition-colors">
                  {worker.name}
                </h3>
                <p className="text-sm text-muted-foreground line-clamp-2">
                  {worker.description}
                </p>
              </div>

              {/* Platforms */}
              <div className="flex flex-wrap gap-2">
                {worker.platform.map((platform) => (
                  <span
                    key={platform}
                    className="inline-flex items-center rounded-md bg-muted px-2 py-1 text-xs font-medium"
                  >
                    {platform}
                  </span>
                ))}
              </div>
            </div>
          </a>
        ))}
      </div>
    </div>
  )
}
