// Worker Catalog Data
// TEAM-481: Consolidated from 8 separate entries to 2 workers with build variants
// Users pick backend (CPU/CUDA/Metal/ROCm) at download time

import type { WorkerCatalogEntry } from './types'

export const WORKERS: WorkerCatalogEntry[] = [
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // LLM Worker - Text generation and chat inference
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'llm-worker-rbee',
    implementation: 'rust',
    version: '0.1.0',
    name: 'LLM Worker',
    description: 'Candle-based LLM inference worker for text generation and chat. Supports CPU, CUDA, Metal, and ROCm acceleration.',
    license: 'GPL-3.0-or-later',
    buildSystem: 'cargo',
    source: {
      type: 'git',
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'main',
      path: 'bin/30_llm_worker_rbee',
    },
    variants: [
      {
        backend: 'cpu',
        platforms: ['linux', 'macos', 'windows'],
        architectures: ['x86_64', 'aarch64'],
        pkgbuildUrl: '/workers/llm-worker-rbee-cpu/PKGBUILD',
        build: {
          features: ['cpu'],
          profile: 'release',
        },
        depends: ['gcc'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee-cpu',
        installPath: '/usr/local/bin/llm-worker-rbee-cpu',
      },
      {
        backend: 'cuda',
        platforms: ['linux', 'windows'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/workers/llm-worker-rbee-cuda/PKGBUILD',
        build: {
          features: ['cuda'],
          profile: 'release',
        },
        depends: ['gcc', 'cuda'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee-cuda',
        installPath: '/usr/local/bin/llm-worker-rbee-cuda',
      },
      {
        backend: 'metal',
        platforms: ['macos'],
        architectures: ['aarch64'],
        pkgbuildUrl: '/workers/llm-worker-rbee-metal/PKGBUILD',
        build: {
          features: ['metal'],
          profile: 'release',
        },
        depends: ['clang'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee-metal',
        installPath: '/usr/local/bin/llm-worker-rbee-metal',
      },
      {
        backend: 'rocm',
        platforms: ['linux'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/workers/llm-worker-rbee-rocm/PKGBUILD',
        build: {
          features: ['rocm'],
          profile: 'release',
        },
        depends: ['gcc', 'rocm'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee-rocm',
        installPath: '/usr/local/bin/llm-worker-rbee-rocm',
      },
    ],
    supportedFormats: ['gguf', 'safetensors'], // TEAM-409: ASPIRATIONAL - GGUF needed for competitive parity
    maxContextLength: 32768,
    supportsStreaming: true,
    supportsBatching: false,
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // SD Worker - Image generation (Stable Diffusion)
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'sd-worker-rbee',
    implementation: 'rust',
    version: '0.1.0',
    name: 'SD Worker',
    description: 'Candle-based Stable Diffusion inference worker for image generation. Supports CPU, CUDA, Metal, and ROCm acceleration.',
    license: 'GPL-3.0-or-later',
    buildSystem: 'cargo',
    source: {
      type: 'git',
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'main',
      path: 'bin/31_sd_worker_rbee',
    },
    variants: [
      {
        backend: 'cpu',
        platforms: ['linux', 'macos', 'windows'],
        architectures: ['x86_64', 'aarch64'],
        pkgbuildUrl: '/workers/sd-worker-rbee-cpu/PKGBUILD',
        build: {
          features: ['cpu'],
          profile: 'release',
        },
        depends: ['gcc'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee-cpu',
        installPath: '/usr/local/bin/sd-worker-rbee-cpu',
      },
      {
        backend: 'cuda',
        platforms: ['linux', 'windows'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/workers/sd-worker-rbee-cuda/PKGBUILD',
        build: {
          features: ['cuda'],
          profile: 'release',
        },
        depends: ['gcc', 'cuda'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee-cuda',
        installPath: '/usr/local/bin/sd-worker-rbee-cuda',
      },
      {
        backend: 'metal',
        platforms: ['macos'],
        architectures: ['aarch64'],
        pkgbuildUrl: '/workers/sd-worker-rbee-metal/PKGBUILD',
        build: {
          features: ['metal'],
          profile: 'release',
        },
        depends: ['clang'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee-metal',
        installPath: '/usr/local/bin/sd-worker-rbee-metal',
      },
      {
        backend: 'rocm',
        platforms: ['linux'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/workers/sd-worker-rbee-rocm/PKGBUILD',
        build: {
          features: ['rocm'],
          profile: 'release',
        },
        depends: ['gcc', 'rocm'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee-rocm',
        installPath: '/usr/local/bin/sd-worker-rbee-rocm',
      },
    ],
    supportedFormats: ['safetensors'],
    supportsStreaming: true,
    supportsBatching: false,
  },
]
