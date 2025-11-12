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
        pkgbuildUrl: '/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/llm-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/llm-worker-rbee-git.rb',
        build: {
          features: ['cpu'],
          profile: 'release',
        },
        depends: ['gcc'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee',
        installPath: '/usr/local/bin/llm-worker-rbee',
      },
      {
        backend: 'cuda',
        platforms: ['linux', 'windows'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/llm-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/llm-worker-rbee-git.rb',
        build: {
          features: ['cuda'],
          profile: 'release',
        },
        depends: ['gcc', 'cuda'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee',
        installPath: '/usr/local/bin/llm-worker-rbee',
      },
      {
        backend: 'metal',
        platforms: ['macos'],
        architectures: ['aarch64'],
        pkgbuildUrl: '/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/llm-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/llm-worker-rbee-git.rb',
        build: {
          features: ['metal'],
          profile: 'release',
        },
        depends: ['clang'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee',
        installPath: '/usr/local/bin/llm-worker-rbee',
      },
      {
        backend: 'rocm',
        platforms: ['linux'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/pkgbuilds/arch/llm-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/llm-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/llm-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/llm-worker-rbee-git.rb',
        build: {
          features: ['rocm'],
          profile: 'release',
        },
        depends: ['gcc', 'rocm'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'llm-worker-rbee',
        installPath: '/usr/local/bin/llm-worker-rbee',
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
        pkgbuildUrl: '/pkgbuilds/arch/sd-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/sd-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/sd-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/sd-worker-rbee-git.rb',
        build: {
          features: ['cpu'],
          profile: 'release',
        },
        depends: ['gcc'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee',
        installPath: '/usr/local/bin/sd-worker-rbee',
      },
      {
        backend: 'cuda',
        platforms: ['linux', 'windows'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/pkgbuilds/arch/sd-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/sd-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/sd-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/sd-worker-rbee-git.rb',
        build: {
          features: ['cuda'],
          profile: 'release',
        },
        depends: ['gcc', 'cuda'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee',
        installPath: '/usr/local/bin/sd-worker-rbee',
      },
      {
        backend: 'metal',
        platforms: ['macos'],
        architectures: ['aarch64'],
        pkgbuildUrl: '/pkgbuilds/arch/sd-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/sd-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/sd-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/sd-worker-rbee-git.rb',
        build: {
          features: ['metal'],
          profile: 'release',
        },
        depends: ['clang'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee',
        installPath: '/usr/local/bin/sd-worker-rbee',
      },
      {
        backend: 'rocm',
        platforms: ['linux'],
        architectures: ['x86_64'],
        pkgbuildUrl: '/pkgbuilds/arch/sd-worker-rbee-bin.PKGBUILD',
        pkgbuildUrlGit: '/pkgbuilds/arch/sd-worker-rbee-git.PKGBUILD',
        homebrewFormula: '/pkgbuilds/homebrew/sd-worker-rbee-bin.rb',
        homebrewFormulaGit: '/pkgbuilds/homebrew/sd-worker-rbee-git.rb',
        build: {
          features: ['rocm'],
          profile: 'release',
        },
        depends: ['gcc', 'rocm'],
        makedepends: ['rust', 'cargo'],
        binaryName: 'sd-worker-rbee',
        installPath: '/usr/local/bin/sd-worker-rbee',
      },
    ],
    supportedFormats: ['safetensors'],
    supportsStreaming: true,
    supportsBatching: false,
  },
]
