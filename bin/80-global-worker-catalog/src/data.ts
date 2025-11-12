// Worker Catalog Data
// TEAM-481: Consolidated from 8 separate entries to 2 workers with build variants
// TEAM-483: Import directly from marketplace-core (no shim file)
// Users pick backend (CPU/CUDA/Metal/ROCm) at download time

import type { GWCWorker } from '@rbee/marketplace-core'

export const WORKERS: GWCWorker[] = [
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // LLM Worker - Text generation and chat inference
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'llm-worker-rbee',
    implementation: 'rust',
    version: '0.1.0',
    name: 'LLM Worker rbee',
    description:
      'Candle-based LLM inference worker for text generation and chat. Supports CPU, CUDA, Metal, and ROCm acceleration.',
    license: 'GPL-3.0-or-later',
    coverImage: 'https://backend.rbee.dev/images/llm-worker-rbee.png',
    readmeUrl:
      'https://raw.githubusercontent.com/rbee-keeper/rbee/refs/heads/development/bin/30_llm_worker_rbee/README.md',
    buildSystem: 'cargo',
    source: {
      type: 'git',
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'production--30_llm_worker_rbee',
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
    capabilities: {
      supportedFormats: ['gguf', 'safetensors'], // Both formats supported (see src/backend/models/mod.rs line 254)
      maxContextLength: 32768,
      supportsStreaming: true,
      supportsBatching: false,
    },
    marketplaceCompatibility: {
      huggingface: {
        // LLM Worker ONLY supports text-generation task
        // See: frontend/apps/marketplace/WORKER_COMPATIBILITY_MATRIX.md line 39-44
        // ❌ NOT SUPPORTED: text2text-generation, fill-mask, token-classification, etc.
        tasks: [
          'text-generation', // ONLY text-generation (Llama, Mistral, Phi, Qwen, Gemma)
        ],
        libraries: [
          'transformers', // HuggingFace transformers library
        ],
      },
    },
  },

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // SD Worker - Image generation (Stable Diffusion)
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {
    id: 'sd-worker-rbee',
    implementation: 'rust',
    version: '0.1.0',
    name: 'SD Worker rbee',
    description:
      'Candle-based Stable Diffusion inference worker for image generation. Supports CPU, CUDA, Metal, and ROCm acceleration.',
    license: 'GPL-3.0-or-later',
    coverImage: 'https://backend.rbee.dev/images/sd-worker-rbee.png',
    readmeUrl: 'https://github.com/rbee-keeper/rbee/blob/development/bin/31_sd_worker_rbee/README.md',
    buildSystem: 'cargo',
    source: {
      type: 'git',
      url: 'https://github.com/rbee-keeper/rbee.git',
      branch: 'production--31_sd_worker_rbee',
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
    capabilities: {
      supportedFormats: ['safetensors'],
      supportsStreaming: true,
      supportsBatching: false,
    },
    marketplaceCompatibility: {
      huggingface: {
        // SD Worker supports image generation models
        tasks: [
          'text-to-image', // Stable Diffusion models
        ],
        libraries: [
          'diffusers', // HuggingFace diffusers library
        ],
      },
      civitai: {
        // SD Worker ONLY supports base Checkpoint models
        // Source: src/backend/models/mod.rs line 10-27 (SDVersion enum)
        // ❌ NO LoRA, ControlNet, TextualInversion, Hypernetwork support in source code
        modelTypes: [
          'Checkpoint',   // ✅ ONLY Checkpoints (V1.5, V2.1, XL, Turbo)
        ],
        baseModels: [
          // SD 1.x series
          'SD 1.4',
          'SD 1.5',
          // SD 2.x series
          'SD 2.0',
          'SD 2.0 768',
          'SD 2.1',
          'SD 2.1 768',
          'SD 2.1 Unclip',
          // SDXL series
          'SDXL 0.9',
          'SDXL 1.0',
          'SDXL 1.0 LCM',
          'SDXL Distilled',
          'SDXL Turbo',
          // SD 3.x series
          'SD 3',
          'SD 3.5',
          // FLUX series (Candle has full support!)
          // Source: reference/candle/candle-transformers/src/models/flux/
          // 'Flux.1 D',  // ⏳ ASPIRATIONAL - 4-6 days to integrate
          // 'Flux.1 S',  // ⏳ ASPIRATIONAL - 4-6 days to integrate
        ],
      },
    },
  },
]
