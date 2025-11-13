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
      'https://raw.githubusercontent.com/rbee-keeper/rbee/development/bin/30_llm_worker_rbee/README.md',
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
    readmeUrl: 'https://raw.githubusercontent.com/rbee-keeper/rbee/development/bin/31_sd_worker_rbee/README.md',
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
        // SD Worker supports Stable Diffusion + FLUX image generation
        // Available filters: author, library, task, tags, model_name, language, trained_dataset
        // See: .docs/HUGGINGFACE_FILTERS_AVAILABLE.md for all options
        tasks: [
          'text-to-image', // SD 1.x, 2.x, XL, FLUX models
        ],
        libraries: [
          'diffusers', // HuggingFace diffusers library
        ],
        // Optional filters (uncomment to enable):
        // tags: ['safetensors'],  // Only safe models (recommended)
        // tags: ['stable-diffusion-xl', 'flux'],  // Only modern models
        // author: ['stabilityai', 'black-forest-labs'],  // Only official models
      },
      civitai: {
        // SD Worker supports Stable Diffusion + FLUX base Checkpoint models
        // Source: bin/31_sd_worker_rbee/src/backend/models/mod.rs (SDVersion enum)
        // ✅ FLUX fully integrated (TEAM-488)
        // ✅ LoRA support FULLY WIRED (TEAM-488)
        // ❌ NO ControlNet, TextualInversion, Hypernetwork support
        modelTypes: [
          'Checkpoint',   // ✅ Checkpoints only (SD 1.x, 2.x, XL, FLUX)
          'LORA',         // ✅ FULLY SUPPORTED (TEAM-488)
        ],
        baseModels: [
          // SD 1.x series - ✅ SUPPORTED
          'SD 1.4',
          'SD 1.5',
          'SD 1.5 LCM',        // ✅ Scheduler variant
          'SD 1.5 Hyper',      // ✅ Scheduler variant
          
          // SD 2.x series - ✅ SUPPORTED
          'SD 2.0',
          'SD 2.0 768',
          'SD 2.1',
          'SD 2.1 768',
          'SD 2.1 Unclip',
          
          // SDXL series - ✅ SUPPORTED
          'SDXL 0.9',
          'SDXL 1.0',
          'SDXL 1.0 LCM',
          'SDXL Distilled',
          'SDXL Turbo',
          'SDXL Lightning',    // ✅ Scheduler variant
          'SDXL Hyper',        // ✅ Scheduler variant
          
          // FLUX series - ✅ FULLY INTEGRATED (TEAM-488)
          // Source: bin/31_sd_worker_rbee/src/backend/models/flux/
          'Flux.1 D',          // ✅ FLUX.1-dev (50 steps, guidance 3.5)
          'Flux.1 S',          // ✅ FLUX.1-schnell (4 steps, fast)
          'Flux.1 Krea',       // ✅ Compatible (FLUX fine-tune)
          'Flux.1 Kontext',    // ✅ Compatible (FLUX fine-tune)
          
          // Anime/Character models - ✅ SUPPORTED (SDXL fine-tunes)
          'Illustrious',       // ✅ SDXL-based anime model
          'Pony',              // ✅ Pony V6 (SDXL-based)
          // 'Pony V7',           // ❌ NOT SUPPORTED (AuraFlow architecture)
          'NoobAI',            // ✅ NoobAI XL (SDXL-based)
          
          // ❌ NOT SUPPORTED (different architectures)
          // 'SD 3',           // ⏳ In Candle, needs integration
          // 'SD 3.5',         // ⏳ In Candle, needs integration
          // 'Kolors',         // ❌ Custom architecture + ChatGLM
          // 'Aura Flow',      // ❌ Not in Candle
          // 'PixArt α',       // ❌ Not in Candle
          // 'PixArt Σ',       // ❌ Not in Candle
        ],
      },
    },
  },
]
