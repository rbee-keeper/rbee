// TEAM-403: Type validation tests
// TEAM-483: Import directly from marketplace-core (no shim file)

import type {
  Architecture,
  BuildSystem,
  BuildVariant,
  GWCWorker,
  Platform,
  WorkerImplementation,
  WorkerType,
} from '@rbee/marketplace-core'
import { describe, expect, it } from 'vitest'

describe('WorkerType Enum', () => {
  it('should validate worker type enum values', () => {
    const validTypes: WorkerType[] = ['cpu', 'cuda', 'metal']
    expect(validTypes).toHaveLength(3)
    expect(validTypes).toContain('cpu')
    expect(validTypes).toContain('cuda')
    expect(validTypes).toContain('metal')
  })
})

describe('Platform Enum', () => {
  it('should validate platform enum values', () => {
    const validPlatforms: Platform[] = ['linux', 'macos', 'windows']
    expect(validPlatforms).toHaveLength(3)
    expect(validPlatforms).toContain('linux')
    expect(validPlatforms).toContain('macos')
    expect(validPlatforms).toContain('windows')
  })
})

describe('Architecture Enum', () => {
  it('should validate architecture enum values', () => {
    const validArchitectures: Architecture[] = ['x86_64', 'aarch64']
    expect(validArchitectures).toHaveLength(2)
    expect(validArchitectures).toContain('x86_64')
    expect(validArchitectures).toContain('aarch64')
  })
})

describe('WorkerImplementation Enum', () => {
  it('should validate worker implementation enum values', () => {
    const validImplementations: WorkerImplementation[] = ['rust', 'python', 'cpp']
    expect(validImplementations).toHaveLength(3)
    expect(validImplementations).toContain('rust')
  })
})

describe('BuildSystem Enum', () => {
  it('should validate build system enum values', () => {
    const validBuildSystems: BuildSystem[] = ['cargo', 'cmake', 'pip', 'npm']
    expect(validBuildSystems).toHaveLength(4)
    expect(validBuildSystems).toContain('cargo')
  })
})

describe('GWCWorker Structure', () => {
  it('should validate complete worker entry structure', () => {
    const worker: GWCWorker = {
      id: 'test-worker',
      implementation: 'rust',
      version: '0.1.0',
      name: 'Test Worker',
      description: 'Test description',
      license: 'GPL-3.0-or-later',
      buildSystem: 'cargo',
      source: {
        type: 'git',
        url: 'https://github.com/test/repo.git',
        branch: 'main',
      },
      variants: [
        {
          backend: 'cpu',
          platforms: ['linux'],
          architectures: ['x86_64'],
          pkgbuildUrl: '/workers/test-worker/PKGBUILD',
          pkgbuildUrlGit: '/workers/test-worker-git/PKGBUILD',
          homebrewFormula: '/workers/test-worker.rb',
          homebrewFormulaGit: '/workers/test-worker-git.rb',
          build: {
            features: ['cpu'],
            profile: 'release',
          },
          depends: ['gcc'],
          makedepends: ['rust', 'cargo'],
          binaryName: 'test-worker',
          installPath: '/usr/local/bin/test-worker',
        },
      ],
      supportedFormats: ['gguf'],
      supportsStreaming: true,
      supportsBatching: false,
    }

    expect(worker.id).toBe('test-worker')
    expect(worker.variants).toHaveLength(1)
    expect(worker.variants[0]?.platforms).toContain('linux')
    expect(worker.variants[0]?.backend).toBe('cpu')
    expect(worker.implementation).toBe('rust')
  })

  it('should validate source type variants - git', () => {
    const gitSource = {
      type: 'git' as const,
      url: 'https://github.com/test/repo.git',
      branch: 'main',
      path: 'bin/worker',
    }

    expect(gitSource.type).toBe('git')
    expect(gitSource.url).toMatch(/^https:\/\//)
    expect(gitSource.branch).toBeDefined()
  })

  it('should validate optional fields', () => {
    const workerWithOptionals: Partial<GWCWorker> = {
      maxContextLength: 32768,
    }

    expect(workerWithOptionals.maxContextLength).toBe(32768)
  })

  it('should validate build variant structure', () => {
    const variant: BuildVariant = {
      backend: 'cuda',
      platforms: ['linux'],
      architectures: ['x86_64'],
      pkgbuildUrl: '/test.PKGBUILD',
      pkgbuildUrlGit: '/test-git.PKGBUILD',
      homebrewFormula: '/test.rb',
      homebrewFormulaGit: '/test-git.rb',
      build: {
        features: ['cuda', 'avx2'],
        profile: 'release',
        flags: ['--target', 'x86_64-unknown-linux-gnu'],
      },
      depends: ['cuda'],
      makedepends: ['rust', 'cargo'],
      binaryName: 'test-worker',
      installPath: '/usr/local/bin/test-worker',
    }

    expect(variant.backend).toBe('cuda')
    expect(variant.build.features).toHaveLength(2)
    expect(variant.build.flags).toBeDefined()
  })
})
