// TEAM-403: Type validation tests
import { describe, it, expect } from 'vitest'
import type { WorkerCatalogEntry, WorkerType, Platform, Architecture, WorkerImplementation, BuildSystem } from '../../src/types'

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
    const validImplementations: WorkerImplementation[] = [
      'rust',
      'python',
      'cpp'
    ]
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

describe('WorkerCatalogEntry Structure', () => {
  it('should validate complete worker entry structure', () => {
    const worker: WorkerCatalogEntry = {
      id: 'test-worker',
      implementation: 'rust',
      workerType: 'cpu',
      version: '0.1.0',
      platforms: ['linux'],
      architectures: ['x86_64'],
      name: 'Test Worker',
      description: 'Test description',
      license: 'GPL-3.0-or-later',
      pkgbuildUrl: '/workers/test-worker/PKGBUILD',
      buildSystem: 'cargo',
      source: {
        type: 'git',
        url: 'https://github.com/test/repo.git',
        branch: 'main'
      },
      build: {
        features: ['cpu'],
        profile: 'release'
      },
      depends: ['gcc'],
      makedepends: ['rust', 'cargo'],
      binaryName: 'test-worker',
      installPath: '/usr/local/bin/test-worker',
      supportedFormats: ['gguf'],
      supportsStreaming: true,
      supportsBatching: false
    }
    
    expect(worker.id).toBe('test-worker')
    expect(worker.platforms).toContain('linux')
    expect(worker.workerType).toBe('cpu')
    expect(worker.implementation).toBe('rust')
  })

  it('should validate source type variants - git', () => {
    const gitSource = {
      type: 'git' as const,
      url: 'https://github.com/test/repo.git',
      branch: 'main',
      path: 'bin/worker'
    }
    
    expect(gitSource.type).toBe('git')
    expect(gitSource.url).toMatch(/^https:\/\//)
    expect(gitSource.branch).toBeDefined()
  })

  it('should validate optional fields', () => {
    const workerWithOptionals: Partial<WorkerCatalogEntry> = {
      maxContextLength: 32768,
      build: {
        features: ['cpu', 'avx2'],
        profile: 'release',
        flags: ['--target', 'x86_64-unknown-linux-gnu']
      }
    }
    
    expect(workerWithOptionals.maxContextLength).toBe(32768)
    expect(workerWithOptionals.build?.features).toHaveLength(2)
    expect(workerWithOptionals.build?.flags).toBeDefined()
  })
})
