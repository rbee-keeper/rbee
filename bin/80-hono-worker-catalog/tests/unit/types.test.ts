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
      'llm-worker-rbee',
      'llama-cpp-adapter',
      'vllm-adapter',
      'ollama-adapter',
      'comfyui-adapter'
    ]
    expect(validImplementations).toHaveLength(5)
    expect(validImplementations).toContain('llm-worker-rbee')
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
      implementation: 'llm-worker-rbee',
      worker_type: 'cpu',
      version: '0.1.0',
      platforms: ['linux'],
      architectures: ['x86_64'],
      name: 'Test Worker',
      description: 'Test description',
      license: 'GPL-3.0-or-later',
      pkgbuild_url: '/workers/test-worker/PKGBUILD',
      build_system: 'cargo',
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
      binary_name: 'test-worker',
      install_path: '/usr/local/bin/test-worker',
      supported_formats: ['gguf'],
      supports_streaming: true,
      supports_batching: false
    }
    
    expect(worker.id).toBe('test-worker')
    expect(worker.platforms).toContain('linux')
    expect(worker.worker_type).toBe('cpu')
    expect(worker.implementation).toBe('llm-worker-rbee')
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
      max_context_length: 32768,
      build: {
        features: ['cpu', 'avx2'],
        profile: 'release',
        flags: ['--target', 'x86_64-unknown-linux-gnu']
      }
    }
    
    expect(workerWithOptionals.max_context_length).toBe(32768)
    expect(workerWithOptionals.build?.features).toHaveLength(2)
    expect(workerWithOptionals.build?.flags).toBeDefined()
  })
})
