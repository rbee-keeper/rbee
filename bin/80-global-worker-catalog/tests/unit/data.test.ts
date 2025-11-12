// TEAM-403: Worker catalog data validation tests
import { describe, expect, it } from 'vitest'
import { WORKERS } from '../../src/data'

describe('Worker Catalog Data', () => {
  it('should have at least 3 workers', () => {
    expect(WORKERS.length).toBeGreaterThanOrEqual(3)
  })

  it('should have unique worker IDs', () => {
    const ids = WORKERS.map((w) => w.id)
    const uniqueIds = new Set(ids)
    expect(uniqueIds.size).toBe(ids.length)
  })

  it('should have valid semver versions', () => {
    const semverRegex = /^\d+\.\d+\.\d+$/
    WORKERS.forEach((worker) => {
      expect(worker.version).toMatch(semverRegex)
    })
  })

  it('should have non-empty descriptions', () => {
    WORKERS.forEach((worker) => {
      expect(worker.description.length).toBeGreaterThan(10)
    })
  })

  it('should have at least one build variant', () => {
    WORKERS.forEach((worker) => {
      expect(worker.variants.length).toBeGreaterThan(0)
    })
  })

  it('should have valid PKGBUILD URLs in variants', () => {
    WORKERS.forEach((worker) => {
      worker.variants.forEach((variant) => {
        expect(variant.pkgbuildUrl).toMatch(/^\/workers\/[\w-]+\/PKGBUILD$/)
      })
    })
  })

  it('should have valid license identifiers', () => {
    const validLicenses = ['GPL-3.0-or-later', 'MIT', 'Apache-2.0', 'Proprietary']
    WORKERS.forEach((worker) => {
      expect(validLicenses).toContain(worker.license)
    })
  })

  it('should have at least one platform per variant', () => {
    WORKERS.forEach((worker) => {
      worker.variants.forEach((variant) => {
        expect(variant.platforms.length).toBeGreaterThan(0)
      })
    })
  })

  it('should have at least one architecture per variant', () => {
    WORKERS.forEach((worker) => {
      worker.variants.forEach((variant) => {
        expect(variant.architectures.length).toBeGreaterThan(0)
      })
    })
  })

  it('should have valid source URLs', () => {
    WORKERS.forEach((worker) => {
      expect(worker.source.url).toMatch(/^https?:\/\//)
    })
  })
})

describe('Worker-Specific Validation', () => {
  it('should validate llm-worker-rbee with CPU variant', () => {
    const llmWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee')
    expect(llmWorker).toBeDefined()
    
    const cpuVariant = llmWorker?.variants.find((v) => v.backend === 'cpu')
    expect(cpuVariant).toBeDefined()
    expect(cpuVariant?.platforms).toContain('linux')
    expect(cpuVariant?.build.features).toContain('cpu')
  })

  it('should validate llm-worker-rbee with CUDA variant', () => {
    const llmWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee')
    expect(llmWorker).toBeDefined()
    
    const cudaVariant = llmWorker?.variants.find((v) => v.backend === 'cuda')
    expect(cudaVariant).toBeDefined()
    expect(cudaVariant?.depends).toContain('cuda')
  })

  it('should validate llm-worker-rbee with Metal variant', () => {
    const llmWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee')
    expect(llmWorker).toBeDefined()
    
    const metalVariant = llmWorker?.variants.find((v) => v.backend === 'metal')
    expect(metalVariant).toBeDefined()
    expect(metalVariant?.platforms).toContain('macos')
  })

  it('should validate sd-worker-rbee with all 4 variants', () => {
    const sdWorker = WORKERS.find((w) => w.id === 'sd-worker-rbee')
    expect(sdWorker).toBeDefined()
    expect(sdWorker?.variants.length).toBe(4)
    
    const backends = sdWorker?.variants.map((v) => v.backend)
    expect(backends).toContain('cpu')
    expect(backends).toContain('cuda')
    expect(backends).toContain('metal')
    expect(backends).toContain('rocm')
  })
})
