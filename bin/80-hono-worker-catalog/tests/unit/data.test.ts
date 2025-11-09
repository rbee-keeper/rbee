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

  it('should have valid PKGBUILD URLs', () => {
    WORKERS.forEach((worker) => {
      expect(worker.pkgbuildUrl).toMatch(/^\/workers\/[\w-]+\/PKGBUILD$/)
    })
  })

  it('should have matching ID in PKGBUILD URL', () => {
    WORKERS.forEach((worker) => {
      expect(worker.pkgbuildUrl).toContain(worker.id)
    })
  })

  it('should have valid license identifiers', () => {
    const validLicenses = ['GPL-3.0-or-later', 'MIT', 'Apache-2.0', 'Proprietary']
    WORKERS.forEach((worker) => {
      expect(validLicenses).toContain(worker.license)
    })
  })

  it('should have at least one platform', () => {
    WORKERS.forEach((worker) => {
      expect(worker.platforms.length).toBeGreaterThan(0)
    })
  })

  it('should have at least one architecture', () => {
    WORKERS.forEach((worker) => {
      expect(worker.architectures.length).toBeGreaterThan(0)
    })
  })

  it('should have valid source URLs', () => {
    WORKERS.forEach((worker) => {
      expect(worker.source.url).toMatch(/^https?:\/\//)
    })
  })
})

describe('Worker-Specific Validation', () => {
  it('should validate llm-worker-rbee-cpu', () => {
    const cpuWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee-cpu')
    expect(cpuWorker).toBeDefined()
    expect(cpuWorker?.workerType).toBe('cpu')
    expect(cpuWorker?.platforms).toContain('linux')
    expect(cpuWorker?.build.features).toContain('cpu')
  })

  it('should validate llm-worker-rbee-cuda', () => {
    const cudaWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee-cuda')
    expect(cudaWorker).toBeDefined()
    expect(cudaWorker?.workerType).toBe('cuda')
    expect(cudaWorker?.depends).toContain('cuda')
  })

  it('should validate llm-worker-rbee-metal', () => {
    const metalWorker = WORKERS.find((w) => w.id === 'llm-worker-rbee-metal')
    expect(metalWorker).toBeDefined()
    expect(metalWorker?.workerType).toBe('metal')
    expect(metalWorker?.platforms).toContain('macos')
  })
})
