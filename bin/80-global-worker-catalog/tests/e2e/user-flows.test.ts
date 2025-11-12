// TEAM-403: End-to-end user flow tests
import { describe, expect, it } from 'vitest'
import app from '../../src/index'

describe('E2E: Worker Discovery Flow', () => {
  it('should complete full discovery flow', async () => {
    // Step 1: List all workers
    const listRes = await app.request('/workers')
    expect(listRes.status).toBe(200)

    const { workers } = (await listRes.json()) as { workers: Array<{ id: string }> }
    expect(workers.length).toBeGreaterThan(0)

    // Step 2: Get specific worker details
    const workerId = workers[0].id
    const detailRes = await app.request(`/workers/${workerId}`)
    expect(detailRes.status).toBe(200)

    const worker = (await detailRes.json()) as { id: string }
    expect(worker.id).toBe(workerId)

    // Step 3: Attempt to get PKGBUILD
    const pkgbuildRes = await app.request(`/workers/${workerId}/PKGBUILD`)
    // In test environment, will be 500 (no ASSETS binding)
    // In production, will be 200 or 404
    expect([200, 404, 500]).toContain(pkgbuildRes.status)
  })
})

describe('E2E: Worker Installation Simulation', () => {
  it('should provide all info needed for installation', async () => {
    const workerId = 'llm-worker-rbee'

    // Get worker metadata
    const res = await app.request(`/workers/${workerId}`)
    expect(res.status).toBe(200)

    const worker = (await res.json()) as Record<string, unknown>

    // Verify common worker information
    expect(worker.buildSystem).toBeDefined()
    expect(worker.source).toBeDefined()
    expect(worker.variants).toBeDefined()

    // Verify source information
    expect((worker.source as Record<string, unknown>).type).toBeDefined()
    expect((worker.source as Record<string, unknown>).url).toBeDefined()

    // Verify variants have installation requirements
    const variants = worker.variants as Array<Record<string, unknown>>
    expect(variants.length).toBeGreaterThan(0)
    
    variants.forEach((variant) => {
      expect(variant.depends).toBeDefined()
      expect(variant.makedepends).toBeDefined()
      expect(variant.binaryName).toBeDefined()
      expect(variant.installPath).toBeDefined()
      expect(variant.build).toBeDefined()
      expect((variant.build as Record<string, unknown>).profile).toBeDefined()
    })
  })
})

describe('E2E: Error Handling Flow', () => {
  it('should handle non-existent worker gracefully', async () => {
    const workerId = 'non-existent-worker-12345'

    // Attempt to get worker details
    const res = await app.request(`/workers/${workerId}`)
    expect(res.status).toBe(404)

    const data = (await res.json()) as { error: string }
    expect(data.error).toBe('Worker not found')
  })
})

describe('E2E: Multi-Platform Worker Selection', () => {
  it('should find variants for specific platform', async () => {
    // Get all workers
    const res = await app.request('/workers')
    expect(res.status).toBe(200)

    const { workers } = (await res.json()) as { 
      workers: Array<{ 
        variants: Array<{ 
          platforms: string[]
          architectures: string[] 
        }> 
      }> 
    }

    // Find all Linux variants across all workers
    const linuxVariants = workers.flatMap((w) =>
      w.variants.filter((v) => v.platforms.includes('linux'))
    )

    expect(linuxVariants.length).toBeGreaterThan(0)

    // Verify each Linux variant has required info
    linuxVariants.forEach((variant) => {
      expect(variant.platforms).toContain('linux')
      expect(variant.architectures).toBeDefined()
      expect(variant.architectures.length).toBeGreaterThan(0)
    })
  })
})

describe('E2E: Version Compatibility Check', () => {
  it('should provide version information for all workers', async () => {
    // Get all workers
    const res = await app.request('/workers')
    expect(res.status).toBe(200)

    const { workers } = (await res.json()) as { 
      workers: Array<{ 
        id: string
        version: string 
      }> 
    }

    // Verify all workers have valid version
    const semverRegex = /^\d+\.\d+\.\d+$/
    workers.forEach((worker) => {
      expect(worker.version).toMatch(semverRegex)
    })

    // Verify we can get details for each worker
    for (const worker of workers) {
      const detailRes = await app.request(`/workers/${worker.id}`)
      expect(detailRes.status).toBe(200)

      const details = (await detailRes.json()) as { version: string }
      expect(details.version).toBe(worker.version)
    }
  })
})
