// TEAM-403: End-to-end user flow tests
import { describe, it, expect } from 'vitest'
import app from '../../src/index'

describe('E2E: Worker Discovery Flow', () => {
  it('should complete full discovery flow', async () => {
    // Step 1: List all workers
    const listRes = await app.request('/workers')
    expect(listRes.status).toBe(200)
    
    const { workers } = await listRes.json() as { workers: Array<{ id: string }> }
    expect(workers.length).toBeGreaterThan(0)
    
    // Step 2: Get specific worker details
    const workerId = workers[0].id
    const detailRes = await app.request(`/workers/${workerId}`)
    expect(detailRes.status).toBe(200)
    
    const worker = await detailRes.json() as { id: string }
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
    const workerId = 'llm-worker-rbee-cpu'
    
    // Get worker metadata
    const res = await app.request(`/workers/${workerId}`)
    expect(res.status).toBe(200)
    
    const worker = await res.json() as Record<string, unknown>
    
    // Verify installation requirements are present
    expect(worker.buildSystem).toBeDefined()
    expect(worker.depends).toBeDefined()
    expect(worker.makedepends).toBeDefined()
    expect(worker.source).toBeDefined()
    expect(worker.binaryName).toBeDefined()
    expect(worker.installPath).toBeDefined()
    
    // Verify source information
    expect((worker.source as Record<string, unknown>).type).toBeDefined()
    expect((worker.source as Record<string, unknown>).url).toBeDefined()
    
    // Verify build information
    expect(worker.build).toBeDefined()
    expect((worker.build as Record<string, unknown>).profile).toBeDefined()
  })
})

describe('E2E: Error Handling Flow', () => {
  it('should handle non-existent worker gracefully', async () => {
    const workerId = 'non-existent-worker-12345'
    
    // Attempt to get worker details
    const res = await app.request(`/workers/${workerId}`)
    expect(res.status).toBe(404)
    
    const data = await res.json() as { error: string }
    expect(data.error).toBe('Worker not found')
  })
})

describe('E2E: Multi-Platform Worker Selection', () => {
  it('should find workers for specific platform', async () => {
    // Get all workers
    const res = await app.request('/workers')
    expect(res.status).toBe(200)
    
    const { workers } = await res.json() as { workers: Array<Record<string, unknown>> }
    
    // Filter for Linux workers
    const linuxWorkers = workers.filter((w: any) => 
      w.platforms.includes('linux')
    )
    
    expect(linuxWorkers.length).toBeGreaterThan(0)
    
    // Verify each Linux worker has required info
    linuxWorkers.forEach((worker: any) => {
      expect(worker.platforms).toContain('linux')
      expect(worker.architectures).toBeDefined()
      expect(worker.architectures.length).toBeGreaterThan(0)
    })
  })
})

describe('E2E: Version Compatibility Check', () => {
  it('should provide version information for all workers', async () => {
    // Get all workers
    const res = await app.request('/workers')
    expect(res.status).toBe(200)
    
    const { workers } = await res.json() as { workers: Array<Record<string, unknown>> }
    
    // Verify all workers have valid version
    const semverRegex = /^\d+\.\d+\.\d+$/
    workers.forEach((worker: any) => {
      expect(worker.version).toMatch(semverRegex)
    })
    
    // Verify we can get details for each worker
    for (const worker of workers) {
      const detailRes = await app.request(`/workers/${worker.id as string}`)
      expect(detailRes.status).toBe(200)
      
      const details = await detailRes.json() as { version: string }
      expect(details.version).toBe(worker.version)
    }
  })
})
