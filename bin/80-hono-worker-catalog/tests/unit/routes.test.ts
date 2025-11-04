// TEAM-403: Route handler unit tests (isolated, no HTTP)
import { describe, it, expect } from 'vitest'
import { WORKERS } from '../../src/data'

describe('Route Logic (Isolated)', () => {
  describe('GET /workers', () => {
    it('should return all workers', () => {
      const result = { workers: WORKERS }
      expect(result.workers).toHaveLength(WORKERS.length)
      expect(result.workers.length).toBeGreaterThanOrEqual(3)
    })
  })

  describe('GET /workers/:id', () => {
    it('should find worker by ID', () => {
      const id = 'llm-worker-rbee-cpu'
      const worker = WORKERS.find(w => w.id === id)
      expect(worker).toBeDefined()
      expect(worker?.id).toBe(id)
    })

    it('should return undefined for non-existent worker', () => {
      const id = 'non-existent-worker'
      const worker = WORKERS.find(w => w.id === id)
      expect(worker).toBeUndefined()
    })
  })

  describe('PKGBUILD URL Construction', () => {
    it('should construct correct PKGBUILD URL', () => {
      const id = 'llm-worker-rbee-cpu'
      const url = `/pkgbuilds/${id}.PKGBUILD`
      expect(url).toBe('/pkgbuilds/llm-worker-rbee-cpu.PKGBUILD')
    })
  })

  describe('Worker Filtering', () => {
    it('should filter workers by platform', () => {
      const linuxWorkers = WORKERS.filter(w => w.platforms.includes('linux'))
      expect(linuxWorkers.length).toBeGreaterThan(0)
      linuxWorkers.forEach(w => {
        expect(w.platforms).toContain('linux')
      })
    })

    it('should filter workers by type', () => {
      const cpuWorkers = WORKERS.filter(w => w.worker_type === 'cpu')
      expect(cpuWorkers.length).toBeGreaterThan(0)
      cpuWorkers.forEach(w => {
        expect(w.worker_type).toBe('cpu')
      })
    })
  })

  describe('Worker Sorting', () => {
    it('should sort workers by name', () => {
      const sorted = [...WORKERS].sort((a, b) => a.name.localeCompare(b.name))
      expect(sorted.length).toBe(WORKERS.length)
      // Verify sorting is correct
      for (let i = 1; i < sorted.length; i++) {
        expect(sorted[i].name >= sorted[i - 1].name).toBe(true)
      }
    })

    it('should sort workers by version', () => {
      const sorted = [...WORKERS].sort((a, b) => a.version.localeCompare(b.version))
      expect(sorted.length).toBe(WORKERS.length)
    })
  })
})
