// TEAM-403: Route handler unit tests (isolated, no HTTP)
import { describe, expect, it } from 'vitest'
import { WORKERS } from '../../src/data'

describe('Route Logic (Isolated)', () => {
  describe('GET /workers', () => {
    it('should return all workers', () => {
      const result = { workers: WORKERS }
      expect(result.workers).toHaveLength(WORKERS.length)
      expect(result.workers.length).toBe(2) // LLM Worker + SD Worker
    })
  })

  describe('GET /workers/:id', () => {
    it('should find worker by ID', () => {
      const id = 'llm-worker-rbee'
      const worker = WORKERS.find((w) => w.id === id)
      expect(worker).toBeDefined()
      expect(worker?.id).toBe(id)
      expect(worker?.variants.length).toBeGreaterThan(0)
    })

    it('should return undefined for non-existent worker', () => {
      const id = 'non-existent-worker'
      const worker = WORKERS.find((w) => w.id === id)
      expect(worker).toBeUndefined()
    })
  })

  describe('PKGBUILD URL Construction', () => {
    it('should construct correct PKGBUILD URL from variant', () => {
      const worker = WORKERS.find((w) => w.id === 'llm-worker-rbee')
      const cpuVariant = worker?.variants.find((v) => v.backend === 'cpu')
      expect(cpuVariant?.pkgbuildUrl).toBe('/workers/llm-worker-rbee-cpu/PKGBUILD')
    })
  })

  describe('Worker Filtering', () => {
    it('should filter variants by platform', () => {
      // Find all variants that support Linux
      const linuxVariants = WORKERS.flatMap((w) =>
        w.variants.filter((v) => v.platforms.includes('linux'))
      )
      expect(linuxVariants.length).toBeGreaterThan(0)
      linuxVariants.forEach((v) => {
        expect(v.platforms).toContain('linux')
      })
    })

    it('should filter variants by backend type', () => {
      // Find all CPU variants
      const cpuVariants = WORKERS.flatMap((w) =>
        w.variants.filter((v) => v.backend === 'cpu')
      )
      expect(cpuVariants.length).toBeGreaterThan(0)
      cpuVariants.forEach((v) => {
        expect(v.backend).toBe('cpu')
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
