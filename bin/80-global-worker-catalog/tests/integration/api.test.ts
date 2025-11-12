// TEAM-403: HTTP API integration tests
import { describe, expect, it } from 'vitest'
import app from '../../src/index'

describe('Worker Catalog API', () => {
  describe('GET /health', () => {
    it('should return 200 OK', async () => {
      const res = await app.request('/health')
      expect(res.status).toBe(200)
    })

    it('should return correct health check data', async () => {
      const res = await app.request('/health')
      const data = (await res.json()) as { status: string; service: string; version: string }

      expect(data.status).toBe('ok')
      expect(data.service).toBe('worker-catalog')
      expect(data.version).toBe('0.1.0')
    })
  })

  describe('GET /workers', () => {
    it('should return 200 OK', async () => {
      const res = await app.request('/workers')
      expect(res.status).toBe(200)
    })

    it('should return JSON array', async () => {
      const res = await app.request('/workers')
      const data = (await res.json()) as { workers: unknown[] }

      expect(data).toHaveProperty('workers')
      expect(Array.isArray(data.workers)).toBe(true)
      expect(data.workers.length).toBeGreaterThan(0)
    })

    it('should return workers with required fields', async () => {
      const res = await app.request('/workers')
      const data = (await res.json()) as { workers: Record<string, unknown>[] }

      const worker = data.workers[0]
      expect(worker).toHaveProperty('id')
      expect(worker).toHaveProperty('name')
      expect(worker).toHaveProperty('version')
      expect(worker).toHaveProperty('description')
      expect(worker).toHaveProperty('variants')
      expect(worker).toHaveProperty('supportedFormats')
    })

    it('should return response in reasonable time', async () => {
      const start = Date.now()
      const res = await app.request('/workers')
      const duration = Date.now() - start

      expect(res.status).toBe(200)
      expect(duration).toBeLessThan(200) // Less than 200ms
    })
  })

  describe('GET /workers/:id', () => {
    it('should return 200 for valid worker', async () => {
      const res = await app.request('/workers/llm-worker-rbee')
      expect(res.status).toBe(200)
    })

    it('should return worker details with variants', async () => {
      const res = await app.request('/workers/llm-worker-rbee')
      const data = (await res.json()) as { 
        id: string
        implementation: string
        variants: Array<{ backend: string }>
      }

      expect(data.id).toBe('llm-worker-rbee')
      expect(data.implementation).toBe('rust')
      expect(data.variants.length).toBeGreaterThan(0)
      expect(data.variants[0]).toHaveProperty('backend')
    })

    it('should return 404 for non-existent worker', async () => {
      const res = await app.request('/workers/non-existent')
      expect(res.status).toBe(404)
    })

    it('should return error message for 404', async () => {
      const res = await app.request('/workers/non-existent')
      const data = (await res.json()) as { error: string }

      expect(data.error).toBe('Worker not found')
    })
  })

  describe('GET /workers/:id/PKGBUILD', () => {
    it('should return 404 for non-existent worker', async () => {
      const res = await app.request('/workers/non-existent/PKGBUILD')
      expect(res.status).toBe(404)
    })

    it('should return error for non-existent worker', async () => {
      const res = await app.request('/workers/non-existent/PKGBUILD')
      const data = (await res.json()) as { error: string }

      expect(data.error).toBe('Worker not found')
    })

    it('should attempt to fetch PKGBUILD for valid worker', async () => {
      const res = await app.request('/workers/llm-worker-rbee/PKGBUILD')
      // Will be 404 or 500 in test environment (no ASSETS binding)
      // In production with Cloudflare, will be 200 or 404
      expect([200, 404, 500]).toContain(res.status)
    })
  })
})
