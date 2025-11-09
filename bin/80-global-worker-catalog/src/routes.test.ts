// API endpoint tests for worker catalog
// Created by: TEAM-451

import { describe, expect, it } from 'vitest'
import app from './index'

describe('Worker Catalog API', () => {
  describe('GET /health', () => {
    it('should return health status', async () => {
      const res = await app.request('/health')
      expect(res.status).toBe(200)

      const data = (await res.json()) as Record<string, unknown>
      expect(data).toHaveProperty('status', 'ok')
      expect(data).toHaveProperty('service', 'worker-catalog')
      expect(data).toHaveProperty('version')
    })
  })

  describe('GET /workers', () => {
    it('should return list of all workers', async () => {
      const res = await app.request('/workers')
      expect(res.status).toBe(200)

      const data = (await res.json()) as { workers: unknown[] }
      expect(data).toHaveProperty('workers')
      expect(Array.isArray(data.workers)).toBe(true)
      expect(data.workers.length).toBeGreaterThan(0)
    })

    it('should include required worker variants', async () => {
      const res = await app.request('/workers')
      const data = (await res.json()) as { workers: Array<{ id: string }> }

      const workerIds = data.workers.map((w) => w.id)

      expect(workerIds).toContain('llm-worker-rbee-cpu')
      expect(workerIds).toContain('llm-worker-rbee-cuda')
      expect(workerIds).toContain('llm-worker-rbee-metal')
      expect(workerIds).toContain('sd-worker-rbee-cpu')
      expect(workerIds).toContain('sd-worker-rbee-cuda')
    })
  })

  describe('GET /workers/:id', () => {
    it('should return specific worker by ID', async () => {
      const res = await app.request('/workers/llm-worker-rbee-cpu')
      expect(res.status).toBe(200)

      const data = (await res.json()) as Record<string, unknown>
      expect(data).toHaveProperty('id', 'llm-worker-rbee-cpu')
      expect(data).toHaveProperty('name')
      expect(data).toHaveProperty('version')
      expect(data).toHaveProperty('platforms')
      expect(data).toHaveProperty('architectures')
    })

    it('should return 404 for invalid worker ID', async () => {
      const res = await app.request('/workers/invalid-worker-id')
      expect(res.status).toBe(404)

      const data = (await res.json()) as Record<string, unknown>
      expect(data).toHaveProperty('error', 'Worker not found')
    })

    it('should work for all worker variants', async () => {
      const workerIds = [
        'llm-worker-rbee-cpu',
        'llm-worker-rbee-cuda',
        'llm-worker-rbee-metal',
        'sd-worker-rbee-cpu',
        'sd-worker-rbee-cuda',
      ]

      for (const id of workerIds) {
        const res = await app.request(`/workers/${id}`)
        expect(res.status).toBe(200)

        const data = (await res.json()) as { id: string }
        expect(data.id).toBe(id)
      }
    })
  })
})
