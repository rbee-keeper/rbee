// Data validation tests for worker catalog
// Created by: TEAM-451

import { describe, expect, it } from 'vitest'
import { WORKERS } from './data'

describe('Worker Catalog Data', () => {
  it('should have all required worker variants', () => {
    const requiredWorkers = [
      'llm-worker-rbee-cpu',
      'llm-worker-rbee-cuda',
      'llm-worker-rbee-metal',
      'sd-worker-rbee-cpu',
      'sd-worker-rbee-cuda',
    ]

    const workerIds = WORKERS.map((w) => w.id)

    for (const required of requiredWorkers) {
      expect(workerIds).toContain(required)
    }
  })

  it('should have all required fields for each worker', () => {
    const requiredFields = [
      'id',
      'implementation',
      'workerType',
      'version',
      'platforms',
      'architectures',
      'name',
      'description',
      'license',
      'binaryName',
    ]

    for (const worker of WORKERS) {
      for (const field of requiredFields) {
        expect(worker).toHaveProperty(field)
        expect((worker as any)[field]).toBeDefined()
      }
    }
  })

  it('should have valid version format', () => {
    const versionRegex = /^\d+\.\d+\.\d+$/

    for (const worker of WORKERS) {
      expect(worker.version).toMatch(versionRegex)
    }
  })

  it('should have valid platforms', () => {
    const validPlatforms = ['linux', 'macos', 'windows']

    for (const worker of WORKERS) {
      expect(worker.platforms.length).toBeGreaterThan(0)
      for (const platform of worker.platforms) {
        expect(validPlatforms).toContain(platform)
      }
    }
  })

  it('should have valid architectures', () => {
    const validArchs = ['x86_64', 'aarch64']

    for (const worker of WORKERS) {
      expect(worker.architectures.length).toBeGreaterThan(0)
      for (const arch of worker.architectures) {
        expect(validArchs).toContain(arch)
      }
    }
  })
})
