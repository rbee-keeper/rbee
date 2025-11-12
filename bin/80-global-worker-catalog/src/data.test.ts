// Data validation tests for worker catalog
// Created by: TEAM-451

import { describe, expect, it } from 'vitest'
import { WORKERS } from './data'

describe('Worker Catalog Data', () => {
  it('should have all required workers with variants', () => {
    const requiredWorkers = ['llm-worker-rbee', 'sd-worker-rbee']

    const workerIds = WORKERS.map((w) => w.id)

    for (const required of requiredWorkers) {
      expect(workerIds).toContain(required)
    }

    // Verify each worker has variants
    for (const worker of WORKERS) {
      expect(worker.variants.length).toBeGreaterThan(0)
    }
  })

  it('should have all required fields for each worker', () => {
    const requiredWorkerFields = [
      'id',
      'implementation',
      'version',
      'name',
      'description',
      'license',
      'variants',
    ]

    const requiredVariantFields = [
      'backend',
      'platforms',
      'architectures',
      'binaryName',
      'pkgbuildUrl',
    ]

    for (const worker of WORKERS) {
      // Check worker-level fields
      for (const field of requiredWorkerFields) {
        expect(worker).toHaveProperty(field)
      }

      // Check variant-level fields
      for (const variant of worker.variants) {
        for (const field of requiredVariantFields) {
          expect(variant).toHaveProperty(field)
        }
      }
    }
  })

  it('should have valid version format', () => {
    const versionRegex = /^\d+\.\d+\.\d+$/

    for (const worker of WORKERS) {
      expect(worker.version).toMatch(versionRegex)
    }
  })

  it('should have valid platforms in variants', () => {
    const validPlatforms = ['linux', 'macos', 'windows']

    for (const worker of WORKERS) {
      for (const variant of worker.variants) {
        expect(variant.platforms.length).toBeGreaterThan(0)
        for (const platform of variant.platforms) {
          expect(validPlatforms).toContain(platform)
        }
      }
    }
  })

  it('should have valid architectures in variants', () => {
    const validArchs = ['x86_64', 'aarch64']

    for (const worker of WORKERS) {
      for (const variant of worker.variants) {
        expect(variant.architectures.length).toBeGreaterThan(0)
        for (const arch of variant.architectures) {
          expect(validArchs).toContain(arch)
        }
      }
    }
  })
})
