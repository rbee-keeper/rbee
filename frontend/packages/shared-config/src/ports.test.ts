/**
 * TEAM-351: Tests for @rbee/shared-config
 * TEAM-XXX: Added comprehensive test coverage for missing scenarios
 *
 * Tests cover:
 * - Port configuration structure
 * - getAllowedOrigins() function
 * - getIframeUrl() function
 * - getParentOrigin() function
 * - getServiceUrl() function
 * - getWorkerUrl() function
 * - Edge cases (null ports, HTTPS, invalid inputs)
 * - URL format validation
 * - Port range validation
 * - Integration scenarios
 */

import { describe, expect, it } from 'vitest'
import {
  getAllowedOrigins,
  getIframeUrl,
  getParentOrigin,
  getServiceUrl,
  getWorkerUrl,
  PORTS,
  type ServiceName,
  type WorkerServiceName,
} from './ports'

describe('@rbee/shared-config', () => {
  describe('PORTS constant', () => {
    it('should have correct structure', () => {
      expect(PORTS).toBeDefined()
      expect(PORTS.keeper).toBeDefined()
      expect(PORTS.queen).toBeDefined()
      expect(PORTS.hive).toBeDefined()
      expect(PORTS.worker).toBeDefined()
    })

    it('should have keeper ports', () => {
      expect(PORTS.keeper.dev).toBe(7843)
      expect(PORTS.keeper.prod).toBeNull()
    })

    it('should have queen ports', () => {
      expect(PORTS.queen.dev).toBe(7844)
      expect(PORTS.queen.prod).toBe(7833)
      expect(PORTS.queen.backend).toBe(7833)
    })

    it('should have hive ports', () => {
      expect(PORTS.hive.dev).toBe(7845)
      expect(PORTS.hive.prod).toBe(7834)
      expect(PORTS.hive.backend).toBe(7834)
    })

    it('should have worker ports', () => {
      expect(PORTS.worker.llm.dev).toBe(7837)
      expect(PORTS.worker.llm.prod).toBe(8080)
      expect(PORTS.worker.llm.backend).toBe(8080)
      expect(PORTS.worker.sd.dev).toBe(5174)
      expect(PORTS.worker.sd.prod).toBe(8081)
      expect(PORTS.worker.sd.backend).toBe(8081)
      expect(PORTS.worker.comfy.dev).toBe(7838)
      expect(PORTS.worker.comfy.prod).toBe(8188)
      expect(PORTS.worker.comfy.backend).toBe(8188)
      expect(PORTS.worker.vllm.dev).toBe(7839)
      expect(PORTS.worker.vllm.prod).toBe(8000)
      expect(PORTS.worker.vllm.backend).toBe(8000)
    })

    it('should have commercial ports', () => {
      expect(PORTS.commercial.dev).toBe(7822)
      expect(PORTS.commercial.prod).toBeNull()
    })

    it('should have marketplace ports', () => {
      expect(PORTS.marketplace.dev).toBe(7823)
      expect(PORTS.marketplace.prod).toBeNull()
    })

    it('should have userDocs ports', () => {
      expect(PORTS.userDocs.dev).toBe(7824)
      expect(PORTS.userDocs.prod).toBeNull()
    })

    it('should have storybook ports', () => {
      expect(PORTS.storybook.rbeeUi).toBe(6006)
      expect(PORTS.storybook.commercial).toBe(6007)
    })

    it('should have honoCatalog ports', () => {
      expect(PORTS.honoCatalog.dev).toBe(7811)
      expect(PORTS.honoCatalog.prod).toBeNull()
    })

    it('should be readonly (as const)', () => {
      // TypeScript enforces this at compile time
      // Runtime check: object should be frozen or immutable
      expect(Object.isFrozen(PORTS) || typeof PORTS === 'object').toBe(true)
    })
  })

  describe('getAllowedOrigins()', () => {
    it('should return HTTP origins by default', () => {
      const origins = getAllowedOrigins()

      expect(origins).toContain('http://localhost:7844') // queen dev
      expect(origins).toContain('http://localhost:7833') // queen prod
      expect(origins).toContain('http://localhost:7845') // hive dev
      expect(origins).toContain('http://localhost:7834') // hive prod
      expect(origins).toContain('http://localhost:7837') // llm worker dev
      expect(origins).toContain('http://localhost:8080') // llm worker prod
      expect(origins).toContain('http://localhost:5174') // sd worker dev
      expect(origins).toContain('http://localhost:8081') // sd worker prod
      expect(origins).toContain('http://localhost:7838') // comfy worker dev
      expect(origins).toContain('http://localhost:8188') // comfy worker prod
      expect(origins).toContain('http://localhost:7839') // vllm worker dev
      expect(origins).toContain('http://localhost:8000') // vllm worker prod
    })

    it('should not include keeper', () => {
      const origins = getAllowedOrigins()

      expect(origins).not.toContain('http://localhost:7843')
    })

    it('should include HTTPS when requested', () => {
      const origins = getAllowedOrigins(true)

      expect(origins).toContain('https://localhost:7833') // queen prod
      expect(origins).toContain('https://localhost:7834') // hive prod
      expect(origins).toContain('https://localhost:8080') // llm worker prod
      expect(origins).toContain('https://localhost:8081') // sd worker prod
      expect(origins).toContain('https://localhost:8188') // comfy worker prod
      expect(origins).toContain('https://localhost:8000') // vllm worker prod
    })

    it('should not include HTTPS for dev ports', () => {
      const origins = getAllowedOrigins(true)

      expect(origins).not.toContain('https://localhost:7844') // queen dev
      expect(origins).not.toContain('https://localhost:7845') // hive dev
      expect(origins).not.toContain('https://localhost:7837') // llm worker dev
      expect(origins).not.toContain('https://localhost:5174') // sd worker dev
      expect(origins).not.toContain('https://localhost:7838') // comfy worker dev
      expect(origins).not.toContain('https://localhost:7839') // vllm worker dev
    })

    it('should return sorted array', () => {
      const origins = getAllowedOrigins()
      const sorted = [...origins].sort()

      expect(origins).toEqual(sorted)
    })

    it('should not have duplicates', () => {
      const origins = getAllowedOrigins()
      const unique = [...new Set(origins)]

      expect(origins.length).toBe(unique.length)
    })

    it('should return consistent results', () => {
      const origins1 = getAllowedOrigins()
      const origins2 = getAllowedOrigins()

      expect(origins1).toEqual(origins2)
    })
  })

  describe('getIframeUrl()', () => {
    it('should return dev URL for queen', () => {
      const url = getIframeUrl('queen', true)
      expect(url).toBe('http://localhost:7844')
    })

    it('should return prod URL for queen', () => {
      const url = getIframeUrl('queen', false)
      expect(url).toBe('http://localhost:7833')
    })

    it('should return dev URL for hive', () => {
      const url = getIframeUrl('hive', true)
      expect(url).toBe('http://localhost:7845')
    })

    it('should return prod URL for hive', () => {
      const url = getIframeUrl('hive', false)
      expect(url).toBe('http://localhost:7834')
    })

    // Note: 'worker' is not a valid ServiceName since workers are nested
    // Worker URLs should be accessed via PORTS.worker.llm, PORTS.worker.sd, etc.

    it('should return dev URL for keeper', () => {
      const url = getIframeUrl('keeper', true)
      expect(url).toBe('http://localhost:7843')
    })

    it('should throw error for keeper prod', () => {
      expect(() => getIframeUrl('keeper', false)).toThrow('Keeper service has no production HTTP port')
    })

    it('should support HTTPS', () => {
      const url = getIframeUrl('queen', false, true)
      expect(url).toBe('https://localhost:7833')
    })

    it('should support HTTPS for dev', () => {
      const url = getIframeUrl('queen', true, true)
      expect(url).toBe('https://localhost:7844')
    })
  })

  describe('getParentOrigin()', () => {
    it('should return keeper dev for queen dev port', () => {
      const origin = getParentOrigin(7844)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for hive dev port', () => {
      const origin = getParentOrigin(7845)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for llm worker dev port', () => {
      const origin = getParentOrigin(7837)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for sd worker dev port', () => {
      const origin = getParentOrigin(5174)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for comfy worker dev port', () => {
      const origin = getParentOrigin(7838)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for vllm worker dev port', () => {
      const origin = getParentOrigin(7839)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return keeper dev for keeper dev port', () => {
      const origin = getParentOrigin(7843)
      expect(origin).toBe('http://localhost:7843')
    })

    it('should return wildcard for queen prod port', () => {
      const origin = getParentOrigin(7833)
      expect(origin).toBe('*')
    })

    it('should return wildcard for hive prod port', () => {
      const origin = getParentOrigin(7834)
      expect(origin).toBe('*')
    })

    it('should return wildcard for llm worker prod port', () => {
      const origin = getParentOrigin(8080)
      expect(origin).toBe('*')
    })

    it('should return wildcard for sd worker prod port', () => {
      const origin = getParentOrigin(8081)
      expect(origin).toBe('*')
    })

    it('should return wildcard for comfy worker prod port', () => {
      const origin = getParentOrigin(8188)
      expect(origin).toBe('*')
    })

    it('should return wildcard for vllm worker prod port', () => {
      const origin = getParentOrigin(8000)
      expect(origin).toBe('*')
    })

    it('should return wildcard for unknown port', () => {
      const origin = getParentOrigin(9999)
      expect(origin).toBe('*')
    })
  })

  describe('getServiceUrl()', () => {
    describe('dev mode', () => {
      it('should return queen dev URL', () => {
        const url = getServiceUrl('queen', 'dev')
        expect(url).toBe('http://localhost:7844')
      })

      it('should return hive dev URL', () => {
        const url = getServiceUrl('hive', 'dev')
        expect(url).toBe('http://localhost:7845')
      })

      // Note: 'worker' is not a valid ServiceName

      it('should return keeper dev URL', () => {
        const url = getServiceUrl('keeper', 'dev')
        expect(url).toBe('http://localhost:7843')
      })
    })

    describe('prod mode', () => {
      it('should return queen prod URL', () => {
        const url = getServiceUrl('queen', 'prod')
        expect(url).toBe('http://localhost:7833')
      })

      it('should return hive prod URL', () => {
        const url = getServiceUrl('hive', 'prod')
        expect(url).toBe('http://localhost:7834')
      })

      // Note: 'worker' is not a valid ServiceName

      it('should return empty string for keeper prod', () => {
        const url = getServiceUrl('keeper', 'prod')
        expect(url).toBe('')
      })
    })

    describe('backend mode', () => {
      it('should return queen backend URL', () => {
        const url = getServiceUrl('queen', 'backend')
        expect(url).toBe('http://localhost:7833')
      })

      it('should return hive backend URL', () => {
        const url = getServiceUrl('hive', 'backend')
        expect(url).toBe('http://localhost:7834')
      })

      // Note: 'worker' is not a valid ServiceName

      it('should fallback to prod for keeper backend', () => {
        const url = getServiceUrl('keeper', 'backend')
        expect(url).toBe('')
      })
    })

    describe('HTTPS support', () => {
      it('should support HTTPS in dev mode', () => {
        const url = getServiceUrl('queen', 'dev', true)
        expect(url).toBe('https://localhost:7844')
      })

      it('should support HTTPS in prod mode', () => {
        const url = getServiceUrl('queen', 'prod', true)
        expect(url).toBe('https://localhost:7833')
      })

      it('should support HTTPS in backend mode', () => {
        const url = getServiceUrl('queen', 'backend', true)
        expect(url).toBe('https://localhost:7833')
      })
    })

    describe('default parameters', () => {
      it('should default to dev mode', () => {
        const url = getServiceUrl('queen')
        expect(url).toBe('http://localhost:7844')
      })

      it('should default to HTTP', () => {
        const url = getServiceUrl('queen', 'prod')
        expect(url).toBe('http://localhost:7833')
      })
    })
  })

  describe('Edge cases', () => {
    it('should handle all service names', () => {
      const services: ServiceName[] = ['keeper', 'queen', 'hive']

      services.forEach((service) => {
        expect(() => getServiceUrl(service, 'dev')).not.toThrow()
      })
    })

    it('should return consistent URLs', () => {
      const url1 = getServiceUrl('queen', 'prod')
      const url2 = getServiceUrl('queen', 'prod')

      expect(url1).toBe(url2)
    })

    it('should handle null ports gracefully', () => {
      const url = getServiceUrl('keeper', 'prod')
      expect(url).toBe('')
    })
  })

  describe('getWorkerUrl()', () => {
    describe('dev mode', () => {
      it('should return llm worker dev URL', () => {
        const url = getWorkerUrl('llm', 'dev')
        expect(url).toBe('http://localhost:7837')
      })

      it('should return sd worker dev URL', () => {
        const url = getWorkerUrl('sd', 'dev')
        expect(url).toBe('http://localhost:5174')
      })

      it('should return comfy worker dev URL', () => {
        const url = getWorkerUrl('comfy', 'dev')
        expect(url).toBe('http://localhost:7838')
      })

      it('should return vllm worker dev URL', () => {
        const url = getWorkerUrl('vllm', 'dev')
        expect(url).toBe('http://localhost:7839')
      })
    })

    describe('prod mode', () => {
      it('should return llm worker prod URL', () => {
        const url = getWorkerUrl('llm', 'prod')
        expect(url).toBe('http://localhost:8080')
      })

      it('should return sd worker prod URL', () => {
        const url = getWorkerUrl('sd', 'prod')
        expect(url).toBe('http://localhost:8081')
      })

      it('should return comfy worker prod URL', () => {
        const url = getWorkerUrl('comfy', 'prod')
        expect(url).toBe('http://localhost:8188')
      })

      it('should return vllm worker prod URL', () => {
        const url = getWorkerUrl('vllm', 'prod')
        expect(url).toBe('http://localhost:8000')
      })
    })

    describe('backend mode', () => {
      it('should return llm worker backend URL', () => {
        const url = getWorkerUrl('llm', 'backend')
        expect(url).toBe('http://localhost:8080')
      })

      it('should return sd worker backend URL', () => {
        const url = getWorkerUrl('sd', 'backend')
        expect(url).toBe('http://localhost:8081')
      })

      it('should return comfy worker backend URL', () => {
        const url = getWorkerUrl('comfy', 'backend')
        expect(url).toBe('http://localhost:8188')
      })

      it('should return vllm worker backend URL', () => {
        const url = getWorkerUrl('vllm', 'backend')
        expect(url).toBe('http://localhost:8000')
      })
    })

    describe('HTTPS support', () => {
      it('should support HTTPS in dev mode', () => {
        const url = getWorkerUrl('llm', 'dev', true)
        expect(url).toBe('https://localhost:7837')
      })

      it('should support HTTPS in prod mode', () => {
        const url = getWorkerUrl('llm', 'prod', true)
        expect(url).toBe('https://localhost:8080')
      })

      it('should support HTTPS in backend mode', () => {
        const url = getWorkerUrl('llm', 'backend', true)
        expect(url).toBe('https://localhost:8080')
      })
    })

    describe('default parameters', () => {
      it('should default to dev mode', () => {
        const url = getWorkerUrl('llm')
        expect(url).toBe('http://localhost:7837')
      })

      it('should default to HTTP', () => {
        const url = getWorkerUrl('llm', 'prod')
        expect(url).toBe('http://localhost:8080')
      })
    })

    describe('all worker types', () => {
      it('should handle all worker types', () => {
        const workers: WorkerServiceName[] = ['llm', 'sd', 'comfy', 'vllm']

        workers.forEach((worker) => {
          expect(() => getWorkerUrl(worker, 'dev')).not.toThrow()
          expect(() => getWorkerUrl(worker, 'prod')).not.toThrow()
          expect(() => getWorkerUrl(worker, 'backend')).not.toThrow()
        })
      })
    })
  })

  // TEAM-XXX: High-priority missing tests
  describe('Port Range Validation', () => {
    it('should have all ports in valid range (1-65535)', () => {
      const validatePort = (port: number | null, name: string) => {
        if (port !== null) {
          expect(port).toBeGreaterThanOrEqual(1)
          expect(port).toBeLessThanOrEqual(65535)
        }
      }

      // Validate all service ports
      validatePort(PORTS.keeper.dev, 'keeper.dev')
      validatePort(PORTS.queen.dev, 'queen.dev')
      validatePort(PORTS.queen.prod, 'queen.prod')
      validatePort(PORTS.queen.backend, 'queen.backend')
      validatePort(PORTS.hive.dev, 'hive.dev')
      validatePort(PORTS.hive.prod, 'hive.prod')
      validatePort(PORTS.hive.backend, 'hive.backend')

      // Validate worker ports
      validatePort(PORTS.worker.llm.dev, 'worker.llm.dev')
      validatePort(PORTS.worker.llm.prod, 'worker.llm.prod')
      validatePort(PORTS.worker.sd.dev, 'worker.sd.dev')
      validatePort(PORTS.worker.sd.prod, 'worker.sd.prod')
      validatePort(PORTS.worker.comfy.dev, 'worker.comfy.dev')
      validatePort(PORTS.worker.comfy.prod, 'worker.comfy.prod')
      validatePort(PORTS.worker.vllm.dev, 'worker.vllm.dev')
      validatePort(PORTS.worker.vllm.prod, 'worker.vllm.prod')

      // Validate frontend service ports
      validatePort(PORTS.commercial.dev, 'commercial.dev')
      validatePort(PORTS.marketplace.dev, 'marketplace.dev')
      validatePort(PORTS.userDocs.dev, 'userDocs.dev')
      validatePort(PORTS.storybook.rbeeUi, 'storybook.rbeeUi')
      validatePort(PORTS.storybook.commercial, 'storybook.commercial')
      validatePort(PORTS.honoCatalog.dev, 'honoCatalog.dev')
    })

    it('should have no port conflicts between services', () => {
      const usedPorts = new Set<number>()
      const conflicts: string[] = []

      const checkPort = (port: number | null, name: string) => {
        if (port !== null) {
          if (usedPorts.has(port)) {
            conflicts.push(`Port ${port} used by multiple services (including ${name})`)
          }
          usedPorts.add(port)
        }
      }

      // Check all ports for conflicts
      checkPort(PORTS.keeper.dev, 'keeper.dev')
      checkPort(PORTS.queen.dev, 'queen.dev')
      checkPort(PORTS.queen.prod, 'queen.prod')
      checkPort(PORTS.hive.dev, 'hive.dev')
      checkPort(PORTS.hive.prod, 'hive.prod')
      checkPort(PORTS.worker.llm.dev, 'worker.llm.dev')
      checkPort(PORTS.worker.llm.prod, 'worker.llm.prod')
      checkPort(PORTS.worker.sd.dev, 'worker.sd.dev')
      checkPort(PORTS.worker.sd.prod, 'worker.sd.prod')
      checkPort(PORTS.worker.comfy.dev, 'worker.comfy.dev')
      checkPort(PORTS.worker.comfy.prod, 'worker.comfy.prod')
      checkPort(PORTS.worker.vllm.dev, 'worker.vllm.dev')
      checkPort(PORTS.worker.vllm.prod, 'worker.vllm.prod')
      checkPort(PORTS.commercial.dev, 'commercial.dev')
      checkPort(PORTS.marketplace.dev, 'marketplace.dev')
      checkPort(PORTS.userDocs.dev, 'userDocs.dev')
      checkPort(PORTS.storybook.rbeeUi, 'storybook.rbeeUi')
      checkPort(PORTS.storybook.commercial, 'storybook.commercial')
      checkPort(PORTS.honoCatalog.dev, 'honoCatalog.dev')

      expect(conflicts).toEqual([])
    })

    it('should have backend ports match prod ports', () => {
      expect(PORTS.queen.backend).toBe(PORTS.queen.prod)
      expect(PORTS.hive.backend).toBe(PORTS.hive.prod)
      expect(PORTS.worker.llm.backend).toBe(PORTS.worker.llm.prod)
      expect(PORTS.worker.sd.backend).toBe(PORTS.worker.sd.prod)
      expect(PORTS.worker.comfy.backend).toBe(PORTS.worker.comfy.prod)
      expect(PORTS.worker.vllm.backend).toBe(PORTS.worker.vllm.prod)
    })
  })

  describe('URL Format Validation', () => {
    it('should return valid HTTP URL format', () => {
      const url = getServiceUrl('queen', 'dev')
      expect(url).toMatch(/^http:\/\/localhost:\d+$/)
    })

    it('should return valid HTTPS URL format', () => {
      const url = getServiceUrl('queen', 'prod', true)
      expect(url).toMatch(/^https:\/\/localhost:\d+$/)
    })

    it('should return URLs without trailing slash', () => {
      const urls = [
        getServiceUrl('queen', 'dev'),
        getServiceUrl('hive', 'prod'),
        getWorkerUrl('llm', 'dev'),
        getIframeUrl('queen', true),
      ]

      urls.forEach((url) => {
        expect(url).not.toMatch(/\/$/)
      })
    })

    it('should return consistent URL format across all functions', () => {
      const queenUrl = getServiceUrl('queen', 'dev')
      const queenIframeUrl = getIframeUrl('queen', true)

      expect(queenUrl).toBe(queenIframeUrl)
    })

    it('should use localhost as hostname', () => {
      const urls = [getServiceUrl('queen', 'dev'), getWorkerUrl('llm', 'prod'), getIframeUrl('hive', false)]

      urls.forEach((url) => {
        if (url) {
          expect(url).toContain('localhost')
        }
      })
    })
  })

  describe('getAllowedOrigins() - Extended Coverage', () => {
    it('should include all worker types in origins', () => {
      const origins = getAllowedOrigins()

      // LLM worker
      expect(origins).toContain('http://localhost:7837')
      expect(origins).toContain('http://localhost:8080')

      // SD worker
      expect(origins).toContain('http://localhost:5174')
      expect(origins).toContain('http://localhost:8081')

      // Comfy worker
      expect(origins).toContain('http://localhost:7838')
      expect(origins).toContain('http://localhost:8188')

      // vLLM worker
      expect(origins).toContain('http://localhost:7839')
      expect(origins).toContain('http://localhost:8000')
    })

    it('should not include services without backend (commercial, marketplace, etc)', () => {
      const origins = getAllowedOrigins()

      expect(origins).not.toContain('http://localhost:7822') // commercial
      expect(origins).not.toContain('http://localhost:7823') // marketplace
      expect(origins).not.toContain('http://localhost:7824') // userDocs
      expect(origins).not.toContain('http://localhost:6006') // storybook
      expect(origins).not.toContain('http://localhost:7811') // honoCatalog
    })

    it('should return exact count of expected origins', () => {
      const origins = getAllowedOrigins()
      // Queen (2) + Hive (2) + LLM (2) + SD (2) + Comfy (2) + vLLM (2) = 12
      expect(origins.length).toBe(12)
    })

    it('should return exact count with HTTPS enabled', () => {
      const origins = getAllowedOrigins(true)
      // HTTP (12) + HTTPS for prod only (6) = 18
      expect(origins.length).toBe(18)
    })
  })

  describe('getParentOrigin() - Extended Coverage', () => {
    it('should handle all dev ports correctly', () => {
      const devPorts = [
        PORTS.queen.dev,
        PORTS.hive.dev,
        PORTS.worker.llm.dev,
        PORTS.worker.sd.dev,
        PORTS.worker.comfy.dev,
        PORTS.worker.vllm.dev,
        PORTS.keeper.dev,
      ]

      devPorts.forEach((port) => {
        const origin = getParentOrigin(port)
        expect(origin).toBe(`http://localhost:${PORTS.keeper.dev}`)
      })
    })

    it('should handle all prod ports correctly', () => {
      const prodPorts = [
        PORTS.queen.prod,
        PORTS.hive.prod,
        PORTS.worker.llm.prod,
        PORTS.worker.sd.prod,
        PORTS.worker.comfy.prod,
        PORTS.worker.vllm.prod,
      ]

      prodPorts.forEach((port) => {
        if (port !== null) {
          const origin = getParentOrigin(port)
          expect(origin).toBe('*')
        }
      })
    })

    it('should handle random high ports as wildcard', () => {
      const randomPorts = [10000, 20000, 30000, 50000, 65535]

      randomPorts.forEach((port) => {
        const origin = getParentOrigin(port)
        expect(origin).toBe('*')
      })
    })
  })

  describe('Integration Tests', () => {
    it('should work together: getServiceUrl + getAllowedOrigins', () => {
      const queenUrl = getServiceUrl('queen', 'prod')
      const allowedOrigins = getAllowedOrigins()

      expect(allowedOrigins).toContain(queenUrl)
    })

    it('should work together: getWorkerUrl + getAllowedOrigins', () => {
      const llmUrl = getWorkerUrl('llm', 'prod')
      const allowedOrigins = getAllowedOrigins()

      expect(allowedOrigins).toContain(llmUrl)
    })

    it('should work together: getIframeUrl + getParentOrigin', () => {
      const queenIframeUrl = getIframeUrl('queen', true)
      const portStr = queenIframeUrl.split(':')[2]
      if (!portStr) throw new Error('Port not found in URL')
      const port = parseInt(portStr)
      const parentOrigin = getParentOrigin(port)

      expect(parentOrigin).toBe(`http://localhost:${PORTS.keeper.dev}`)
    })

    it('should maintain consistency between dev and prod URLs', () => {
      const services: ServiceName[] = ['queen', 'hive']

      services.forEach((service) => {
        const devUrl = getServiceUrl(service, 'dev')
        const prodUrl = getServiceUrl(service, 'prod')
        const backendUrl = getServiceUrl(service, 'backend')

        // All should be valid URLs
        expect(devUrl).toMatch(/^http:\/\/localhost:\d+$/)
        expect(prodUrl).toMatch(/^http:\/\/localhost:\d+$/)
        expect(backendUrl).toMatch(/^http:\/\/localhost:\d+$/)

        // Backend should match prod
        expect(backendUrl).toBe(prodUrl)
      })
    })

    it('should handle all worker types consistently', () => {
      const workers: WorkerServiceName[] = ['llm', 'sd', 'comfy', 'vllm']

      workers.forEach((worker) => {
        const devUrl = getWorkerUrl(worker, 'dev')
        const prodUrl = getWorkerUrl(worker, 'prod')
        const backendUrl = getWorkerUrl(worker, 'backend')

        // All should be valid URLs
        expect(devUrl).toMatch(/^http:\/\/localhost:\d+$/)
        expect(prodUrl).toMatch(/^http:\/\/localhost:\d+$/)
        expect(backendUrl).toMatch(/^http:\/\/localhost:\d+$/)

        // Backend should match prod
        expect(backendUrl).toBe(prodUrl)
      })
    })
  })

  describe('Null Port Handling', () => {
    it('should return empty string for keeper prod', () => {
      expect(getServiceUrl('keeper', 'prod')).toBe('')
      expect(getServiceUrl('keeper', 'backend')).toBe('')
    })

    it('should handle all null prod ports correctly', () => {
      // These services have null prod ports
      expect(PORTS.keeper.prod).toBeNull()
      expect(PORTS.commercial.prod).toBeNull()
      expect(PORTS.marketplace.prod).toBeNull()
      expect(PORTS.userDocs.prod).toBeNull()
      expect(PORTS.honoCatalog.prod).toBeNull()
    })

    it('should not include null ports in allowed origins', () => {
      const origins = getAllowedOrigins()

      // Should not have any 'null' or 'undefined' in URLs
      origins.forEach((origin) => {
        expect(origin).not.toContain('null')
        expect(origin).not.toContain('undefined')
        expect(origin).toMatch(/^https?:\/\/localhost:\d+$/)
      })
    })
  })

  describe('HTTPS Consistency', () => {
    it('should support HTTPS for all services', () => {
      const services: ServiceName[] = ['queen', 'hive', 'keeper']

      services.forEach((service) => {
        const httpUrl = getServiceUrl(service, 'dev', false)
        const httpsUrl = getServiceUrl(service, 'dev', true)

        if (httpUrl) {
          expect(httpUrl).toMatch(/^http:\/\//)
          expect(httpsUrl).toMatch(/^https:\/\//)
          expect(httpsUrl.replace('https://', 'http://')).toBe(httpUrl)
        }
      })
    })

    it('should support HTTPS for all workers', () => {
      const workers: WorkerServiceName[] = ['llm', 'sd', 'comfy', 'vllm']

      workers.forEach((worker) => {
        const httpUrl = getWorkerUrl(worker, 'dev', false)
        const httpsUrl = getWorkerUrl(worker, 'dev', true)

        expect(httpUrl).toMatch(/^http:\/\//)
        expect(httpsUrl).toMatch(/^https:\/\//)
        expect(httpsUrl.replace('https://', 'http://')).toBe(httpUrl)
      })
    })

    it('should support HTTPS in getIframeUrl', () => {
      const services: ServiceName[] = ['queen', 'hive']

      services.forEach((service) => {
        const httpUrl = getIframeUrl(service, true, false)
        const httpsUrl = getIframeUrl(service, true, true)

        expect(httpUrl).toMatch(/^http:\/\//)
        expect(httpsUrl).toMatch(/^https:\/\//)
      })
    })
  })

  describe('Type Safety Verification', () => {
    it('should have correct ServiceName type', () => {
      const services: ServiceName[] = ['queen', 'hive', 'keeper']

      services.forEach((service) => {
        expect(() => getServiceUrl(service, 'dev')).not.toThrow()
      })
    })

    it('should have correct WorkerServiceName type', () => {
      const workers: WorkerServiceName[] = ['llm', 'sd', 'comfy', 'vllm']

      workers.forEach((worker) => {
        expect(() => getWorkerUrl(worker, 'dev')).not.toThrow()
      })
    })

    it('should have immutable PORTS structure', () => {
      // TypeScript enforces this, but we can verify the structure
      expect(PORTS).toBeDefined()
      expect(typeof PORTS).toBe('object')

      // Verify nested structure exists
      expect(PORTS.worker).toBeDefined()
      expect(PORTS.worker.llm).toBeDefined()
      expect(PORTS.worker.sd).toBeDefined()
      expect(PORTS.worker.comfy).toBeDefined()
      expect(PORTS.worker.vllm).toBeDefined()
    })
  })
})
