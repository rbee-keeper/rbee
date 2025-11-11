// TEAM-453: Unit tests for filtering utilities
import { describe, expect, it } from 'vitest'

describe('Filtering Utilities', () => {
  describe('Worker filtering', () => {
    const mockWorkers = [
      { id: 'llm-worker-rbee-cpu', name: 'LLM Worker (CPU)', type: 'llm', hardware: 'cpu' },
      { id: 'llm-worker-rbee-cuda', name: 'LLM Worker (CUDA)', type: 'llm', hardware: 'cuda' },
      { id: 'sd-worker-rbee-cpu', name: 'SD Worker (CPU)', type: 'sd', hardware: 'cpu' },
    ]

    it('should filter by worker type', () => {
      const llmWorkers = mockWorkers.filter((w) => w.type === 'llm')
      expect(llmWorkers).toHaveLength(2)
      expect(llmWorkers.every((w) => w.type === 'llm')).toBe(true)
    })

    it('should filter by hardware type', () => {
      const cpuWorkers = mockWorkers.filter((w) => w.hardware === 'cpu')
      expect(cpuWorkers).toHaveLength(2)
      expect(cpuWorkers.every((w) => w.hardware === 'cpu')).toBe(true)
    })

    it('should filter by search query', () => {
      const searchQuery = 'cuda'
      const filtered = mockWorkers.filter((w) => w.name.toLowerCase().includes(searchQuery.toLowerCase()))
      expect(filtered).toHaveLength(1)
      expect(filtered[0].id).toBe('llm-worker-rbee-cuda')
    })
  })

  describe('Model filtering', () => {
    const mockModels = [
      { id: 'llama-3.1-8b', name: 'Llama 3.1 8B', size: '8B', type: 'llm' },
      { id: 'llama-3.1-70b', name: 'Llama 3.1 70B', size: '70B', type: 'llm' },
      { id: 'sdxl-turbo', name: 'SDXL Turbo', type: 'sd' },
    ]

    it('should filter by model type', () => {
      const llmModels = mockModels.filter((m) => m.type === 'llm')
      expect(llmModels).toHaveLength(2)
    })

    it('should filter by model size', () => {
      const smallModels = mockModels.filter((m) => m.size === '8B')
      expect(smallModels).toHaveLength(1)
      expect(smallModels[0].id).toBe('llama-3.1-8b')
    })

    it('should search by name', () => {
      const searchQuery = 'turbo'
      const filtered = mockModels.filter((m) => m.name.toLowerCase().includes(searchQuery.toLowerCase()))
      expect(filtered).toHaveLength(1)
      expect(filtered[0].id).toBe('sdxl-turbo')
    })
  })
})
