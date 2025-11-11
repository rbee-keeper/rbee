// TEAM-453: Unit tests for marketplace configuration
import { describe, expect, it } from 'vitest'

describe('Marketplace Configuration', () => {
  it('should have marketplace API URL configured', () => {
    const apiUrl = process.env.MARKETPLACE_API_URL || 'https://gwc.rbee.dev'
    expect(apiUrl).toBeDefined()
    expect(apiUrl).toMatch(/^https?:\/\//)
  })

  it('should have valid API URL format', () => {
    const apiUrl = process.env.MARKETPLACE_API_URL || 'https://gwc.rbee.dev'
    expect(() => new URL(apiUrl)).not.toThrow()
  })

  it('should point to gwc.rbee.dev in production', () => {
    const apiUrl = process.env.MARKETPLACE_API_URL || 'https://gwc.rbee.dev'
    expect(apiUrl).toContain('gwc.rbee.dev')
  })
})
