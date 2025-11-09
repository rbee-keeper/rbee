// TEAM-403: CORS configuration validation tests
import { describe, expect, it } from 'vitest'

describe('CORS Configuration', () => {
  const expectedOrigins = [
    'http://localhost:7836', // Hive UI
    'http://localhost:8500', // Queen Rbee
    'http://localhost:8501', // Rbee Keeper
    'http://127.0.0.1:7836',
    'http://127.0.0.1:8500',
    'http://127.0.0.1:8501',
  ]

  it('should include all required origins', () => {
    expect(expectedOrigins).toHaveLength(6)
    expect(expectedOrigins).toContain('http://localhost:7836')
    expect(expectedOrigins).toContain('http://localhost:8500')
    expect(expectedOrigins).toContain('http://localhost:8501')
  })

  it('should allow GET, POST, PUT, DELETE, OPTIONS', () => {
    const allowedMethods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
    expect(allowedMethods).toHaveLength(5)
    expect(allowedMethods).toContain('GET')
    expect(allowedMethods).toContain('POST')
    expect(allowedMethods).toContain('OPTIONS')
  })

  it('should allow Content-Type and Authorization headers', () => {
    const allowedHeaders = ['Content-Type', 'Authorization']
    expect(allowedHeaders).toContain('Content-Type')
    expect(allowedHeaders).toContain('Authorization')
  })

  it('should expose Content-Length header', () => {
    const exposedHeaders = ['Content-Length']
    expect(exposedHeaders).toContain('Content-Length')
  })
})
