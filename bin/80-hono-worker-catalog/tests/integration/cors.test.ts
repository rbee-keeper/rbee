// TEAM-403: CORS integration tests
import { describe, it, expect } from 'vitest'
import app from '../../src/index'

describe('CORS Integration', () => {
  it('should allow requests from Hive UI (localhost:7836)', async () => {
    const res = await app.request('/workers', {
      headers: {
        'Origin': 'http://localhost:7836'
      }
    })
    
    expect(res.status).toBe(200)
    const corsHeader = res.headers.get('Access-Control-Allow-Origin')
    expect(corsHeader).toBeTruthy()
  })

  it('should allow requests from Queen (localhost:8500)', async () => {
    const res = await app.request('/workers', {
      headers: {
        'Origin': 'http://localhost:8500'
      }
    })
    
    expect(res.status).toBe(200)
    const corsHeader = res.headers.get('Access-Control-Allow-Origin')
    expect(corsHeader).toBeTruthy()
  })

  it('should allow requests from Keeper (localhost:8501)', async () => {
    const res = await app.request('/workers', {
      headers: {
        'Origin': 'http://localhost:8501'
      }
    })
    
    expect(res.status).toBe(200)
    const corsHeader = res.headers.get('Access-Control-Allow-Origin')
    expect(corsHeader).toBeTruthy()
  })

  it('should handle OPTIONS preflight requests', async () => {
    const res = await app.request('/workers', {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:7836',
        'Access-Control-Request-Method': 'GET'
      }
    })
    
    // OPTIONS should return 204 or 200
    expect([200, 204]).toContain(res.status)
  })

  it('should include CORS headers in responses', async () => {
    const res = await app.request('/workers', {
      headers: {
        'Origin': 'http://localhost:7836'
      }
    })
    
    expect(res.status).toBe(200)
    
    // Check for CORS headers
    const allowOrigin = res.headers.get('Access-Control-Allow-Origin')
    expect(allowOrigin).toBeTruthy()
  })
})
