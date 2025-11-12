// API Routes for Worker Catalog
// TEAM-453: Added input validation and cache headers
// TEAM-481: Env type is globally available from worker-configuration.d.ts

import { Hono } from 'hono'
import { WORKERS } from './data'
import { validateWorkerId } from './middleware/validation'

export const routes = new Hono<{ Bindings: Env }>()

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// WORKER CATALOG ENDPOINTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GET /workers
 * List all available worker variants
 */
routes.get('/workers', (c) => {
  return c.json(
    { workers: WORKERS },
    200,
    {
      'Cache-Control': 'public, max-age=3600, s-maxage=3600', // 1 hour
      'CDN-Cache-Control': 'max-age=3600',
    }
  )
})

/**
 * GET /workers/:id
 * Get a specific worker by ID
 */
routes.get('/workers/:id', validateWorkerId(), (c) => {
  const id = c.req.param('id')
  const worker = WORKERS.find((w) => w.id === id)

  if (!worker) {
    return c.json({ error: 'Worker not found' }, 404)
  }

  return c.json(
    worker,
    200,
    {
      'Cache-Control': 'public, max-age=3600, s-maxage=3600', // 1 hour
      'CDN-Cache-Control': 'max-age=3600',
    }
  )
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PKGBUILD ENDPOINTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/**
 * GET /workers/:id/PKGBUILD
 * Serve PKGBUILD file for a specific worker
 */
routes.get('/workers/:id/PKGBUILD', validateWorkerId(), async (c) => {
  const id = c.req.param('id')

  // Verify worker exists
  const worker = WORKERS.find((w) => w.id === id)
  if (!worker) {
    return c.json({ error: 'Worker not found' }, 404)
  }

  try {
    // TEAM-453: Check if ASSETS binding exists (not available in test environment)
    if (!c.env?.ASSETS) {
      return c.json({ error: 'ASSETS binding not available in test environment' }, 500)
    }

    // Fetch PKGBUILD from assets
    const pkgbuild = await c.env.ASSETS.fetch(new Request(`http://placeholder/pkgbuilds/${id}.PKGBUILD`))

    if (!pkgbuild.ok) {
      return c.json({ error: 'PKGBUILD not found' }, 404)
    }

    return new Response(pkgbuild.body, {
      headers: {
        'Content-Type': 'text/plain',
        'Cache-Control': 'public, max-age=3600',
      },
    })
  } catch (error) {
    console.error(`Failed to fetch PKGBUILD for ${id}:`, error)
    return c.json({ error: 'Failed to fetch PKGBUILD' }, 500)
  }
})
