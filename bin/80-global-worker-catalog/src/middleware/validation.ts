// TEAM-453: Input validation middleware
// Validates and sanitizes user input

import type { Context, Next } from 'hono'

/**
 * Validate worker ID parameter
 * Prevents path traversal and injection attacks
 */
export function validateWorkerId() {
  return async (c: Context, next: Next) => {
    const id = c.req.param('id')

    // Check if ID exists
    if (!id) {
      return c.json({ error: 'Worker ID is required' }, 400)
    }

    // Validate format: lowercase alphanumeric and hyphens only
    if (!/^[a-z0-9-]+$/.test(id)) {
      return c.json(
        {
          error: 'Invalid worker ID format',
          message: 'Worker ID must contain only lowercase letters, numbers, and hyphens',
        },
        400,
      )
    }

    // Prevent path traversal
    if (id.includes('..') || id.includes('/') || id.includes('\\')) {
      return c.json({ error: 'Invalid worker ID' }, 400)
    }

    // Length check (reasonable limit)
    if (id.length > 100) {
      return c.json({ error: 'Worker ID too long' }, 400)
    }

    await next()
  }
}

/**
 * Validate request method
 * Only allow GET and OPTIONS
 */
export function validateMethod() {
  return async (c: Context, next: Next) => {
    const method = c.req.method

    if (!['GET', 'OPTIONS', 'HEAD'].includes(method)) {
      return c.json(
        {
          error: 'Method not allowed',
          message: `${method} is not supported. Only GET requests are allowed.`,
        },
        405,
      )
    }

    await next()
  }
}
