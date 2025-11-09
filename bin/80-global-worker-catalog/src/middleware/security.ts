// TEAM-453: Security middleware
// Adds security headers and protections

import type { Context, Next } from 'hono'

/**
 * Security headers middleware
 * Adds comprehensive security headers to all responses
 */
export function securityHeaders() {
  return async (c: Context, next: Next) => {
    await next()

    // Prevent MIME type sniffing
    c.header('X-Content-Type-Options', 'nosniff')

    // Prevent clickjacking
    c.header('X-Frame-Options', 'DENY')

    // XSS protection (legacy but still useful)
    c.header('X-XSS-Protection', '1; mode=block')

    // Referrer policy
    c.header('Referrer-Policy', 'strict-origin-when-cross-origin')

    // Permissions policy (disable unnecessary features)
    c.header('Permissions-Policy', 'geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=()')

    // Content Security Policy (strict for API)
    c.header('Content-Security-Policy', "default-src 'none'; frame-ancestors 'none'")

    // Remove server header
    c.header('Server', 'rbee-gwc')
  }
}

/**
 * Request logging middleware
 * Logs all requests for monitoring and debugging
 */
export function requestLogger() {
  return async (c: Context, next: Next) => {
    const start = Date.now()
    const method = c.req.method
    const path = c.req.path
    const ip = c.req.header('CF-Connecting-IP') || c.req.header('X-Forwarded-For') || 'unknown'
    const userAgent = c.req.header('User-Agent') || 'unknown'

    await next()

    const duration = Date.now() - start
    const status = c.res.status

    console.log(`[${method}] ${path} ${status} ${duration}ms (${ip}) ${userAgent.substring(0, 50)}`)
  }
}

/**
 * Error handler middleware
 * Catches errors and returns safe error responses
 */
export function errorHandler() {
  return async (c: Context, next: Next) => {
    try {
      await next()
    } catch (error) {
      console.error('[ERROR]', error)

      // Don't leak internal error details
      return c.json(
        {
          error: 'Internal server error',
          message: 'An unexpected error occurred. Please try again later.',
          timestamp: new Date().toISOString(),
        },
        500,
      )
    }
  }
}
