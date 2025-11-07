// TEAM-457: Environment configuration for Hono worker catalog
// Cloudflare Workers use a different env pattern (bindings)

import type { Env as CloudflareEnv } from '../worker-configuration'

/**
 * Get environment with automatic dev/prod detection
 * In Cloudflare Workers, env is passed via bindings
 */
export function getEnv(env: CloudflareEnv) {
  const isDev = env.ENVIRONMENT === 'development'
  const isProd = env.ENVIRONMENT === 'production'
  const isPreview = env.ENVIRONMENT === 'preview'

  return {
    // Environment flags
    isDev,
    isProd,
    isPreview,
    environment: env.ENVIRONMENT,

    // CORS configuration
    corsOrigin: env.CORS_ORIGIN,

    // Computed values
    allowedOrigins: isDev
      ? ['http://localhost:7823', 'http://localhost:7822', 'http://localhost:7811']
      : isPreview
        ? ['https://marketplace-preview.rbee.dev', 'https://preview.rbee.dev']
        : ['https://marketplace.rbee.dev', 'https://rbee.dev', 'https://docs.rbee.dev'],
  } as const
}

export type WorkerEnv = ReturnType<typeof getEnv>
