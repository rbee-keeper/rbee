// Worker Catalog Service
// Provides metadata and PKGBUILD files for worker installation
// TEAM-453: Added comprehensive security defenses

import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { routes } from './routes'
import { errorHandler, requestLogger, securityHeaders } from './middleware/security'
import { validateMethod } from './middleware/validation'

const app = new Hono<{ Bindings: Env }>()

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SECURITY MIDDLEWARE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-453: Comprehensive security defenses

// Error handling (must be first)
app.use('/*', errorHandler())

// Request logging
app.use('/*', requestLogger())

// Security headers
app.use('/*', securityHeaders())

// Method validation
app.use('/*', validateMethod())

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CORS MIDDLEWARE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-453: Allow production domains for marketplace SSG builds

app.use(
  '/*',
  cors({
    origin: [
      // Development
      'http://localhost:7845', // Hive UI
      'http://localhost:7822', // Commercial
      'http://localhost:7823', // Marketplace
      'http://localhost:7844', // Queen Rbee UI
      'http://localhost:7843', // Rbee Keeper
      'http://127.0.0.1:7845',
      'http://127.0.0.1:7822',
      'http://127.0.0.1:7823',
      'http://127.0.0.1:7844',
      'http://127.0.0.1:7843',
      
      // Production
      'https://marketplace.rbee.dev',
      'https://rbee.dev',
      'https://docs.rbee.dev',
    ],
    allowMethods: ['GET', 'OPTIONS', 'HEAD'], // Read-only
    allowHeaders: ['Content-Type'],
    exposeHeaders: ['Content-Length', 'Cache-Control'],
    maxAge: 3600, // 1 hour
    credentials: false, // No cookies needed
  }),
)

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ROOT REDIRECT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Redirect root to marketplace workers page
app.get('/', (c) => {
  return c.redirect('https://marketplace.rbee.dev/workers', 301)
})

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ROUTES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.route('/', routes)

// Health check
app.get('/health', (c) => {
  return c.json({
    status: 'ok',
    service: 'worker-catalog',
    version: '0.1.0',
  })
})

export default app
