/**
 * PROGRAMMATIC SOURCE OF TRUTH for all port configurations
 *
 * CANONICAL SOURCE: /PORT_CONFIGURATION.md
 *
 * CRITICAL: When adding a new service:
 * 1. Update PORT_CONFIGURATION.md (canonical source)
 * 2. Update this file to match
 * 3. Run `pnpm generate:rust` (if applicable)
 * 4. Update backend Cargo.toml default port
 *
 * TEAM-351: Shared port configuration
 * TEAM-457: Added commercial, marketplace, user-docs, hono-catalog
 * TEAM-XXX: Added comfy-worker, vllm-worker ports; added getWorkerUrl() helper
 * TEAM-XXX: Added admin worker (8788, install.rbee.dev)
 *
 * @packageDocumentation
 */

// TEAM-457: Helper to get port from environment variable or use default
function getPort(envVar: string, defaultPort: number): number {
  // @ts-expect-error - Vite's import.meta.env is available at runtime
  if (typeof import.meta !== 'undefined' && import.meta.env) {
    // @ts-expect-error
    const envValue = import.meta.env[envVar]
    if (envValue) {
      const parsed = parseInt(envValue, 10)
      if (!isNaN(parsed)) {
        return parsed
      }
    }
  }
  return defaultPort
}

/**
 * Port configuration for each service
 * Source: PORT_CONFIGURATION.md
 */
export const PORTS = {
  // Backend Services (HTTP APIs)
  queen: {
    dev: getPort('VITE_QUEEN_UI_DEV_PORT', 7834), // Vite dev server
    prod: getPort('VITE_QUEEN_PORT', 7833), // Embedded in backend
    backend: getPort('VITE_QUEEN_PORT', 7833), // Backend HTTP server
  },
  hive: {
    dev: getPort('VITE_HIVE_UI_DEV_PORT', 7836),
    prod: getPort('VITE_HIVE_PORT', 7835),
    backend: getPort('VITE_HIVE_PORT', 7835),
  },
  // TEAM-XXX: Worker backend ports are DYNAMIC (assigned by hive starting from 8080)
  // Only dev ports are fixed for local development
  // In production, query the hive for actual worker URLs
  worker: {
    llm: {
      dev: 7837,
      prod: null, // Dynamic - assigned by hive
      backend: null, // Dynamic - assigned by hive
    },
    sd: {
      dev: 5174,
      prod: null, // Dynamic - assigned by hive
      backend: null, // Dynamic - assigned by hive
    },
    comfy: {
      dev: 7838,
      prod: null, // Dynamic - assigned by hive
      backend: null, // Dynamic - assigned by hive
    },
    vllm: {
      dev: 7839,
      prod: null, // Dynamic - assigned by hive
      backend: null, // Dynamic - assigned by hive
    },
  },

  // Frontend Services (Development)
  keeper: {
    dev: getPort('VITE_KEEPER_DEV_PORT', 5173),
    prod: null, // Tauri app, no HTTP port
  },
  commercial: {
    dev: getPort('VITE_COMMERCIAL_PORT', 7822),
    prod: null, // Deployed to Cloudflare
  },
  marketplace: {
    dev: getPort('VITE_MARKETPLACE_PORT', 7823),
    prod: null, // Deployed to Cloudflare
  },
  userDocs: {
    dev: getPort('VITE_USER_DOCS_PORT', 7811),
    prod: null, // Deployed to Cloudflare
  },

  // Storybooks
  storybook: {
    rbeeUi: getPort('VITE_RBEE_UI_STORYBOOK_PORT', 6006),
    commercial: getPort('VITE_COMMERCIAL_STORYBOOK_PORT', 6007),
  },

  // Cloudflare Workers
  honoCatalog: {
    dev: getPort('VITE_HONO_CATALOG_PORT', 8787),
    prod: null, // Deployed to Cloudflare
  },
  admin: {
    dev: getPort('VITE_ADMIN_PORT', 8788),
    prod: null, // Deployed to Cloudflare
  },
} as const

// TEAM-457: ServiceName for functions that expect simple {dev, prod, backend} structure
// Excludes: worker (nested), commercial/marketplace/userDocs/honoCatalog (no backend), storybook (no backend)
export type ServiceName = 'queen' | 'hive' | 'keeper'

// Worker service names for nested worker structure
export type WorkerServiceName = 'llm' | 'sd' | 'comfy' | 'vllm'

/**
 * Generate allowed origins for postMessage listener
 *
 * @param includeHttps - Include HTTPS variants for production (default: false)
 * @returns Array of unique allowed origins
 */
export function getAllowedOrigins(includeHttps = false): string[] {
  const origins = new Set<string>()

  // Queen and Hive
  const simpleServices = [PORTS.queen, PORTS.hive]
  for (const ports of simpleServices) {
    if (ports.dev !== null) {
      origins.add(`http://localhost:${ports.dev}`)
    }
    if (ports.prod !== null) {
      origins.add(`http://localhost:${ports.prod}`)
      if (includeHttps) {
        origins.add(`https://localhost:${ports.prod}`)
      }
    }
  }

  // Workers (nested structure)
  const workerTypes = [PORTS.worker.llm, PORTS.worker.sd, PORTS.worker.comfy, PORTS.worker.vllm]
  for (const ports of workerTypes) {
    if (ports.dev !== null) {
      origins.add(`http://localhost:${ports.dev}`)
    }
    if (ports.prod !== null) {
      origins.add(`http://localhost:${ports.prod}`)
      if (includeHttps) {
        origins.add(`https://localhost:${ports.prod}`)
      }
    }
  }

  return Array.from(origins).sort()
}

// ============================================================
// BUG FIX: TEAM-374 | Iframe not loading - reverted /dev proxy
// ============================================================
// SUSPICION:
// - Thought using /dev proxy would avoid CORS issues
// - Expected proxy to correctly forward all Vite requests
//
// INVESTIGATION:
// - Changed getIframeUrl to use /dev proxy (http://localhost:7833/dev)
// - Observed: HTML loads but JS modules get 404 errors
// - Found: Browser tries to load /@react-refresh from backend, not through proxy
// - Root cause: Vite's module resolution doesn't work through the proxy
//
// ROOT CAUSE:
// - When iframe loads /dev, HTML comes through correctly
// - But HTML contains <script src="/@react-refresh"> etc
// - Browser resolves these as absolute paths from backend
// - Backend has no /@react-refresh route - only /dev/@react-refresh would work
// - Vite needs to run on its own origin for HMR and module resolution
//
// FIX:
// - Reverted to direct Vite dev server URLs (ports 7834/7836)
// - Dev mode: http://localhost:7834 (Queen) / http://localhost:7836 (Hive)
// - Prod mode: Still uses backend URLs (7833/7835)
// - CORS is not an issue because iframes are same-origin (localhost)
//
// TESTING:
// - Verified Queen iframe loads at http://localhost:7834
// - Verified Hive iframe loads at http://localhost:7836
// - Checked browser console - no 404 errors for JS modules
// - Confirmed HMR (hot module reload) works
// ============================================================

/**
 * Get iframe URL for embedding services
 *
 * @param service - Service name
 * @param isDev - Whether in development mode
 * @param useHttps - Whether to use HTTPS (default: false)
 * @returns URL string or empty string if service has no HTTP port
 * @throws Error if service doesn't support iframe embedding
 */
export function getIframeUrl(service: ServiceName, isDev: boolean, useHttps = false): string {
  const ports = PORTS[service]
  const port = isDev ? ports.dev : ports.prod

  if (service === 'keeper' && !isDev) {
    throw new Error('Keeper service has no production HTTP port (Tauri app). Use dev mode or check service name.')
  }

  if (port === null) {
    return ''
  }

  const protocol = useHttps ? 'https' : 'http'

  // TEAM-374: In dev mode, load directly from Vite dev server
  // Do NOT use /dev proxy - breaks Vite's module resolution
  return `${protocol}://localhost:${port}`
}

/**
 * Get parent origin for postMessage
 *
 * @param currentPort - Current window port
 * @returns Origin string or '*' for wildcard (Tauri app)
 */
export function getParentOrigin(currentPort: number): string {
  const isDevPort =
    currentPort === PORTS.queen.dev ||
    currentPort === PORTS.hive.dev ||
    currentPort === PORTS.worker.llm.dev ||
    currentPort === PORTS.worker.sd.dev ||
    currentPort === PORTS.worker.comfy.dev ||
    currentPort === PORTS.worker.vllm.dev ||
    currentPort === PORTS.keeper.dev

  return isDevPort ? `http://localhost:${PORTS.keeper.dev}` : '*'
}

/**
 * Get service URL for HTTP requests
 *
 * @param service - Service name
 * @param mode - 'dev' | 'prod' | 'backend'
 * @param useHttps - Use HTTPS instead of HTTP (default: false)
 * @returns URL string or empty string if port is null
 */
export function getServiceUrl(
  service: ServiceName,
  mode: 'dev' | 'prod' | 'backend' = 'dev',
  useHttps = false,
): string {
  const ports = PORTS[service]

  let port: number | null
  if (mode === 'backend') {
    port = 'backend' in ports ? (ports as any).backend : ports.prod
  } else {
    port = mode === 'dev' ? ports.dev : ports.prod
  }

  if (port === null) {
    return ''
  }

  const protocol = useHttps ? 'https' : 'http'
  return `${protocol}://localhost:${port}`
}

/**
 * Get worker service URL for HTTP requests
 *
 * @param worker - Worker type ('llm' | 'sd' | 'comfy' | 'vllm')
 * @param mode - 'dev' | 'prod' | 'backend'
 * @param useHttps - Use HTTPS instead of HTTP (default: false)
 * @returns URL string or empty string if port is null
 */
export function getWorkerUrl(
  worker: WorkerServiceName,
  mode: 'dev' | 'prod' | 'backend' = 'dev',
  useHttps = false,
): string {
  const ports = PORTS.worker[worker]

  let port: number | null
  if (mode === 'backend') {
    port = ports.backend
  } else {
    port = mode === 'dev' ? ports.dev : ports.prod
  }

  if (port === null) {
    return ''
  }

  const protocol = useHttps ? 'https' : 'http'
  return `${protocol}://localhost:${port}`
}
