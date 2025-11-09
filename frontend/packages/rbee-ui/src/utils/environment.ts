// TEAM-421: Environment detection for marketplace components
// Distinguishes between Tauri (rbee-keeper) and Next.js SSG (marketplace)

/**
 * Detect if running in Tauri environment (rbee-keeper)
 *
 * Tauri injects `window.__TAURI__` object at runtime.
 * This is the most reliable way to detect Tauri.
 */
export function isTauriEnvironment(): boolean {
  if (typeof window === 'undefined') return false
  return '__TAURI__' in window
}

/**
 * Detect if running in Next.js environment (marketplace)
 *
 * Next.js sets `process.env.NEXT_RUNTIME` during build.
 * At runtime in browser, we check for Next.js-specific globals.
 */
export function isNextJsEnvironment(): boolean {
  if (typeof window === 'undefined') {
    // SSR/SSG build time - check Node.js environment
    return typeof process !== 'undefined' && process.env.NEXT_RUNTIME !== undefined
  }

  // Browser runtime - check for Next.js router
  return typeof window !== 'undefined' && 'next' in window && !isTauriEnvironment() // Exclude Tauri
}

/**
 * Detect if running in SSR/SSG context (server-side)
 */
export function isServerSide(): boolean {
  return typeof window === 'undefined'
}

/**
 * Detect if running in browser (client-side)
 */
export function isClientSide(): boolean {
  return typeof window !== 'undefined'
}

/**
 * Environment types
 */
export type Environment =
  | 'tauri' // Tauri app (rbee-keeper)
  | 'nextjs-ssg' // Next.js static site (marketplace)
  | 'nextjs-ssr' // Next.js server-side rendering
  | 'browser' // Generic browser (not Tauri, not Next.js)
  | 'server' // Server-side (SSR/SSG build)

/**
 * Get current environment
 *
 * Priority order:
 * 1. Server-side (SSR/SSG) - no window object
 * 2. Tauri - has window.__TAURI__
 * 3. Next.js - has window.next or NEXT_RUNTIME
 * 4. Browser - fallback
 */
export function getEnvironment(): Environment {
  // Server-side (SSR/SSG build time)
  if (isServerSide()) {
    if (isNextJsEnvironment()) {
      return 'nextjs-ssr'
    }
    return 'server'
  }

  // Client-side (browser runtime)
  if (isTauriEnvironment()) {
    return 'tauri'
  }

  if (isNextJsEnvironment()) {
    return 'nextjs-ssg'
  }

  return 'browser'
}

/**
 * Check if actions should use Tauri commands (direct invocation)
 *
 * Returns true if:
 * - Running in Tauri environment
 * - Can directly invoke Rust commands
 */
export function shouldUseTauriCommands(): boolean {
  return isTauriEnvironment()
}

/**
 * Check if actions should use deep links (rbee:// protocol)
 *
 * Returns true if:
 * - Running in browser (Next.js SSG or generic browser)
 * - NOT in Tauri (Tauri should use direct commands)
 * - NOT server-side (can't open deep links during SSG)
 */
export function shouldUseDeepLinks(): boolean {
  return isClientSide() && !isTauriEnvironment()
}

/**
 * Check if we can perform actions (download, install)
 *
 * Returns false during SSR/SSG build time.
 * Returns true in browser (either via Tauri commands or deep links).
 */
export function canPerformActions(): boolean {
  return isClientSide()
}

/**
 * Get action strategy for current environment
 */
export type ActionStrategy = 'tauri-command' | 'deep-link' | 'none'

export function getActionStrategy(): ActionStrategy {
  if (!canPerformActions()) return 'none'
  if (shouldUseTauriCommands()) return 'tauri-command'
  if (shouldUseDeepLinks()) return 'deep-link'
  return 'none'
}

/**
 * Debug helper - get environment info
 */
export function getEnvironmentInfo() {
  return {
    environment: getEnvironment(),
    isServerSide: isServerSide(),
    isClientSide: isClientSide(),
    isTauri: isTauriEnvironment(),
    isNextJs: isNextJsEnvironment(),
    actionStrategy: getActionStrategy(),
    canPerformActions: canPerformActions(),
    shouldUseTauriCommands: shouldUseTauriCommands(),
    shouldUseDeepLinks: shouldUseDeepLinks(),
  }
}
