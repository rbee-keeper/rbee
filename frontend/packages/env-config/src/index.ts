// TEAM-457: Shared environment configuration for all rbee apps
// Automatically detects dev/prod and provides consistent URL configuration
//
// HIERARCHY:
// 1. PORT_CONFIGURATION.md = Canonical source of truth (human-readable)
// 2. @rbee/shared-config = Programmatic source of truth (ports)
// 3. @rbee/env-config = Environment-aware wrapper (this file)

import { PORTS as SHARED_PORTS } from '@rbee/shared-config'

/**
 * Environment detection
 */
export const isDev = process.env.NODE_ENV === 'development'
export const isProd = process.env.NODE_ENV === 'production'
export const isTest = process.env.NODE_ENV === 'test'

/**
 * Port configuration (from @rbee/shared-config)
 * Source: PORT_CONFIGURATION.md → shared-config/src/ports.ts
 */
export const PORTS = {
  commercial: SHARED_PORTS.commercial.dev,
  marketplace: SHARED_PORTS.marketplace.dev,
  userDocs: SHARED_PORTS.userDocs.dev,
  honoCatalog: SHARED_PORTS.honoCatalog.dev,
  queen: SHARED_PORTS.queen.backend,
  hive: SHARED_PORTS.hive.backend,
  llmWorker: SHARED_PORTS.worker.llm.backend,
  sdWorker: SHARED_PORTS.worker.sd.backend,
} as const

/**
 * Production URLs (defaults)
 */
const PROD_URLS = {
  commercial: 'https://rbee.dev',
  marketplace: 'https://marketplace.rbee.dev',
  docs: 'https://docs.rbee.dev',
  github: 'https://github.com/veighnsche/llama-orch',
} as const

/**
 * Development URLs (auto-generated from ports)
 */
const DEV_URLS = {
  commercial: `http://localhost:${PORTS.commercial}`,
  marketplace: `http://localhost:${PORTS.marketplace}`,
  docs: `http://localhost:${PORTS.userDocs}`,
  github: PROD_URLS.github, // GitHub is always production
} as const

/**
 * Get URL based on environment with optional override
 */
function getUrl(key: keyof typeof PROD_URLS, envVar?: string): string {
  // 1. Check environment variable override
  if (envVar && process.env[envVar]) {
    return process.env[envVar]
  }

  // 2. Auto-detect based on NODE_ENV
  if (isDev) {
    return DEV_URLS[key]
  }

  // 3. Fall back to production
  return PROD_URLS[key]
}

/**
 * Environment configuration with automatic dev/prod detection
 */
export const env = {
  // URLs with automatic dev/prod switching
  commercialUrl: getUrl('commercial', 'NEXT_PUBLIC_SITE_URL'),
  marketplaceUrl: getUrl('marketplace', 'NEXT_PUBLIC_MARKETPLACE_URL'),
  docsUrl: getUrl('docs', 'NEXT_PUBLIC_DOCS_URL'),
  githubUrl: getUrl('github', 'NEXT_PUBLIC_GITHUB_URL'),

  // Contact emails (always production)
  legalEmail: process.env.NEXT_PUBLIC_LEGAL_EMAIL || 'legal@rbee.dev',
  supportEmail: process.env.NEXT_PUBLIC_SUPPORT_EMAIL || 'support@rbee.dev',

  // Environment flags
  isDev,
  isProd,
  isTest,
} as const

/**
 * URL helper functions for consistent URL generation
 */
export const urls = {
  // Main sites
  commercial: env.commercialUrl,
  marketplace: {
    home: env.marketplaceUrl,
    models: `${env.marketplaceUrl}/models`,
    llmModels: `${env.marketplaceUrl}/models`,
    sdModels: `${env.marketplaceUrl}/models?type=sd`,
    workers: `${env.marketplaceUrl}/workers`,
    llmWorkers: `${env.marketplaceUrl}/workers`,
    imageWorkers: `${env.marketplaceUrl}/workers?type=image`,
    model: (slug: string) => `${env.marketplaceUrl}/models/${slug}`,
    worker: (slug: string) => `${env.marketplaceUrl}/workers/${slug}`,
  },
  docs: env.docsUrl,
  github: {
    repo: env.githubUrl,
    docs: `${env.githubUrl}/tree/main/docs`,
    issues: `${env.githubUrl}/issues`,
    discussions: `${env.githubUrl}/discussions`,
  },
  contact: {
    legal: `mailto:${env.legalEmail}`,
    support: `mailto:${env.supportEmail}`,
  },
} as const

/**
 * CORS origins for API configuration
 */
export const corsOrigins = {
  development: [DEV_URLS.commercial, DEV_URLS.marketplace, DEV_URLS.docs],
  production: [PROD_URLS.commercial, PROD_URLS.marketplace, PROD_URLS.docs],
  all: [...new Set([...Object.values(DEV_URLS), ...Object.values(PROD_URLS)])],
} as const

// Build-time validation and logging (only in development, server-side only)
if (isDev && typeof globalThis !== 'undefined' && typeof (globalThis as any).window === 'undefined') {
  console.log('[env-config] Environment configuration loaded:', {
    NODE_ENV: process.env.NODE_ENV,
    commercialUrl: env.commercialUrl,
    marketplaceUrl: env.marketplaceUrl,
    docsUrl: env.docsUrl,
    githubUrl: env.githubUrl,
  })
}

// Export types for TypeScript
export type Environment = typeof env
export type URLs = typeof urls
