// TEAM-482: Global Worker Catalog (GWC) types
// Matches types from bin/80-global-worker-catalog/src/types.ts

/**
 * Worker type (backend acceleration)
 */
export type WorkerType = 'cpu' | 'cuda' | 'metal' | 'rocm'

/**
 * Platform (operating system)
 */
export type Platform = 'linux' | 'macos' | 'windows'

/**
 * Architecture (CPU instruction set)
 */
export type Architecture = 'x86_64' | 'aarch64'

/**
 * Worker implementation type
 */
export type WorkerImplementation = 'rust' | 'python' | 'cpp'

/**
 * Build system type
 */
export type BuildSystem = 'cargo' | 'cmake' | 'pip' | 'npm'

/**
 * Build variant for a worker
 */
export interface BuildVariant {
  backend: WorkerType
  platforms: Platform[]
  architectures: Architecture[]
  pkgbuildUrl: string
  pkgbuildUrlGit: string
  homebrewFormula: string
  homebrewFormulaGit: string
  build: {
    features?: string[]
    profile?: string
    flags?: string[]
  }
  depends: string[]
  makedepends: string[]
  binaryName: string
  installPath: string
}

/**
 * Worker catalog entry (from GWC API)
 */
export interface GWCWorker {
  id: string
  implementation: WorkerImplementation
  version: string
  name: string
  description: string
  license: string
  buildSystem: BuildSystem
  source: {
    type: 'git' | 'tarball'
    url: string
    branch?: string
    tag?: string
    path?: string
  }
  variants: BuildVariant[]
  supportedFormats: string[]
  maxContextLength?: number
  supportsStreaming: boolean
  supportsBatching: boolean
}

/**
 * GWC API response for list endpoint
 */
export interface GWCListWorkersResponse {
  workers: GWCWorker[]
}

/**
 * GWC list workers parameters (for future filtering)
 */
export interface GWCListWorkersParams {
  backend?: WorkerType
  platform?: Platform
  limit?: number
}
