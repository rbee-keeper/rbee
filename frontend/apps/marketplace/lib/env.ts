// TEAM-457: Re-export shared environment configuration
// All environment logic is now in @rbee/env-config package

export { env, urls, isDev, isProd, PORTS, corsOrigins } from '@rbee/env-config'
export type { Environment, URLs } from '@rbee/env-config'
