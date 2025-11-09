// TEAM-457: Re-export shared environment configuration
// All environment logic is now in @rbee/env-config package

export type { Environment, URLs } from '@rbee/env-config'
export { corsOrigins, env, isDev, isProd, PORTS, urls } from '@rbee/env-config'
