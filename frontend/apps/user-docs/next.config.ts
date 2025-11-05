import type { NextConfig } from 'next'
import nextra from 'nextra'

const nextConfig: NextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  // Transpile workspace packages
  transpilePackages: ['@rbee/ui'],
  experimental: {
    optimizePackageImports: ['@rbee/ui'],
  },
}

const withNextra = nextra({
  // Nextra configuration options
  defaultShowCopyCode: true,
  search: {
    codeblocks: false,
  },
})

export default withNextra(nextConfig)

// added by create cloudflare to enable calling `getCloudflareContext()` in `next dev`
import { initOpenNextCloudflareForDev } from '@opennextjs/cloudflare'

initOpenNextCloudflareForDev()
