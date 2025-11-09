import type { NextConfig } from 'next'
import nextra from 'nextra'

const nextConfig: NextConfig = {
  // TEAM-427: Static export for Cloudflare Pages
  output: 'export',
  
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

// TEAM-427: OpenNext not needed for static export (output: 'export')
// Only initialize for dev mode without top-level await
if (process.env.NODE_ENV === 'development') {
  import('@opennextjs/cloudflare').then(({ initOpenNextCloudflareForDev }) => {
    initOpenNextCloudflareForDev()
  })
}
