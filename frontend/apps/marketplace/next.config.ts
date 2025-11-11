import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // TEAM-475: SSR enabled - no static export
  // Marketplace now uses Server-Side Rendering for real-time data fetching
  // Deployed to Cloudflare Pages with @cloudflare/next-on-pages
  
  // TEAM-463: Disable image optimization (not needed for marketplace)
  images: {
    unoptimized: true,
  },

  // TEAM-421: Enable WASM support for marketplace-sdk
  // TEAM-475: Updated for bundler target WASM (auto-initializes, works with webpack)
  webpack: (config) => {
    // Add WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    }

    return config
  },
}

export default nextConfig
