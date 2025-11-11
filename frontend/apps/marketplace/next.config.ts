import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // TEAM-462: Static export for Cloudflare Pages
  // TEAM-464: Only use static export in production, allow dynamic in dev
  output: process.env.NODE_ENV === 'production' ? 'export' : undefined,

  // TEAM-463: Disable image optimization for static export
  images: {
    unoptimized: true,
  },

  // TEAM-421: Enable WASM support for marketplace-sdk
  webpack: (config, { isServer }) => {
    // Add WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    }

    // Handle .wasm files
    config.module.rules.push({
      test: /\.wasm$/,
      type: 'webassembly/async',
    })

    // Ensure WASM files are copied for server-side rendering
    if (isServer) {
      config.output.webassemblyModuleFilename = 'chunks/[id].wasm'
      config.plugins.push(new WasmChunksFixPlugin())
    }

    return config
  },
  // TEAM-XXX: SSG output - no server-side rendering
}

// Fix for WASM chunks in Next.js
class WasmChunksFixPlugin {
  apply(compiler: any) {
    compiler.hooks.thisCompilation.tap('WasmChunksFixPlugin', (compilation: any) => {
      compilation.hooks.processAssets.tap({ name: 'WasmChunksFixPlugin' }, () => {
        // No-op: Just ensures WASM files are included
      })
    })
  }
}

export default nextConfig
