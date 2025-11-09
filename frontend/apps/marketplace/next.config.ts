import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // TEAM-462: Static export for Cloudflare Pages
  output: 'export',
  
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
  // Turbopack disabled: OpenNext doesn't support Turbopack builds
  // Use webpack for production builds to ensure Cloudflare Workers compatibility
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
