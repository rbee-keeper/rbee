import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
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
  // TEAM-450: Next.js 16 requires explicit turbopack config when webpack config exists
  // Empty config allows Turbopack to work with default settings
  turbopack: {},
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

// added by create cloudflare to enable calling `getCloudflareContext()` in `next dev`
import { initOpenNextCloudflareForDev } from '@opennextjs/cloudflare'

initOpenNextCloudflareForDev()
