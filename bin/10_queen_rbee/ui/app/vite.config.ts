import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import topLevelAwait from 'vite-plugin-top-level-await'
import wasm from 'vite-plugin-wasm'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 7834, // queen-rbee UI dev server
    strictPort: true,
  },
  plugins: [
    tailwindcss(), // Official Tailwind v4 Vite plugin (must be first)
    wasm(),
    topLevelAwait(),
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  optimizeDeps: {
    exclude: ['@rbee/queen-rbee-sdk'], // TEAM-375: Exclude WASM package from pre-bundling
  },
  build: {
    cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
  },
  define: {
    'process.env': {}, // Polyfill for libraries that check process.env
  },
})
