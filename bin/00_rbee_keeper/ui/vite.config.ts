// TEAM-294: Vite config with Tailwind + React
// Uses shared dependencies from @repo/vite-config
// TEAM-296: Added path alias for generated Tauri bindings

import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '127.0.0.1', // TEAM-XXX: mac compat - Bind to 127.0.0.1 for Tauri WKWebView
    port: 5173, // Dedicated port for rbee-keeper UI
    strictPort: true, // Fail if port is in use instead of trying another
  },
  optimizeDeps: {
    force: true, // Force dependency pre-bundling on server start
  },
  plugins: [
    tailwindcss(), // Official Tailwind v4 Vite plugin (must be first)
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  build: {
    cssMinify: false, // Disable CSS minification to avoid lightningcss issues with Tailwind
  },
  define: {
    'process.env': {}, // Polyfill for libraries that check process.env
  },
})
