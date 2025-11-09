// This file has been automatically migrated to valid ESM format by Storybook.
import { createRequire } from 'node:module'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import type { StorybookConfig } from '@storybook/react-vite'
import tailwindcss from '@tailwindcss/vite'

const require = createRequire(import.meta.url)

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|mjs|ts|tsx)'],
  addons: [getAbsolutePath('@storybook/addon-themes')],
  framework: {
    name: getAbsolutePath('@storybook/react-vite'),
    options: {
      builder: {
        viteConfigPath: undefined,
      },
    },
  },
  core: {
    disableTelemetry: true,
    disableWhatsNewNotifications: true,
  },
  viteFinal: async (config) => {
    config.plugins = config.plugins || []
    config.plugins.push(tailwindcss())

    // Define process.env for browser
    config.define = {
      ...config.define,
      'process.env': {},
      'process.env.NODE_ENV': JSON.stringify('development'),
    }

    // Resolve next-themes properly
    config.resolve = config.resolve || {}
    config.resolve.alias = {
      ...config.resolve.alias,
      'next/link': require.resolve('./mocks/next-link.tsx'),
      'next/navigation': require.resolve('./mocks/next-navigation.tsx'),
      'next/image': require.resolve('./mocks/next-image.tsx'),
    }

    return config
  },
}
export default config

function getAbsolutePath(value: string): any {
  return dirname(fileURLToPath(import.meta.resolve(`${value}/package.json`)))
}
