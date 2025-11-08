import tailwindcss from '@tailwindcss/postcss'
import postcssNesting from 'postcss-nesting'
import postcssRemoveInvalidLength from './postcss-remove-invalid-length.mjs'

const config = {
  plugins: [
    tailwindcss,
    postcssNesting,
    postcssRemoveInvalidLength, // TEAM-450: Remove auto-generated invalid CSS rules
  ],
}

export default config
