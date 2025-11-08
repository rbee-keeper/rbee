/**
 * TEAM-450: PostCSS plugin to remove auto-generated invalid CSS rules
 *
 * Tailwind v4 auto-generates utilities for [length:...] and [length:var(...)]
 * even though we don't use them. These generate invalid CSS properties that
 * Turbopack's strict parser rejects.
 *
 * This plugin removes ONLY these specific invalid rules after Tailwind compilation.
 */
export default function postcssRemoveInvalidLength() {
  return {
    postcssPlugin: 'postcss-remove-invalid-length',
    Rule(rule) {
      // Remove rules with invalid 'length' property
      const hasInvalidLength = rule.nodes?.some(node => {
        if (node.type === 'decl') {
          // Invalid CSS: property named 'length' (not a real CSS property)
          return node.prop === 'length';
        }
        return false;
      });

      if (hasInvalidLength) {
        rule.remove();
      }
    }
  };
}

postcssRemoveInvalidLength.postcss = true;
