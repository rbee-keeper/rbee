#!/usr/bin/env node
/**
 * WCAG AA Contrast Checker for rbee Theme Tokens
 *
 * Validates that color pairs meet minimum contrast ratios:
 * - Normal text (body copy): 4.5:1
 * - Large text (18px+, 14px+ bold): 3.0:1
 * - UI components (borders, focus rings): 3.0:1
 *
 * Usage: pnpm a11y:theme
 */

const { readFileSync } = require('fs')
const { join } = require('path')

// ============================================================================
// WCAG Contrast Calculation
// ============================================================================

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null
}

function relativeLuminance(hex) {
  const rgb = hexToRgb(hex)
  if (!rgb) return 0

  const rsRGB = rgb.r / 255
  const gsRGB = rgb.g / 255
  const bsRGB = rgb.b / 255

  const r = rsRGB <= 0.03928 ? rsRGB / 12.92 : Math.pow((rsRGB + 0.055) / 1.055, 2.4)
  const g = gsRGB <= 0.03928 ? gsRGB / 12.92 : Math.pow((gsRGB + 0.055) / 1.055, 2.4)
  const b = bsRGB <= 0.03928 ? bsRGB / 12.92 : Math.pow((bsRGB + 0.055) / 1.055, 2.4)

  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

function contrastRatio(hex1, hex2) {
  const l1 = relativeLuminance(hex1)
  const l2 = relativeLuminance(hex2)
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

// ============================================================================
// Theme Token Parser
// ============================================================================

function parseThemeTokens(cssContent) {
  const light = {}
  const dark = {}

  // Extract :root block
  const rootMatch = cssContent.match(/:root\s*\{([^}]+)\}/s)
  if (rootMatch) {
    const rootContent = rootMatch[1]
    const tokenRegex = /--([a-z-]+):\s*(#[0-9a-fA-F]{6})/g
    let match
    while ((match = tokenRegex.exec(rootContent)) !== null) {
      light[match[1]] = match[2]
    }
  }

  // Extract .dark block
  const darkMatch = cssContent.match(/\.dark\s*\{([^}]+)\}/s)
  if (darkMatch) {
    const darkContent = darkMatch[1]
    const tokenRegex = /--([a-z-]+):\s*(#[0-9a-fA-F]{6})/g
    let match
    while ((match = tokenRegex.exec(darkContent)) !== null) {
      dark[match[1]] = match[2]
    }
  }

  return { light, dark }
}

// ============================================================================
// Contrast Validation
// ============================================================================

const AA_TEXT_PAIRS = [
  { bg: 'background', fg: 'foreground', minRatio: 4.5, description: 'Body text (normal)' },
  { bg: 'card', fg: 'card-foreground', minRatio: 4.5, description: 'Card text (normal)' },
  { bg: 'popover', fg: 'popover-foreground', minRatio: 4.5, description: 'Popover text (normal)' },
  { bg: 'muted', fg: 'muted-foreground', minRatio: 4.5, description: 'Muted text (normal)' },
  { bg: 'secondary', fg: 'secondary-foreground', minRatio: 4.5, description: 'Secondary text (normal)' },
  { bg: 'accent', fg: 'accent-foreground', minRatio: 4.5, description: 'Accent text (normal)' },
  { bg: 'destructive', fg: 'destructive-foreground', minRatio: 4.5, description: 'Destructive text (normal)' },
]

const LARGE_TEXT_PAIRS = [
  { bg: 'primary', fg: 'primary-foreground', minRatio: 3.0, description: 'Primary CTA (large text only)' },
  { bg: 'success', fg: 'success-foreground', minRatio: 3.0, description: 'Success CTA (large text only)' },
  { bg: 'danger', fg: 'danger-foreground', minRatio: 3.0, description: 'Danger CTA (large text only)' },
]

const UI_PAIRS = [
  { bg: 'background', fg: 'border', minRatio: 3.0, description: 'Border visibility' },
  { bg: 'background', fg: 'input', minRatio: 3.0, description: 'Input border visibility' },
  { bg: 'background', fg: 'ring', minRatio: 3.0, description: 'Focus ring visibility' },
  { bg: 'sidebar', fg: 'sidebar-border', minRatio: 3.0, description: 'Sidebar border visibility' },
  { bg: 'sidebar', fg: 'sidebar-ring', minRatio: 3.0, description: 'Sidebar focus ring visibility' },
]

function checkContrast(tokens, pairs, mode) {
  let passed = 0
  let failed = 0
  const warnings = []

  for (const pair of pairs) {
    const bgColor = tokens[pair.bg]
    const fgColor = tokens[pair.fg]

    if (!bgColor || !fgColor) {
      warnings.push(`[WARN] ${mode}: Missing token --${pair.bg} or --${pair.fg}`)
      continue
    }

    const ratio = contrastRatio(bgColor, fgColor)
    const passes = ratio >= pair.minRatio

    if (passes) {
      passed++
      console.log(`✓ ${mode}: ${pair.description} = ${ratio.toFixed(2)}:1 (needs >= ${pair.minRatio}:1)`)
    } else {
      failed++
      console.error(
        `✗ ${mode}: ${pair.description} = ${ratio.toFixed(2)}:1 (needs >= ${pair.minRatio}:1) [${bgColor} on ${fgColor}]`
      )
    }
  }

  return { passed, failed, warnings }
}

// ============================================================================
// Main
// ============================================================================

function main() {
  console.log('🎨 rbee Theme Contrast Checker (WCAG AA)\n')

  // Read theme tokens
  const themePath = join(__dirname, '../../packages/rbee-ui/src/tokens/theme-tokens.css')
  const cssContent = readFileSync(themePath, 'utf-8')
  const { light, dark } = parseThemeTokens(cssContent)

  console.log(`Loaded ${Object.keys(light).length} light tokens, ${Object.keys(dark).length} dark tokens\n`)

  // Check all pairs
  let totalPassed = 0
  let totalFailed = 0
  const allWarnings = []

  console.log('━━━ NORMAL TEXT (4.5:1) ━━━\n')
  const normalLight = checkContrast(light, AA_TEXT_PAIRS, 'light')
  const normalDark = checkContrast(dark, AA_TEXT_PAIRS, 'dark')
  totalPassed += normalLight.passed + normalDark.passed
  totalFailed += normalLight.failed + normalDark.failed
  allWarnings.push(...normalLight.warnings, ...normalDark.warnings)

  console.log('\n━━━ LARGE TEXT / CTAs (3.0:1) ━━━\n')
  const largeLight = checkContrast(light, LARGE_TEXT_PAIRS, 'light')
  const largeDark = checkContrast(dark, LARGE_TEXT_PAIRS, 'dark')
  totalPassed += largeLight.passed + largeDark.passed
  totalFailed += largeLight.failed + largeDark.failed
  allWarnings.push(...largeLight.warnings, ...largeDark.warnings)

  console.log('\n━━━ UI COMPONENTS (3.0:1) ━━━\n')
  const uiLight = checkContrast(light, UI_PAIRS, 'light')
  const uiDark = checkContrast(dark, UI_PAIRS, 'dark')
  totalPassed += uiLight.passed + uiDark.passed
  totalFailed += uiLight.failed + uiDark.failed
  allWarnings.push(...uiLight.warnings, ...uiDark.warnings)

  // Print warnings
  if (allWarnings.length > 0) {
    console.log('\n━━━ WARNINGS ━━━\n')
    allWarnings.forEach((w) => console.warn(w))
  }

  // Summary
  console.log('\n━━━ SUMMARY ━━━\n')
  console.log(`✓ Passed: ${totalPassed}`)
  console.log(`✗ Failed: ${totalFailed}`)
  console.log(`⚠ Warnings: ${allWarnings.length}`)

  if (totalFailed > 0) {
    console.error('\n❌ WCAG AA contrast check FAILED')
    process.exitCode = 1
  } else {
    console.log('\n✅ WCAG AA contrast check PASSED')
    process.exitCode = 0
  }
}

main()
