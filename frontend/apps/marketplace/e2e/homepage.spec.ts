// TEAM-453: E2E tests for marketplace homepage
import { expect, test } from '@playwright/test'

test.describe('Marketplace Homepage', () => {
  test('should load successfully', async ({ page }) => {
    await page.goto('/')
    await expect(page).toHaveTitle(/marketplace|rbee/i)
  })

  test('should have navigation', async ({ page }) => {
    await page.goto('/')

    // Check for navigation elements
    const nav = page.locator('nav')
    await expect(nav).toBeVisible()
  })

  test('should have hero section', async ({ page }) => {
    await page.goto('/')

    // Check for main heading
    const heading = page.locator('h1').first()
    await expect(heading).toBeVisible()
  })

  test('should have worker and model sections', async ({ page }) => {
    await page.goto('/')

    // Should have links to workers and models
    const workersLink = page.locator('a[href*="workers"]').first()
    const modelsLink = page.locator('a[href*="models"]').first()

    await expect(workersLink.or(modelsLink)).toBeVisible()
  })

  test('should be responsive', async ({ page }) => {
    await page.goto('/')

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page.locator('body')).toBeVisible()

    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 })
    await expect(page.locator('body')).toBeVisible()
  })

  test('should have no console errors', async ({ page }) => {
    const errors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text())
      }
    })

    await page.goto('/')
    await page.waitForLoadState('networkidle')

    expect(errors).toHaveLength(0)
  })
})
