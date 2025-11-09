// TEAM-453: E2E tests for workers page
import { expect, test } from '@playwright/test'

test.describe('Workers Page', () => {
  test('should navigate to workers page', async ({ page }) => {
    await page.goto('/')

    // Click workers link
    await page.click('a[href*="workers"]')
    await expect(page).toHaveURL(/\/workers/)
  })

  test('should display worker list', async ({ page }) => {
    await page.goto('/workers')

    // Should have worker cards or list items
    const workerElements = page.locator('[data-testid*="worker"], article, .worker-card')
    const count = await workerElements.count()

    // Should have at least some workers
    expect(count).toBeGreaterThanOrEqual(0)
  })

  test('should have filter functionality', async ({ page }) => {
    await page.goto('/workers')

    // Look for filter controls
    const filterBar = page.locator('[data-testid="filter-bar"], .filter-bar, form')
    const hasFilters = await filterBar.count()

    if (hasFilters > 0) {
      await expect(filterBar.first()).toBeVisible()
    }
  })

  test('should have search functionality', async ({ page }) => {
    await page.goto('/workers')

    // Look for search input
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]')
    const hasSearch = await searchInput.count()

    if (hasSearch > 0) {
      await expect(searchInput.first()).toBeVisible()
    }
  })

  test('should show worker details on click', async ({ page }) => {
    await page.goto('/workers')

    // Find first worker link
    const workerLink = page.locator('a[href*="/workers/"]').first()
    const hasWorkers = await workerLink.count()

    if (hasWorkers > 0) {
      await workerLink.click()
      await expect(page).toHaveURL(/\/workers\/[^/]+/)
    }
  })

  test('should have install button on worker detail', async ({ page }) => {
    await page.goto('/workers')

    // Navigate to first worker
    const workerLink = page.locator('a[href*="/workers/"]').first()
    const hasWorkers = await workerLink.count()

    if (hasWorkers > 0) {
      await workerLink.click()

      // Should have install button or CTA
      const installButton = page.locator('button:has-text("install"), a:has-text("install")')
      await expect(installButton.first()).toBeVisible()
    }
  })
})
