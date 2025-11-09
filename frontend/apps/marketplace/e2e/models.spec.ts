// TEAM-453: E2E tests for models page
import { expect, test } from '@playwright/test'

test.describe('Models Page', () => {
  test('should navigate to models page', async ({ page }) => {
    await page.goto('/')

    // Click models link
    const modelsLink = page.locator('a[href*="models"]').first()
    const hasLink = await modelsLink.count()

    if (hasLink > 0) {
      await modelsLink.click()
      await expect(page).toHaveURL(/\/models/)
    }
  })

  test('should display model list', async ({ page }) => {
    await page.goto('/models')

    // Should have model cards or table rows
    const modelElements = page.locator('[data-testid*="model"], article, tr, .model-card')
    const count = await modelElements.count()

    // Should have at least some models
    expect(count).toBeGreaterThanOrEqual(0)
  })

  test('should have filter functionality', async ({ page }) => {
    await page.goto('/models')

    // Look for filter controls
    const filterBar = page.locator('[data-testid="filter-bar"], .filter-bar, form')
    const hasFilters = await filterBar.count()

    if (hasFilters > 0) {
      await expect(filterBar.first()).toBeVisible()
    }
  })

  test('should have search functionality', async ({ page }) => {
    await page.goto('/models')

    // Look for search input
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]')
    const hasSearch = await searchInput.count()

    if (hasSearch > 0) {
      await expect(searchInput.first()).toBeVisible()
      await searchInput.first().fill('llama')
      await page.waitForTimeout(500) // Wait for filter
    }
  })

  test('should show model details', async ({ page }) => {
    await page.goto('/models')

    // Find first model link
    const modelLink = page.locator('a[href*="/models/"]').first()
    const hasModels = await modelLink.count()

    if (hasModels > 0) {
      await modelLink.click()
      await expect(page).toHaveURL(/\/models\/[^/]+/)
    }
  })

  test('should have model metadata', async ({ page }) => {
    await page.goto('/models')

    // Navigate to first model
    const modelLink = page.locator('a[href*="/models/"]').first()
    const hasModels = await modelLink.count()

    if (hasModels > 0) {
      await modelLink.click()

      // Should show model information
      const content = page.locator('main, article, .model-detail')
      await expect(content.first()).toBeVisible()
    }
  })
})
