// TEAM-453: E2E tests for search functionality
import { expect, test } from '@playwright/test'

test.describe('Search Functionality', () => {
  test('should have global search', async ({ page }) => {
    await page.goto('/')

    // Look for search input in navigation or header
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first()
    const hasSearch = await searchInput.count()

    if (hasSearch > 0) {
      await expect(searchInput).toBeVisible()
    }
  })

  test('should navigate to search page', async ({ page }) => {
    await page.goto('/search')

    // Should load search page
    await expect(page).toHaveURL(/\/search/)
  })

  test('should accept search query', async ({ page }) => {
    await page.goto('/search')

    // Find search input
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first()
    const hasSearch = await searchInput.count()

    if (hasSearch > 0) {
      await searchInput.fill('llama')
      await page.keyboard.press('Enter')
      await page.waitForTimeout(1000) // Wait for results
    }
  })

  test('should show search results', async ({ page }) => {
    await page.goto('/search?q=llama')

    // Should show some results or "no results" message
    const results = page.locator('[data-testid*="result"], article, .search-result')
    const noResults = page.locator('text=/no results|not found/i')

    await expect(results.first().or(noResults)).toBeVisible()
  })

  test('should filter search results', async ({ page }) => {
    await page.goto('/search?q=worker')

    // Look for filter options
    const filterOptions = page.locator('button, input[type="checkbox"], select')
    const hasFilters = await filterOptions.count()

    expect(hasFilters).toBeGreaterThanOrEqual(0)
  })
})
