# Puppeteer Testing Guide for Filter Components

## How to Test Filters with Puppeteer

### Basic Filter Click Test

```javascript
// 1. Navigate to the page
await page.goto('http://localhost:7823/models/huggingface');
await page.waitForSelector('button');

// 2. Find the filter button by text content
const filterButtons = await page.$$('button');
let sizeButton = null;

for (const button of filterButtons) {
  const text = await button.evaluate(el => el.textContent);
  if (text?.includes('Model Size')) {
    sizeButton = button;
    break;
  }
}

// 3. Click the filter button
await sizeButton.click();
await page.waitForTimeout(300); // Wait for dropdown

// 4. Find and click filter option
const menuItems = await page.$$('[role="menuitem"]');
let smallOption = null;

for (const item of menuItems) {
  const text = await item.evaluate(el => el.textContent);
  if (text?.includes('Small')) {
    smallOption = item;
    break;
  }
}

await smallOption.click();
await page.waitForTimeout(1500); // Wait for navigation

// 5. Verify URL changed
const url = page.url();
console.log('URL after filter:', url);
// Expected: http://localhost:7823/models/huggingface?size=small

// 6. Verify data changed
const modelData = await page.evaluate(() => {
  const rows = Array.from(document.querySelectorAll('tbody tr')).slice(0, 3);
  return rows.map(row => {
    const cells = row.querySelectorAll('td');
    return {
      name: cells[0]?.textContent?.trim(),
      downloads: cells[2]?.textContent?.trim()
    };
  });
});
console.log('First 3 models:', modelData);
```

### Why Clicking Is Hard

The filter dropdowns use **Radix UI**, which generates random IDs like `radix-_R_b9bn5ritqlb_`. These IDs change on every page load, so you can't use fixed selectors.

**Solution**: Use `page.evaluate()` to find elements by text content inside the browser context.

### Complete Test Example

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: false });
  const page = await browser.newPage();
  
  // Navigate
  await page.goto('http://localhost:7823/models/huggingface');
  await page.waitForSelector('tbody tr');
  
  // Get initial data
  const initialData = await page.evaluate(() => {
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    return {
      count: rows.length,
      first: rows[0]?.querySelector('td')?.textContent?.trim()
    };
  });
  console.log('Initial:', initialData);
  
  // Click Model Size filter
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const sizeBtn = buttons.find(btn => btn.textContent?.includes('Model Size'));
    if (sizeBtn) sizeBtn.click();
  });
  
  await page.waitForTimeout(300);
  
  // Click Small option
  await page.evaluate(() => {
    const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
    const smallOption = items.find(item => item.textContent?.includes('Small'));
    if (smallOption) smallOption.click();
  });
  
  await page.waitForTimeout(1500);
  
  // Get filtered data
  const filteredData = await page.evaluate(() => {
    const rows = Array.from(document.querySelectorAll('tbody tr'));
    return {
      url: window.location.href,
      count: rows.length,
      first: rows[0]?.querySelector('td')?.textContent?.trim()
    };
  });
  console.log('Filtered:', filteredData);
  
  // Verify
  if (filteredData.url.includes('size=small')) {
    console.log('✅ URL changed correctly');
  } else {
    console.log('❌ URL did not change');
  }
  
  await browser.close();
})();
```

### Testing Multiple Filters

```javascript
// After clicking first filter (size=small), click second filter
await page.evaluate(() => {
  const buttons = Array.from(document.querySelectorAll('button'));
  const sortBtn = buttons.find(btn => btn.textContent?.includes('Sort By'));
  if (sortBtn) sortBtn.click();
});

await page.waitForTimeout(300);

await page.evaluate(() => {
  const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
  const likesOption = items.find(item => item.textContent?.includes('Most Likes'));
  if (likesOption) likesOption.click();
});

await page.waitForTimeout(1500);

const multiFilterURL = page.url();
console.log('URL with 2 filters:', multiFilterURL);
// Expected: http://localhost:7823/models/huggingface?size=small&sort=likes
```

### Common Issues

#### 1. Dropdown Not Appearing

**Symptom**: Clicking button doesn't show menu items

**Solution**: Wait longer after click
```javascript
await page.waitForTimeout(500); // Increase wait time
```

#### 2. Random Radix IDs

**Symptom**: Selectors like `#radix-_R_b9bn5ritqlb_` don't work next time

**Solution**: Use text-based selection in `page.evaluate()`

#### 3. Click Not Working

**Symptom**: `page.click()` fails or does nothing

**Solution**: Use `page.evaluate()` to click inside browser context
```javascript
await page.evaluate((selector) => {
  document.querySelector(selector)?.click();
}, '#my-button');
```

### Verifying Data Changes

```javascript
// Check if model data actually changed
const verifyChange = await page.evaluate(() => {
  const rows = Array.from(document.querySelectorAll('tbody tr')).slice(0, 5);
  const models = rows.map(row => {
    const cells = row.querySelectorAll('td');
    return {
      name: cells[0]?.textContent?.trim(),
      author: cells[1]?.textContent?.trim(),
      downloads: cells[2]?.textContent?.trim(),
      likes: cells[3]?.textContent?.trim()
    };
  });
  
  return {
    models,
    hasDownloads: models.some(m => m.downloads && m.downloads !== '0'),
    hasAuthor: models.some(m => m.author && m.author !== '—')
  };
});

console.log('Data verification:', verifyChange);

if (verifyChange.hasDownloads && verifyChange.hasAuthor) {
  console.log('✅ Models have full metadata');
} else {
  console.log('❌ Models missing metadata (showing 0s or dashes)');
}
```

### Full Integration Test

```javascript
async function testFilterFlow() {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  
  try {
    // 1. Navigate
    await page.goto('http://localhost:7823/models/huggingface');
    await page.waitForSelector('tbody tr');
    
    // 2. Capture initial state
    const initial = await page.evaluate(() => ({
      url: window.location.href,
      firstModel: document.querySelector('tbody tr td')?.textContent?.trim()
    }));
    
    // 3. Apply first filter
    await page.evaluate(() => {
      const btn = Array.from(document.querySelectorAll('button'))
        .find(b => b.textContent?.includes('Model Size'));
      btn?.click();
    });
    await page.waitForTimeout(300);
    
    await page.evaluate(() => {
      const opt = Array.from(document.querySelectorAll('[role="menuitem"]'))
        .find(i => i.textContent?.includes('Small'));
      opt?.click();
    });
    await page.waitForTimeout(1500);
    
    // 4. Verify first filter
    const afterFirst = await page.evaluate(() => ({
      url: window.location.href,
      firstModel: document.querySelector('tbody tr td')?.textContent?.trim()
    }));
    
    console.assert(afterFirst.url.includes('size=small'), 'First filter failed');
    
    // 5. Apply second filter
    await page.evaluate(() => {
      const btn = Array.from(document.querySelectorAll('button'))
        .find(b => b.textContent?.includes('Sort By'));
      btn?.click();
    });
    await page.waitForTimeout(300);
    
    await page.evaluate(() => {
      const opt = Array.from(document.querySelectorAll('[role="menuitem"]'))
        .find(i => i.textContent?.includes('Most Likes'));
      opt?.click();
    });
    await page.waitForTimeout(1500);
    
    // 6. Verify both filters
    const afterSecond = await page.evaluate(() => ({
      url: window.location.href,
      firstModel: document.querySelector('tbody tr td')?.textContent?.trim()
    }));
    
    console.assert(
      afterSecond.url.includes('size=small') && afterSecond.url.includes('sort=likes'),
      'Second filter failed'
    );
    
    console.assert(
      afterSecond.firstModel !== afterFirst.firstModel,
      'Data did not change after second filter'
    );
    
    console.log('✅ All tests passed');
    
  } catch (error) {
    console.error('❌ Test failed:', error);
  } finally {
    await browser.close();
  }
}

testFilterFlow();
```

## Key Takeaways

1. **Use `page.evaluate()` for dynamic content** - Radix IDs change on every load
2. **Wait after clicks** - Dropdowns and navigation take time
3. **Verify URL and data** - Check both URL params and actual model data changed
4. **Test multiple filters** - Ensure they combine correctly
5. **Check metadata** - Verify downloads/likes/author are not showing as 0/"—"
