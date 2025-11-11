#!/usr/bin/env tsx
// TEAM-467: Test marketplace-node SDK before using in manifest generation
import { listHuggingFaceModels } from '@rbee/marketplace-node'

async function test() {
  console.log('Testing marketplace-node SDK...')

  try {
    const models = await listHuggingFaceModels({
      limit: 5,
      sort: 'popular',
    })

    console.log(`✅ Fetched ${models.length} models`)
    console.log('\nFirst model:')
    console.log(JSON.stringify(models[0], null, 2))

    // Verify expected fields
    const model = models[0]
    const hasRequiredFields =
      model.id && model.name && model.tags && typeof model.downloads === 'number' && typeof model.likes === 'number'

    if (hasRequiredFields) {
      console.log('\n✅ Model has all required fields')
    } else {
      console.log('\n❌ Model missing required fields')
    }
  } catch (error) {
    console.error('❌ SDK test failed:', error)
    process.exit(1)
  }
}

test()
