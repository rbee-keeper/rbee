Object.defineProperty(exports, '__esModule', { value: true })
// TEAM-422: Test script to verify CivitAI API fix
const civitai_1 = require('./src/civitai')
async function testCivitAI() {
  console.log('Testing CivitAI API with fixed parameters...\n')
  try {
    const models = await (0, civitai_1.fetchCivitAIModels)({
      limit: 5,
      types: ['Checkpoint', 'LORA'],
      sort: 'Most Downloaded',
      nsfw: false,
    })
    console.log(`✅ SUCCESS: Fetched ${models.length} models`)
    console.log('\nFirst model:')
    console.log(`  - Name: ${models[0]?.name}`)
    console.log(`  - Type: ${models[0]?.type}`)
    console.log(`  - Downloads: ${models[0]?.stats.downloadCount}`)
    console.log(`  - ID: ${models[0]?.id}`)
  } catch (error) {
    console.error('❌ FAILED:', error)
    process.exit(1)
  }
}
testCivitAI()
