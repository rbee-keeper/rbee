#!/usr/bin/env tsx
// TEAM-471: Debug script to test a single CivitAI filter
// TEAM-476: Fixed TypeScript errors with Partial<CivitaiFilters>
import { parseCivitAIFilter } from '../config/filter-parser'
import { getCompatibleCivitaiModels } from '@rbee/marketplace-node'

const filterPath = process.argv[2] || 'filter/x/week/sd21'

console.log(`🔍 Testing filter: ${filterPath}`)
console.log(`📋 Parsing filter...`)

const parsedFilters = parseCivitAIFilter(filterPath)
console.log(`✅ Parsed filters:`, JSON.stringify(parsedFilters, null, 2))

// Construct full filter object with defaults for API call
const apiFilters = {
  limit: 100,
  ...parsedFilters,
}

console.log(`\n🌐 Calling CivitAI API...`)
console.log(`   URL params that will be sent:`)
console.log(`   - limit: ${apiFilters.limit}`)
console.log(`   - sort: ${apiFilters.sort ?? 'Highest Rated'}`)
console.log(`   - modelType: ${apiFilters.modelType ?? 'All'}`)
console.log(`   - timePeriod: ${apiFilters.timePeriod ?? 'AllTime'}`)
console.log(`   - baseModel: ${apiFilters.baseModel ?? 'All'}`)
console.log(`   - nsfwLevel: ${apiFilters.nsfwLevel ?? 'XXX'}`)

getCompatibleCivitaiModels(apiFilters)
  .then((models) => {
    console.log(`\n✅ SUCCESS: Received ${models.length} models`)
    if (models.length > 0) {
      console.log(`\n📦 First model:`)
      console.log(`   - ID: ${models[0].id}`)
      console.log(`   - Name: ${models[0].name}`)
      console.log(`   - Tags: ${models[0].tags.slice(0, 3).join(', ')}`)
    }
  })
  .catch((error) => {
    console.error(`\n❌ ERROR:`, error)
    console.error(`\n💡 This might be:`)
    console.error(`   - CivitAI API is down or overloaded`)
    console.error(`   - Invalid filter combination`)
    console.error(`   - Rate limiting (try again in a few seconds)`)
    process.exit(1)
  })
