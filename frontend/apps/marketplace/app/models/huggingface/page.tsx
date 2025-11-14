// TEAM-476: HuggingFace models page - CARD presentation
// TEAM-477: Added MVP compatibility banner
// TEAM-478: Redesigned to card layout (2 columns)
// TEAM-478: Added clickable cards linking to detail pages
// TEAM-481: Refactored to use reusable HFModelListCard component
// TEAM-505: Redesigned with sidebar filters (inspired by HuggingFace official site)
// TEAM-502: Updated to use HFFilterSidebar (worker-driven filters)

import { fetchGWCWorkers } from '@rbee/marketplace-core'
import { HuggingFaceSidebarFilters } from '../../../components/HuggingFaceSidebarFilters'

export const dynamic = 'force-static'

export default async function HuggingFaceModelsPage() {
  const workers = await fetchGWCWorkers({ limit: 50 })

  return <HuggingFaceSidebarFilters workers={workers} />
}
