// TEAM-415: Client-side model list with routing
'use client'

import type { ModelTableItem } from '@rbee/ui/marketplace'
import { ModelTableWithRouting } from './ModelTableWithRouting'

interface Props {
  models: ModelTableItem[]
}

export function ModelListClient({ models }: Props) {
  return <ModelTableWithRouting models={models} />
}
