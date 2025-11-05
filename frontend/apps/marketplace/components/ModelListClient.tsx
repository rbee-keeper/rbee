// TEAM-415: Client-side model list with routing
'use client'

import { ModelTableWithRouting } from './ModelTableWithRouting'
import type { ModelTableItem } from '@rbee/ui/marketplace'

interface Props {
  models: ModelTableItem[]
}

export function ModelListClient({ models }: Props) {
  return <ModelTableWithRouting models={models} />
}
