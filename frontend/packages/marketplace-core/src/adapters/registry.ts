// TEAM-476: Adapter Registry - ONE place to register all vendor adapters
// TEAM-482: Added GWC (Global Worker Catalog) adapter

import type { MarketplaceAdapter } from './adapter'
import { civitaiAdapter } from './civitai'
import { gwcAdapter } from './gwc'
import { huggingfaceAdapter } from './huggingface'

/**
 * Registry of all marketplace adapters
 * Add new vendors here - that's it!
 */
export const adapters = {
  civitai: civitaiAdapter,
  huggingface: huggingfaceAdapter,
  gwc: gwcAdapter, // TEAM-482: Global Worker Catalog
  // Future: ollama, replicate, etc.
} as const

/**
 * Vendor names (type-safe)
 */
export type VendorName = keyof typeof adapters

/**
 * Get adapter by vendor name
 */
export function getAdapter(vendor: VendorName): MarketplaceAdapter {
  return adapters[vendor]
}
