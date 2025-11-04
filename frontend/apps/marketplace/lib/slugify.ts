/**
 * Slugify utilities for model IDs
 * Converts HuggingFace model IDs to URL-friendly slugs
 */

/**
 * Convert a model ID to a URL-friendly slug
 * 
 * @example
 * modelIdToSlug('sentence-transformers/all-MiniLM-L6-v2')
 * // => 'sentence-transformers--all-minilm-l6-v2'
 * 
 * modelIdToSlug('meta-llama/Llama-2-7b-chat-hf')
 * // => 'meta-llama--llama-2-7b-chat-hf'
 */
export function modelIdToSlug(modelId: string): string {
  return modelId
    .toLowerCase()
    .replace(/[^a-z0-9/-]/g, '-')  // Replace non-alphanumeric (except slash) with dash
    .replace(/-+/g, '-')  // Collapse multiple dashes (but not slashes yet)
    .replace(/\//g, '--')  // Replace slashes with double dash (AFTER collapsing)
    .replace(/^-|-$/g, '')  // Remove leading/trailing dashes
}

/**
 * Convert a slug back to a model ID
 * 
 * @example
 * slugToModelId('sentence-transformers--all-minilm-l6-v2')
 * // => 'sentence-transformers/all-MiniLM-L6-v2'
 * 
 * Note: Case information is lost in slugification, so we need to
 * look up the original model ID from the data source
 */
export function slugToModelId(slug: string): string {
  // Convert double dash back to slash
  return slug.replace(/--/g, '/')
}

/**
 * Check if a string is a valid model slug
 */
export function isValidSlug(slug: string): boolean {
  return /^[a-z0-9]+(-[a-z0-9]+)*--[a-z0-9]+(-[a-z0-9]+)*$/.test(slug)
}

/**
 * Extract organization and model name from slug
 * 
 * @example
 * parseSlug('sentence-transformers--all-minilm-l6-v2')
 * // => { org: 'sentence-transformers', model: 'all-minilm-l6-v2' }
 */
export function parseSlug(slug: string): { org: string; model: string } | null {
  const parts = slug.split('--')
  if (parts.length !== 2) return null
  
  return {
    org: parts[0],
    model: parts[1]
  }
}
