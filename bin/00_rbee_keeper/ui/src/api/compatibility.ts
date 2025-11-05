// TEAM-411: Compatibility API wrapper for Keeper
// Wraps Tauri commands for compatibility checking

import { invoke } from '@tauri-apps/api/core'

/**
 * Compatibility confidence level
 */
export type CompatibilityConfidence = 'high' | 'medium' | 'low' | 'none'

/**
 * Compatibility check result
 */
export interface CompatibilityResult {
  compatible: boolean
  confidence: CompatibilityConfidence
  reasons: string[]
  warnings: string[]
  recommendations: string[]
}

/**
 * Check if a model is compatible with a worker
 * 
 * @param modelId - HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
 * @param workerType - Worker type ("cpu", "cuda", or "metal")
 * @returns Compatibility result with reasons and recommendations
 * 
 * @example
 * ```typescript
 * const result = await checkModelCompatibility('meta-llama/Llama-3.2-1B', 'cpu')
 * if (result.compatible) {
 *   console.log('✅ Model is compatible!')
 * }
 * ```
 */
export async function checkModelCompatibility(
  modelId: string,
  workerType: string
): Promise<CompatibilityResult> {
  return invoke('check_model_compatibility', { modelId, workerType })
}

/**
 * List all workers compatible with a model
 * 
 * @param modelId - HuggingFace model ID
 * @returns Array of compatible worker type strings
 * 
 * @example
 * ```typescript
 * const workers = await listCompatibleWorkers('meta-llama/Llama-3.2-1B')
 * console.log(`Compatible with: ${workers.join(', ')}`)
 * ```
 */
export async function listCompatibleWorkers(
  modelId: string
): Promise<string[]> {
  return invoke('list_compatible_workers', { modelId })
}

/**
 * List all models compatible with a worker
 * 
 * @param workerType - Worker type ("cpu", "cuda", or "metal")
 * @param limit - Maximum number of models to return (default: 50)
 * @returns Array of compatible model IDs
 * 
 * @example
 * ```typescript
 * const models = await listCompatibleModels('cuda', 100)
 * console.log(`Found ${models.length} compatible models`)
 * ```
 */
export async function listCompatibleModels(
  workerType: string,
  limit?: number
): Promise<string[]> {
  return invoke('list_compatible_models', { workerType, limit })
}
