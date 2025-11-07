"use strict";
// TEAM-460: CivitAI API integration
// API Documentation: https://github.com/civitai/civitai/wiki/REST-API-Reference
Object.defineProperty(exports, "__esModule", { value: true });
exports.fetchCivitAIModels = fetchCivitAIModels;
exports.fetchCivitAIModel = fetchCivitAIModel;
/**
 * Fetch models from CivitAI API
 *
 * @param options - Search options
 * @returns Array of CivitAI models
 */
async function fetchCivitAIModels(options = {}) {
    const { query, limit = 20, types = ['Checkpoint', 'LORA'], sort = 'Most Downloaded', nsfw = false, } = options;
    const params = new URLSearchParams({
        limit: String(limit),
        sort,
        nsfw: String(nsfw),
    });
    if (query) {
        params.append('query', query);
    }
    // TEAM-422: CivitAI API requires multiple 'types' parameters, not comma-separated
    // Correct: ?types=Checkpoint&types=LORA
    // Wrong: ?types=Checkpoint,LORA
    if (types.length > 0) {
        types.forEach(type => {
            params.append('types', type);
        });
    }
    const url = `https://civitai.com/api/v1/models?${params}`;
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        return data.items;
    }
    catch (error) {
        console.error('[marketplace-node] CivitAI API error:', error);
        throw error;
    }
}
/**
 * Fetch a specific model from CivitAI by ID
 *
 * @param modelId - CivitAI model ID
 * @returns CivitAI model details
 */
async function fetchCivitAIModel(modelId) {
    const url = `https://civitai.com/api/v1/models/${modelId}`;
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`CivitAI API error: ${response.status} ${response.statusText}`);
        }
        return await response.json();
    }
    catch (error) {
        console.error('[marketplace-node] CivitAI API error:', error);
        throw error;
    }
}
