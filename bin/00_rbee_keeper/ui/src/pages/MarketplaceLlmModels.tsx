// TEAM-405: Marketplace LLM Models page - Using reusable components
// DATA LAYER: Tauri commands + React Query
// PRESENTATION/CONTROL: ModelListTableTemplate from rbee-ui

import { useNavigate } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { PageContainer } from "@rbee/ui/molecules";
import { ModelListTableTemplate, useModelFilters } from "@rbee/ui/marketplace";
import type { Model } from "@/generated/bindings";

export function MarketplaceLlmModels() {
  const navigate = useNavigate();
  
  // CONTROL LAYER: Filter state management
  const { filters, setSearch, setSort, toggleTag, clearFilters, sortOptions, filterChips } = useModelFilters({
    defaultSort: "downloads",
    availableChips: [
      { id: "transformers", label: "Transformers" },
      { id: "safetensors", label: "SafeTensors" },
      { id: "gguf", label: "GGUF" },
      { id: "pytorch", label: "PyTorch" },
    ],
  });

  // DATA LAYER: Fetch models from Tauri
  const { data: rawModels = [], isLoading, error } = useQuery({
    queryKey: ["marketplace", "llm-models", filters],
    queryFn: async () => {
      const result = await invoke<Model[]>("marketplace_list_models", {
        query: filters.search || null,
        sort: filters.sort,
        filterTags: filters.tags.length > 0 ? filters.tags : null,
        limit: 50,
      });
      return result;
    },
    staleTime: 5 * 60 * 1000,
  });

  // PRESENTATION LAYER: Render with reusable template
  return (
    <PageContainer
      title="LLM Models"
      description="Discover and download state-of-the-art language models from HuggingFace"
      padding="default"
    >
      <ModelListTableTemplate
        models={rawModels}
        onModelClick={(modelId) => navigate(`/marketplace/llm-models/${encodeURIComponent(modelId)}`)}
        isLoading={isLoading}
        error={error ? String(error) : undefined}
        emptyMessage="No models found"
        emptyDescription="Try adjusting your search query"
        filters={filters}
        onFiltersChange={(newFilters) => {
          if (newFilters.search !== filters.search) setSearch(newFilters.search);
          if (newFilters.sort !== filters.sort) setSort(newFilters.sort);
          if (JSON.stringify(newFilters.tags) !== JSON.stringify(filters.tags)) {
            // Find which tag changed and toggle it
            const added = newFilters.tags.find(t => !filters.tags.includes(t));
            const removed = filters.tags.find(t => !newFilters.tags.includes(t));
            if (added) toggleTag(added);
            if (removed) toggleTag(removed);
          }
        }}
        filterOptions={{
          defaultSort: "downloads",
          sortOptions,
          availableChips: [
            { id: "transformers", label: "Transformers" },
            { id: "safetensors", label: "SafeTensors" },
            { id: "gguf", label: "GGUF" },
            { id: "pytorch", label: "PyTorch" },
          ],
        }}
      />
    </PageContainer>
  );
}
