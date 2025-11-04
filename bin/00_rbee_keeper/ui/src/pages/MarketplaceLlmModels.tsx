// TEAM-405: Marketplace LLM Models page
// Browse and search HuggingFace models using marketplace-sdk

import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { Input } from "@rbee/ui/atoms";
import { MarketplaceGrid } from "@rbee/ui/marketplace/organisms/MarketplaceGrid";
import { ModelCard } from "@rbee/ui/marketplace/organisms/ModelCard";
import type { Model } from "@/generated/bindings";

export function MarketplaceLlmModels() {
  const [searchQuery, setSearchQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");

  // Debounce search query
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    // Debounce by 500ms
    setTimeout(() => {
      setDebouncedQuery(value);
    }, 500);
  };

  // Fetch models from HuggingFace via Tauri command
  const { data: models = [], isLoading, error } = useQuery({
    queryKey: ["marketplace", "llm-models", debouncedQuery],
    queryFn: async () => {
      const result = await invoke<Model[]>("marketplace_list_models", {
        query: debouncedQuery || null,
        limit: 50,
      });
      return result;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-border p-6">
        <h1 className="text-2xl font-bold mb-2">LLM Models</h1>
        <p className="text-muted-foreground">
          Browse and search language models from HuggingFace
        </p>
      </div>

      {/* Search */}
      <div className="border-b border-border p-6">
        <Input
          type="text"
          placeholder="Search models (e.g., 'llama', 'mistral', 'phi')..."
          value={searchQuery}
          onChange={(e) => handleSearchChange(e.target.value)}
          className="max-w-2xl"
        />
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-6">
        <MarketplaceGrid
          items={models}
          renderItem={(model) => <ModelCard key={model.id} model={model} />}
          isLoading={isLoading}
          error={error ? String(error) : undefined}
          emptyMessage="No models found"
          emptyDescription="Try adjusting your search query"
          columns={3}
        />
      </div>
    </div>
  );
}
