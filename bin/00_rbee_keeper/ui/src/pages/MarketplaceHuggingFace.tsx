// TEAM-405: Marketplace LLM Models page - Using reusable components
// TEAM-413: Fixed to use ModelTable instead of non-existent ModelListTableTemplate
// TEAM-423: Renamed to MarketplaceHuggingFace for clarity
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: ModelTable from rbee-ui

import { useNavigate } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { PageContainer } from "@rbee/ui/molecules";
import { ModelTable } from "@rbee/ui/marketplace";
import type { Model } from "@/generated/bindings";

export function MarketplaceHuggingFace() {
  const navigate = useNavigate();

  // DATA LAYER: Fetch models from Tauri
  const { data: rawModels = [], isLoading, error } = useQuery({
    queryKey: ["marketplace", "llm-models"],
    queryFn: async () => {
      const result = await invoke<Model[]>("marketplace_list_models", {
        query: null,
        sort: "downloads",
        filterTags: null,
        limit: 50,
      });
      return result;
    },
    staleTime: 5 * 60 * 1000,
  });

  // PRESENTATION LAYER: Render with reusable component
  return (
    <PageContainer
      title="LLM Models"
      description="Discover and download state-of-the-art language models from HuggingFace"
      padding="default"
    >
      {isLoading && <div>Loading models...</div>}
      {error && <div>Error: {String(error)}</div>}
      {!isLoading && !error && (
        <ModelTable
          models={rawModels}
          onModelClick={(modelId: string) => navigate(`/marketplace/llm-models/${encodeURIComponent(modelId)}`)}
        />
      )}
    </PageContainer>
  );
}
