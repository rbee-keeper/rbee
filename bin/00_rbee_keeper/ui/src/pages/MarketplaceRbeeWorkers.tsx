// TEAM-405: Marketplace Rbee Workers page - Using reusable components
// TEAM-421: Implemented with WorkerListTemplate and marketplace_list_workers command
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: WorkerListTemplate from rbee-ui

import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { PageContainer } from "@rbee/ui/molecules";
import { WorkerListTemplate } from "@rbee/ui/marketplace";
import type { WorkerCatalogEntry } from "@/generated/bindings";

export function MarketplaceRbeeWorkers() {
  const navigate = useNavigate();
  // DATA LAYER: Fetch workers from Tauri
  const { data: rawWorkers = [], isLoading, error } = useQuery({
    queryKey: ["marketplace", "rbee-workers"],
    queryFn: async () => {
      const result = await invoke<WorkerCatalogEntry[]>("marketplace_list_workers");
      return result;
    },
    staleTime: 5 * 60 * 1000,
  });

  // TEAM-421: Transform WorkerCatalogEntry to WorkerCard format
  const workers = rawWorkers.map((worker) => ({
    id: worker.id,
    name: worker.name,
    description: worker.description,
    version: worker.version,
    platform: worker.platforms.map((p: string) => p.toLowerCase()),
    architecture: worker.architectures.map((a: string) => a.toLowerCase()),
    workerType: worker.workerType.toLowerCase() as 'cpu' | 'cuda' | 'metal',
  }));

  // PRESENTATION LAYER: Render with reusable component
  return (
    <PageContainer
      title="Rbee Workers"
      description="Browse and install rbee workers for running AI models on different hardware"
      padding="default"
    >
      {isLoading && <div>Loading workers...</div>}
      {error && <div>Error: {String(error)}</div>}
      {!isLoading && !error && (
        <WorkerListTemplate
          title="Inference Workers"
          description="Download workers optimized for your hardware"
          workers={workers}
          isLoading={isLoading}
          error={error ? String(error) : undefined}
          onWorkerClick={(workerId) => navigate(`/marketplace/rbee-workers/${encodeURIComponent(workerId)}`)}
        />
      )}
    </PageContainer>
  );
}
