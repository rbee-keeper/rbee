// TEAM-405: Model details page - Using reusable template
// TEAM-411: Added compatibility checking
// TEAM-421: Added environment-aware actions
// DATA LAYER: Tauri commands + React Query
// PRESENTATION: ModelDetailPageTemplate from rbee-ui

import { useParams, useNavigate } from "react-router-dom";
import { invoke } from "@tauri-apps/api/core";
import { useQuery } from "@tanstack/react-query";
import { PageContainer } from "@rbee/ui/molecules";
import { Button, Card, CardContent } from "@rbee/ui/atoms";
import { ModelDetailPageTemplate, useArtifactActions } from "@rbee/ui/marketplace";
import type { ModelDetailData } from "@rbee/ui/marketplace";
import { ArrowLeft, Sparkles } from "lucide-react";
import type { Model } from "@/generated/bindings";
import { checkModelCompatibility } from "@/api/compatibility";

export function ModelDetailsPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const navigate = useNavigate();
  
  // TEAM-421: Environment-aware actions
  const actions = useArtifactActions({
    onActionSuccess: (action) => {
      console.log(`✅ ${action} started successfully`);
    },
    onActionError: (action, error) => {
      console.error(`❌ ${action} failed:`, error);
    },
  });

  // Fetch the specific model by ID
  const { data: rawModel, isLoading, error } = useQuery({
    queryKey: ["marketplace", "model", modelId],
    queryFn: async () => {
      if (!modelId) throw new Error("No model ID provided");
      const result = await invoke<Model>("marketplace_get_model", {
        modelId: decodeURIComponent(modelId),
      });
      return result;
    },
    enabled: !!modelId,
    staleTime: 5 * 60 * 1000,
  });

  // TEAM-411: Check compatibility with all worker types
  const { data: compatibilityData } = useQuery({
    queryKey: ["compatibility", modelId],
    queryFn: async () => {
      if (!modelId) return null;
      const decodedId = decodeURIComponent(modelId);
      
      // Check compatibility with all worker types
      const [cpuCompat, cudaCompat, metalCompat] = await Promise.all([
        checkModelCompatibility(decodedId, 'cpu').catch(() => null),
        checkModelCompatibility(decodedId, 'cuda').catch(() => null),
        checkModelCompatibility(decodedId, 'metal').catch(() => null),
      ]);
      
      return {
        cpu: cpuCompat,
        cuda: cudaCompat,
        metal: metalCompat,
      };
    },
    enabled: !!modelId,
    staleTime: 60 * 60 * 1000, // Cache for 1 hour
  });

  // Transform Model to ModelDetailData format
  const model: ModelDetailData | undefined = rawModel ? {
    id: rawModel.id,
    name: rawModel.name,
    description: rawModel.description,
    author: rawModel.author,
    downloads: rawModel.downloads,
    likes: rawModel.likes,
    size: rawModel.size,
    tags: rawModel.tags,
    // Pass through all HuggingFace-specific fields
    ...(rawModel as any)
  } : undefined;

  // TEAM-411: Build compatible workers array for template
  const compatibleWorkers = compatibilityData ? [
    compatibilityData.cpu && {
      worker: {
        id: 'cpu',
        name: 'CPU Worker',
        worker_type: 'cpu' as const,
        platform: ['linux', 'macos', 'windows'],
      },
      compatibility: compatibilityData.cpu,
    },
    compatibilityData.cuda && {
      worker: {
        id: 'cuda',
        name: 'CUDA Worker',
        worker_type: 'cuda' as const,
        platform: ['linux'],
      },
      compatibility: compatibilityData.cuda,
    },
    compatibilityData.metal && {
      worker: {
        id: 'metal',
        name: 'Metal Worker',
        worker_type: 'metal' as const,
        platform: ['macos'],
      },
      compatibility: compatibilityData.metal,
    },
  ].filter(Boolean) as any[] : undefined;

  if (isLoading) {
    return (
      <PageContainer title="Loading..." description="Fetching model details" padding="default">
        <div className="animate-pulse space-y-6">
          <div className="h-64 bg-muted/20 rounded-lg" />
          <div className="h-32 bg-muted/20 rounded-lg" />
          <div className="h-48 bg-muted/20 rounded-lg" />
        </div>
      </PageContainer>
    );
  }

  if (error || !model) {
    return (
      <PageContainer title="Model Not Found" description="The requested model could not be found" padding="default">
        <Card>
          <CardContent className="p-12 text-center">
            <Sparkles className="size-16 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-xl font-semibold mb-2">Model not found</h3>
            <p className="text-muted-foreground mb-6">
              {error
                ? `Error: ${String(error)}`
                : "The model you're looking for doesn't exist or has been removed."}
            </p>
            <Button onClick={() => navigate("/marketplace/llm-models")}>
              <ArrowLeft className="size-4 mr-2" />
              Back to Models
            </Button>
          </CardContent>
        </Card>
      </PageContainer>
    );
  }

  return (
    <PageContainer
      title={model.name}
      description={model.author ? `by ${model.author}` : "Language Model"}
      padding="default"
    >
      <ModelDetailPageTemplate
        model={model}
        onBack={() => navigate("/marketplace/llm-models")}
        onDownload={() => actions.downloadModel(model.id)}
        isLoading={isLoading}
        compatibleWorkers={compatibleWorkers}
      />
    </PageContainer>
  );
}
