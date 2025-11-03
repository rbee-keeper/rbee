// TEAM-391: Text-to-image React hook
// Pattern: TanStack Query for async state management

import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import type { TextToImageParams, GenerationProgress, GenerationResult } from './types';

export interface UseTextToImageOptions {
  baseUrl: string;
  workerId: string;
}

export function useTextToImage({ baseUrl, workerId }: UseTextToImageOptions) {
  const [progress, setProgress] = useState<GenerationProgress | null>(null);

  const mutation = useMutation({
    mutationFn: async (params: TextToImageParams): Promise<GenerationResult> => {
      // TODO: TEAM-392+ will implement this using @rbee/sd-worker-sdk
      // For now, return a stub
      console.log('🎨 [SD Worker React] useTextToImage called (stub)', params);
      
      // Simulate progress
      for (let i = 1; i <= (params.steps || 20); i++) {
        setProgress({
          step: i,
          total: params.steps || 20,
          percentage: (i / (params.steps || 20)) * 100,
        });
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      return {
        imageBase64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
        seed: params.seed || Math.floor(Math.random() * 1000000),
        params,
      };
    },
    onSuccess: () => {
      setProgress(null);
    },
    onError: () => {
      setProgress(null);
    },
  });

  const generate = useCallback(
    (params: TextToImageParams) => {
      setProgress(null);
      return mutation.mutateAsync(params);
    },
    [mutation]
  );

  return {
    generate,
    isLoading: mutation.isPending,
    progress,
    result: mutation.data,
    error: mutation.error,
    reset: mutation.reset,
  };
}
