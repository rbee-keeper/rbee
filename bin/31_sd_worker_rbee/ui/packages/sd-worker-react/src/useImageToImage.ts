// TEAM-391: Image-to-image React hook
// Pattern: Same as useTextToImage

import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import type { ImageToImageParams, GenerationProgress, GenerationResult } from './types';

export interface UseImageToImageOptions {
  baseUrl: string;
  workerId: string;
}

export function useImageToImage({ baseUrl, workerId }: UseImageToImageOptions) {
  const [progress, setProgress] = useState<GenerationProgress | null>(null);

  const mutation = useMutation({
    mutationFn: async (params: ImageToImageParams): Promise<GenerationResult> => {
      // TODO: TEAM-392+ will implement this
      console.log('🎨 [SD Worker React] useImageToImage called (stub)', params);
      
      return {
        imageBase64: params.initImage, // Echo back for now
        seed: params.seed || Math.floor(Math.random() * 1000000),
        params,
      };
    },
  });

  const generate = useCallback(
    (params: ImageToImageParams) => {
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
