// TEAM-391: Inpainting React hook
// Pattern: Same as useTextToImage

import { useMutation } from '@tanstack/react-query'
import { useCallback, useState } from 'react'
import type { GenerationProgress, GenerationResult, InpaintingParams } from './types'

export interface UseInpaintingOptions {
  baseUrl: string
  workerId: string
}

export function useInpainting({ baseUrl, workerId }: UseInpaintingOptions) {
  const [progress, setProgress] = useState<GenerationProgress | null>(null)

  const mutation = useMutation({
    mutationFn: async (params: InpaintingParams): Promise<GenerationResult> => {
      // TODO: TEAM-392+ will implement this
      console.log('🎨 [SD Worker React] useInpainting called (stub)', params)

      return {
        imageBase64: params.initImage, // Echo back for now
        seed: params.seed || Math.floor(Math.random() * 1000000),
        params,
      }
    },
  })

  const generate = useCallback(
    (params: InpaintingParams) => {
      setProgress(null)
      return mutation.mutateAsync(params)
    },
    [mutation],
  )

  return {
    generate,
    isLoading: mutation.isPending,
    progress,
    result: mutation.data,
    error: mutation.error,
    reset: mutation.reset,
  }
}
