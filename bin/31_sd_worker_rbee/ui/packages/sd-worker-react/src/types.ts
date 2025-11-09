// TEAM-391: Shared types for SD Worker React hooks

export interface TextToImageParams {
  prompt: string
  negativePrompt?: string
  steps?: number
  guidanceScale?: number
  seed?: number
  width?: number
  height?: number
}

export interface ImageToImageParams extends TextToImageParams {
  initImage: string // base64
  strength: number // 0.0-1.0
}

export interface InpaintingParams extends TextToImageParams {
  initImage: string // base64
  maskImage: string // base64
}

export interface GenerationProgress {
  step: number
  total: number
  percentage: number
}

export interface GenerationResult {
  imageBase64: string
  seed: number
  params: TextToImageParams | ImageToImageParams | InpaintingParams
}
