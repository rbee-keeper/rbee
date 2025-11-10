// TEAM-405: HuggingFace model detail template
// TEAM-410: Added compatibility section
// TEAM-421: Refactored to use ArtifactDetailPageTemplate for consistency
// TEAM-463: Renamed from ModelDetailPageTemplate to HuggingFaceModelDetail (Rule Zero)
// TEAM-464: Refactored to match CivitAI layout structure (files on right sidebar)
//! Complete presentation layer for HuggingFace LLM model details
//! SEO-compatible, works with Tauri, Next.js SSG/SSR

import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Separator } from '@rbee/ui/atoms'
import {
  Calendar,
  CheckCircle2,
  Code,
  Cpu,
  Download,
  ExternalLink,
  Hash,
  HardDrive,
  Languages,
  Layers,
  MessageSquare,
  Package,
  Shield,
  Tag,
} from 'lucide-react'
import { MarkdownContent } from '@rbee/ui/molecules'
import { DatasetsUsedCard } from '../../molecules/DatasetsUsedCard'
import { InferenceProvidersCard } from '../../molecules/InferenceProvidersCard'
import { ModelFilesList } from '../../molecules/ModelFilesList'
import { ModelMetadataCard } from '../../molecules/ModelMetadataCard'
import { WidgetDataCard } from '../../molecules/WidgetDataCard'
import { CivitAIStatsHeader } from '../../organisms/CivitAIStatsHeader'
import { WorkerCompatibilityList } from '../../organisms/WorkerCompatibilityList'
import type { CompatibilityResult, Worker } from '../../types/compatibility'

export interface HFModelDetailData {
  // Basic fields (always present)
  id: string
  name: string
  description: string
  author?: string
  downloads: number
  likes: number
  size: string
  tags: string[]

  // HuggingFace specific (optional)
  pipeline_tag?: string
  library_name?: string
  sha?: string
  mask_token?: string
  
  // TEAM-464: Widget data for inference examples
  widgetData?: Array<{
    source_sentence?: string
    sentences?: string[]
    text?: string
  }>
  
  config?: {
    architectures?: string[]
    model_type?: string
    tokenizer_config?: {
      unk_token?: string
      sep_token?: string
      pad_token?: string
      cls_token?: string
      mask_token?: string
      bos_token?: string | { content?: string; [key: string]: any }
      eos_token?: string | { content?: string; [key: string]: any }
      chat_template?: string
    }
  }
  
  // TEAM-464: Extended card data
  cardData?: {
    base_model?: string
    license?: string
    language?: string | string[]
    datasets?: string[]
    pipeline_tag?: string
  }
  
  // TEAM-464: Transformers info for inference
  transformersInfo?: {
    auto_model?: string
    pipeline_tag?: string
    processor?: string
  }
  
  // TEAM-464: Inference status
  inference?: 'warm' | 'cold' | string
  
  // TEAM-464: Safetensors parameters
  safetensors?: {
    parameters?: {
      I64?: number
      F32?: number
      [key: string]: number | undefined
    }
    total?: number
  }
  
  // TEAM-464: Spaces using this model
  spaces?: string[]
  
  // TEAM-463: ⚠️ TYPE CONTRACT - must match artifacts-contract::ModelFile
  siblings?: Array<{ filename: string; size?: number | null }>
  createdAt?: string
  lastModified?: string

  // TEAM-427: CivitAI specific (optional)
  images?: Array<{ url: string; nsfw?: boolean }>
  externalUrl?: string // CivitAI or HuggingFace URL
  externalLabel?: string // "View on CivitAI" or "View on HuggingFace"
  
  // TEAM-464: README content (raw markdown for react-markdown)
  readmeMarkdown?: string
}

export interface HFModelDetailPageTemplateProps {
  /** Model data to display */
  model: HFModelDetailData

  /** Called when back button is clicked */
  onBack?: () => void

  /** Called when download button is clicked */
  onDownload?: () => void

  /** HuggingFace URL (optional, defaults to https://huggingface.co/{id}) */
  huggingFaceUrl?: string

  /** Show back button */
  showBackButton?: boolean

  /** Loading state */
  isLoading?: boolean

  /** TEAM-410: Compatible workers with compatibility results */
  compatibleWorkers?: Array<{
    worker: Worker
    compatibility: CompatibilityResult
  }>
}

const formatDate = (dateString: string): string => {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  })
}

/**
 * Complete model detail page template
 *
 * SEO-compatible with semantic HTML and proper heading hierarchy.
 * Works with any data source (Tauri, Next.js SSG/SSR, API).
 *
 * @example Tauri
 * ```tsx
 * const { data: model } = useQuery({
 *   queryFn: () => invoke('marketplace_get_model', { modelId })
 * })
 *
 * <ModelDetailPageTemplate
 *   model={model}
 *   onBack={() => navigate('/marketplace')}
 *   onDownload={() => handleDownload(model.id)}
 * />
 * ```
 *
 * @example Next.js SSG
 * ```tsx
 * export async function generateMetadata({ params }) {
 *   const model = await fetchModel(params.id)
 *   return {
 *     title: model.name,
 *     description: model.description
 *   }
 * }
 *
 * <ModelDetailPageTemplate
 *   model={model}
 *   showBackButton={false}
 * />
 * ```
 */
export function HFModeldetail({
  model,
  onDownload,
  huggingFaceUrl,
  isLoading = false,
  compatibleWorkers,
}: HFModelDetailPageTemplateProps) {
  // TEAM-427: Use externalUrl from model data, fallback to HuggingFace
  const externalUrl = model.externalUrl || huggingFaceUrl || `https://huggingface.co/${model.id}`
  const externalLabel = model.externalLabel || 'View on HuggingFace'

  if (isLoading) {
    return (
      <div className="animate-pulse space-y-6">
        <div className="h-64 bg-muted/20 rounded-lg" />
        <div className="h-32 bg-muted/20 rounded-lg" />
        <div className="h-48 bg-muted/20 rounded-lg" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Stats Header */}
      <CivitAIStatsHeader
        downloads={model.downloads || 0}
        likes={model.likes || 0}
        rating={0} // HF doesn't have ratings, show 0
      />

      {/* Main Content Grid - 3 columns: 2 for content, 1 for sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Columns (span 2) - README & Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* TEAM-464: README / Model Card */}
          {model.readmeMarkdown && (
            <MarkdownContent markdown={model.readmeMarkdown} title="Model Card" />
          )}

          {/* TEAM-427: Image Gallery for CivitAI models */}
          {model.images && model.images.length > 0 && (
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Example Images</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {model.images.map((image, index) => (
                    <div key={index} className="relative aspect-square rounded-lg overflow-hidden bg-muted">
                      <img
                        src={image.url}
                        alt={`Example ${index + 1}`}
                        className="w-full h-full object-cover hover:scale-105 transition-transform duration-200"
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* TEAM-410: Compatible Workers Section */}
          {compatibleWorkers && compatibleWorkers.length > 0 && (
            <Card className="shadow-lg">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="size-5" />
                  <CardTitle>Compatible Workers</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <WorkerCompatibilityList workers={compatibleWorkers} />
              </CardContent>
            </Card>
          )}

          {/* TEAM-464: Inference Providers */}
          <InferenceProvidersCard
            inferenceStatus={model.inference}
            libraryName={model.library_name}
            transformersInfo={model.transformersInfo}
          />

          {/* TEAM-464: Widget Data / Usage Examples */}
          <WidgetDataCard widgetData={model.widgetData} pipelineTag={model.pipeline_tag} />

          {/* TEAM-464: Datasets Used */}
          <DatasetsUsedCard datasets={model.cardData?.datasets} />

          {/* Basic Info */}
          <ModelMetadataCard
            title="Basic Information"
            items={[
              { label: 'Model ID', value: model.id, code: true },
              ...(model.author ? [{ label: 'Author', value: model.author }] : []),
              ...(model.pipeline_tag
                ? [{ label: 'Pipeline', value: model.pipeline_tag, icon: <Code className="size-4" /> }]
                : []),
              ...(model.sha
                ? [{ label: 'SHA', value: `${model.sha.substring(0, 12)}...`, icon: <Hash className="size-4" /> }]
                : []),
            ]}
          />

          {/* Model Config */}
          {model.config && (
            <ModelMetadataCard
              title="Model Configuration"
              items={[
                ...(model.config.architectures
                  ? [
                      {
                        label: 'Architecture',
                        value: model.config.architectures[0],
                        icon: <Cpu className="size-4" />,
                      },
                    ]
                  : []),
                ...(model.config.model_type
                  ? [
                      {
                        label: 'Model Type',
                        value: model.config.model_type,
                      },
                    ]
                  : []),
                ...(model.config.tokenizer_config?.bos_token
                  ? [
                      {
                        label: 'BOS Token',
                        value:
                          typeof model.config.tokenizer_config.bos_token === 'string'
                            ? model.config.tokenizer_config.bos_token
                            : model.config.tokenizer_config.bos_token.content ||
                              JSON.stringify(model.config.tokenizer_config.bos_token),
                        code: true,
                      },
                    ]
                  : []),
                ...(model.config.tokenizer_config?.eos_token
                  ? [
                      {
                        label: 'EOS Token',
                        value:
                          typeof model.config.tokenizer_config.eos_token === 'string'
                            ? model.config.tokenizer_config.eos_token
                            : model.config.tokenizer_config.eos_token.content ||
                              JSON.stringify(model.config.tokenizer_config.eos_token),
                        code: true,
                      },
                    ]
                  : []),
              ]}
            />
          )}

          {/* Chat Template */}
          {model.config?.tokenizer_config?.chat_template && (
            <Card className="shadow-lg">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <MessageSquare className="size-4" />
                  <CardTitle>Chat Template</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <code className="block text-xs bg-muted p-4 rounded font-mono whitespace-pre-wrap break-all">
                  {model.config.tokenizer_config.chat_template}
                </code>
              </CardContent>
            </Card>
          )}

          {/* Card Data */}
          {model.cardData && (
            <ModelMetadataCard
              title="Additional Information"
              items={[
                ...(model.cardData.base_model
                  ? [
                      {
                        label: 'Base Model',
                        value: model.cardData.base_model,
                      },
                    ]
                  : []),
                ...(model.cardData.license
                  ? [
                      {
                        label: 'License',
                        value: model.cardData.license,
                        icon: <Shield className="size-4" />,
                      },
                    ]
                  : []),
                ...(model.cardData.language
                  ? [
                      {
                        label: 'Languages',
                        value:
                          typeof model.cardData.language === 'string'
                            ? model.cardData.language
                            : model.cardData.language.slice(0, 5).join(', ') +
                              (model.cardData.language.length > 5 ? '...' : ''),
                        icon: <Languages className="size-4" />,
                      },
                    ]
                  : []),
              ]}
            />
          )}

          {/* Timestamps */}
          {(model.createdAt || model.lastModified) && (
            <ModelMetadataCard
              title="Timeline"
              items={[
                ...(model.createdAt
                  ? [
                      {
                        label: 'Created',
                        value: formatDate(model.createdAt),
                        icon: <Calendar className="size-4" />,
                      },
                    ]
                  : []),
                ...(model.lastModified
                  ? [
                      {
                        label: 'Last Modified',
                        value: formatDate(model.lastModified),
                        icon: <Calendar className="size-4" />,
                      },
                    ]
                  : []),
              ]}
            />
          )}

          {/* Example Prompts */}
          {model.widgetData && model.widgetData.length > 0 && (
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle>Example Prompts</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {model.widgetData.map((widget, index) => (
                  <div key={index} className="p-3 bg-muted/50 rounded-md">
                    <p className="text-sm">{widget.text}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Column (span 1) - Details Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Model Details Card */}
          <Card className="p-6 space-y-4 shadow-lg">
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <Package className="size-5 text-primary" />
              Details
            </h3>
            <div className="space-y-3">
              <div>
                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Package className="size-4 text-purple-500" />
                    <span>Type</span>
                  </div>
                  <Badge variant="secondary" className="font-medium">
                    {model.pipeline_tag || 'Model'}
                  </Badge>
                </div>
                <Separator />
              </div>
              {model.cardData?.base_model && (
                <div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Layers className="size-4 text-blue-500" />
                      <span>Base Model</span>
                    </div>
                    <span className="font-medium text-sm">{model.cardData.base_model}</span>
                  </div>
                  <Separator />
                </div>
              )}
              <div>
                <div className="flex items-center justify-between py-2">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <HardDrive className="size-4 text-orange-500" />
                    <span>Size</span>
                  </div>
                  <span className="font-medium text-sm">{model.size}</span>
                </div>
                {model.cardData?.license && <Separator />}
              </div>
              {model.cardData?.license && (
                <div>
                  <div className="flex items-center justify-between py-2">
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Shield className="size-4 text-green-500" />
                      <span>License</span>
                    </div>
                    <Badge variant="secondary" className="font-medium">
                      {model.cardData.license}
                    </Badge>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Files Card - matching CivitAI position */}
          {model.siblings && model.siblings.length > 0 && (
            <Card className="p-6 space-y-4 shadow-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Tag className="size-5 text-primary" />
                  <h3 className="font-semibold text-lg">
                    {model.siblings.length} {model.siblings.length === 1 ? 'File' : 'Files'}
                  </h3>
                </div>
              </div>
              <ModelFilesList files={model.siblings} />
            </Card>
          )}

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <Card className="p-6 space-y-4 shadow-lg">
              <h3 className="font-semibold text-lg">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {model.tags.slice(0, 10).map((tag) => (
                  <Badge key={tag} variant="outline" className="text-xs hover:bg-primary hover:text-primary-foreground transition-colors cursor-default">
                    {tag}
                  </Badge>
                ))}
              </div>
            </Card>
          )}

          {/* External Link */}
          <Button
            variant="outline"
            size="lg"
            className="w-full shadow-md hover:shadow-lg transition-all"
            asChild
          >
            <a href={externalUrl} target="_blank" rel="noopener noreferrer">
              <ExternalLink className="size-4 mr-2" />
              {externalLabel}
            </a>
          </Button>

          {/* Download Button */}
          {onDownload && (
            <Button
              size="lg"
              className="w-full shadow-md hover:shadow-lg transition-all"
              onClick={onDownload}
            >
              <Download className="size-4 mr-2" />
              Download Model
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
