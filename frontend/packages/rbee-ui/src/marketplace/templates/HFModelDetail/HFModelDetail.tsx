// TEAM-405: HuggingFace model detail template
// TEAM-410: Added compatibility section
// TEAM-421: Refactored to use ArtifactDetailPageTemplate for consistency
// TEAM-463: Renamed from ModelDetailPageTemplate to HuggingFaceModelDetail (Rule Zero)
//! Complete presentation layer for HuggingFace LLM model details
//! SEO-compatible, works with Tauri, Next.js SSG/SSR

import { Badge, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import {
  Calendar,
  CheckCircle2,
  Code,
  Cpu,
  Download,
  ExternalLink,
  HardDrive,
  Hash,
  Heart,
  Languages,
  MessageSquare,
  Shield,
} from 'lucide-react'
import { ModelFilesList } from '../../molecules/ModelFilesList'
import { ModelMetadataCard } from '../../molecules/ModelMetadataCard'
import { WorkerCompatibilityList } from '../../organisms/WorkerCompatibilityList'
import type { CompatibilityResult, Worker } from '../../types/compatibility'
import { ArtifactDetailPageTemplate } from '../ArtifactDetailPageTemplate'

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
  sha?: string
  config?: {
    architectures?: string[]
    model_type?: string
    tokenizer_config?: {
      bos_token?: string | { content?: string; [key: string]: any }
      eos_token?: string | { content?: string; [key: string]: any }
      chat_template?: string
    }
  }
  cardData?: {
    base_model?: string
    license?: string
    language?: string[]
  }
  // TEAM-463: ⚠️ TYPE CONTRACT - must match artifacts-contract::ModelFile
  siblings?: Array<{ filename: string; size?: number | null }>
  widgetData?: Array<{ text: string }>
  createdAt?: string
  lastModified?: string

  // TEAM-427: CivitAI specific (optional)
  images?: Array<{ url: string; nsfw?: boolean }>
  externalUrl?: string // CivitAI or HuggingFace URL
  externalLabel?: string // "View on CivitAI" or "View on HuggingFace"
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
  onBack,
  onDownload,
  huggingFaceUrl,
  showBackButton = true,
  isLoading = false,
  compatibleWorkers,
}: HFModelDetailPageTemplateProps) {
  // TEAM-427: Use externalUrl from model data, fallback to HuggingFace
  const externalUrl = model.externalUrl || huggingFaceUrl || `https://huggingface.co/${model.id}`
  const externalLabel = model.externalLabel || 'View on HuggingFace'

  // Left sidebar content (model files)
  const leftSidebar = model.siblings && model.siblings.length > 0 ? <ModelFilesList files={model.siblings} /> : null

  // Main content sections
  const mainContent = (
    <>
      {/* TEAM-427: Image Gallery for CivitAI models */}
      {model.images && model.images.length > 0 && (
        <section>
          <Card>
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
        </section>
      )}

      {/* TEAM-410: Compatible Workers Section */}
      {compatibleWorkers && compatibleWorkers.length > 0 && (
        <section>
          <Card>
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
        </section>
      )}

      {/* Basic Info */}
      <section>
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
      </section>

      {/* Model Config */}
      {model.config && (
        <section>
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
        </section>
      )}

      {/* Chat Template */}
      {model.config?.tokenizer_config?.chat_template && (
        <section>
          <Card>
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
        </section>
      )}

      {/* Card Data */}
      {model.cardData && (
        <section>
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
                        model.cardData.language.slice(0, 5).join(', ') +
                        (model.cardData.language.length > 5 ? '...' : ''),
                      icon: <Languages className="size-4" />,
                    },
                  ]
                : []),
            ]}
          />
        </section>
      )}

      {/* Timestamps */}
      {(model.createdAt || model.lastModified) && (
        <section>
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
        </section>
      )}

      {/* Example Prompts */}
      {model.widgetData && model.widgetData.length > 0 && (
        <section>
          <Card>
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
        </section>
      )}

      {/* Tags */}
      {model.tags && model.tags.length > 0 && (
        <section className="pt-6 border-t">
          <h2 className="text-lg font-semibold mb-3">Tags</h2>
          <div className="flex flex-wrap gap-2">
            {model.tags.map((tag) => (
              <Badge key={tag} variant="outline" className="text-sm">
                {tag}
              </Badge>
            ))}
          </div>
        </section>
      )}
    </>
  )

  return (
    <ArtifactDetailPageTemplate
      name={model.name}
      author={model.author}
      description={model.description}
      isLoading={isLoading}
      backButton={
        showBackButton && onBack
          ? {
              label: 'Back to Models',
              onClick: onBack,
            }
          : undefined
      }
      stats={[
        { icon: <Download className="size-4" />, value: model.downloads, label: 'downloads' },
        { icon: <Heart className="size-4" />, value: model.likes, label: 'likes' },
        { icon: <HardDrive className="size-4" />, value: model.size, label: '' },
      ]}
      badges={model.pipeline_tag ? [{ label: model.pipeline_tag, variant: 'secondary' }] : undefined}
      primaryAction={
        onDownload
          ? {
              label: 'Download Model',
              icon: <Download className="size-4 mr-2" />,
              onClick: onDownload,
            }
          : undefined
      }
      secondaryAction={{
        label: externalLabel,
        icon: <ExternalLink className="size-4 mr-2" />,
        href: externalUrl,
      }}
      leftSidebar={leftSidebar}
      mainContent={mainContent}
    />
  )
}
