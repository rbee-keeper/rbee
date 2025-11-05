// TEAM-405: Model detail page template
// TEAM-410: Added compatibility section
//! Complete presentation layer for model details
//! SEO-compatible, works with Tauri, Next.js SSG/SSR

import { Button, Badge, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ModelMetadataCard } from '../../molecules/ModelMetadataCard'
import { ModelFilesList } from '../../molecules/ModelFilesList'
import { WorkerCompatibilityList } from '../../organisms/WorkerCompatibilityList'
import type { CompatibilityResult, Worker } from '../../types/compatibility'
import { 
  ArrowLeft, 
  ExternalLink, 
  Code, 
  Languages, 
  Shield, 
  Calendar,
  Hash,
  Cpu,
  MessageSquare,
  CheckCircle2,
  Download,
  Heart,
  HardDrive
} from 'lucide-react'

export interface ModelDetailData {
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
  siblings?: Array<{ rfilename: string }>
  widgetData?: Array<{ text: string }>
  createdAt?: string
  lastModified?: string
}

export interface ModelDetailPageTemplateProps {
  /** Model data to display */
  model: ModelDetailData
  
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
    day: 'numeric'
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
export function ModelDetailPageTemplate({
  model,
  onBack,
  onDownload,
  huggingFaceUrl,
  showBackButton = true,
  isLoading = false,
  compatibleWorkers
}: ModelDetailPageTemplateProps) {
  const hfUrl = huggingFaceUrl || `https://huggingface.co/${model.id}`

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
    <div className="space-y-8">
      {/* Back button */}
      {showBackButton && onBack && (
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="mb-4"
        >
          <ArrowLeft className="size-4 mr-2" />
          Back to Models
        </Button>
      )}

      {/* Hero Header */}
      <header className="space-y-6">
        <div className="space-y-4">
          {/* Model name and author */}
          <div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-2">
              {model.name}
            </h1>
            {model.author && (
              <p className="text-xl text-muted-foreground">
                by <span className="font-semibold">{model.author}</span>
              </p>
            )}
          </div>

          {/* Primary stats bar */}
          <div className="flex flex-wrap items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <Download className="size-4 text-muted-foreground" />
              <span className="font-semibold">{model.downloads.toLocaleString()}</span>
              <span className="text-muted-foreground">downloads</span>
            </div>
            <div className="flex items-center gap-2">
              <Heart className="size-4 text-muted-foreground" />
              <span className="font-semibold">{model.likes.toLocaleString()}</span>
              <span className="text-muted-foreground">likes</span>
            </div>
            <div className="flex items-center gap-2">
              <HardDrive className="size-4 text-muted-foreground" />
              <span className="font-semibold">{model.size}</span>
            </div>
            {model.pipeline_tag && (
              <Badge variant="secondary" className="text-sm">
                {model.pipeline_tag}
              </Badge>
            )}
          </div>

          {/* Action buttons */}
          <div className="flex flex-wrap gap-3">
            {onDownload && (
              <Button size="lg" onClick={onDownload}>
                <Download className="size-4 mr-2" />
                Download Model
              </Button>
            )}
            <Button variant="outline" size="lg" asChild>
              <a
                href={hfUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <ExternalLink className="size-4 mr-2" />
                View on HuggingFace
              </a>
            </Button>
          </div>
        </div>
      </header>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Model Files */}
        <aside className="lg:col-span-1 space-y-6">
          {/* Model Files */}
          {model.siblings && model.siblings.length > 0 && (
            <ModelFilesList files={model.siblings} />
          )}
        </aside>

        {/* Right column - Details */}
        <article className="lg:col-span-2 space-y-6">
          {/* Description */}
          <section>
            <Card>
              <CardHeader>
                <CardTitle>About</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground leading-relaxed">{model.description}</p>
              </CardContent>
            </Card>
          </section>

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
                ...(model.pipeline_tag ? [{ label: 'Pipeline', value: model.pipeline_tag, icon: <Code className="size-4" /> }] : []),
                ...(model.sha ? [{ label: 'SHA', value: model.sha.substring(0, 12) + '...', icon: <Hash className="size-4" /> }] : []),
              ]}
            />
          </section>

          {/* Model Config */}
          {model.config && (
            <section>
              <ModelMetadataCard
                title="Model Configuration"
                items={[
                  ...(model.config.architectures ? [{ 
                    label: 'Architecture', 
                    value: model.config.architectures[0],
                    icon: <Cpu className="size-4" />
                  }] : []),
                  ...(model.config.model_type ? [{ 
                    label: 'Model Type', 
                    value: model.config.model_type 
                  }] : []),
                  ...(model.config.tokenizer_config?.bos_token ? [{ 
                    label: 'BOS Token', 
                    value: typeof model.config.tokenizer_config.bos_token === 'string' 
                      ? model.config.tokenizer_config.bos_token 
                      : model.config.tokenizer_config.bos_token.content || JSON.stringify(model.config.tokenizer_config.bos_token),
                    code: true
                  }] : []),
                  ...(model.config.tokenizer_config?.eos_token ? [{ 
                    label: 'EOS Token', 
                    value: typeof model.config.tokenizer_config.eos_token === 'string' 
                      ? model.config.tokenizer_config.eos_token 
                      : model.config.tokenizer_config.eos_token.content || JSON.stringify(model.config.tokenizer_config.eos_token),
                    code: true
                  }] : []),
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
                  ...(model.cardData.base_model ? [{ 
                    label: 'Base Model', 
                    value: model.cardData.base_model 
                  }] : []),
                  ...(model.cardData.license ? [{ 
                    label: 'License', 
                    value: model.cardData.license,
                    icon: <Shield className="size-4" />
                  }] : []),
                  ...(model.cardData.language ? [{ 
                    label: 'Languages', 
                    value: model.cardData.language.slice(0, 5).join(', ') + (model.cardData.language.length > 5 ? '...' : ''),
                    icon: <Languages className="size-4" />
                  }] : []),
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
                  ...(model.createdAt ? [{ 
                    label: 'Created', 
                    value: formatDate(model.createdAt),
                    icon: <Calendar className="size-4" />
                  }] : []),
                  ...(model.lastModified ? [{ 
                    label: 'Last Modified', 
                    value: formatDate(model.lastModified),
                    icon: <Calendar className="size-4" />
                  }] : []),
                ]}
              />
            </section>
          )}

          {/* Tags */}
          <section>
            <Card>
              <CardHeader>
                <CardTitle>Tags</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {model.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-sm">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          </section>

          {/* Widget Data */}
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
        </article>
      </div>
    </div>
  )
}
