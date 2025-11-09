// TEAM-421: Unified artifact detail page template
// Used by both ModelDetailPageTemplate and WorkerDetailPageTemplate for consistency
//! Generic presentation layer for marketplace artifacts (models, workers, etc.)

import { Badge, Button, Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { ArrowLeft } from 'lucide-react'
import type { ReactNode } from 'react'

export interface ArtifactDetailPageTemplateProps {
  /** Artifact name (displayed as H1) */
  name: string

  /** Artifact description */
  description: string

  /** Author/creator name (optional) */
  author?: string

  /** Primary action button (e.g., "Download Model", "Install Worker") */
  primaryAction?: {
    label: string
    icon?: ReactNode
    onClick: () => void
    disabled?: boolean
  }

  /** Secondary action button (e.g., "View on HuggingFace", "View on GitHub") */
  secondaryAction?: {
    label: string
    icon?: ReactNode
    href: string
  }

  /** Back button configuration */
  backButton?: {
    label: string
    onClick: () => void
  }

  /** Stats to display below title (downloads, likes, size, etc.) */
  stats?: Array<{
    icon: ReactNode
    value: string | number
    label: string
  }>

  /** Badges to display (version, type, license, etc.) */
  badges?: Array<{
    label: string
    variant?: 'default' | 'secondary' | 'outline' | 'accent'
  }>

  /** Left sidebar content (model files, worker info, etc.) */
  leftSidebar?: ReactNode

  /** Main content sections (cards, descriptions, etc.) */
  mainContent: ReactNode

  /** Loading state */
  isLoading?: boolean
}

/**
 * Unified artifact detail page template
 *
 * Provides consistent layout and styling for all marketplace artifacts.
 * Used by ModelDetailPageTemplate and WorkerDetailPageTemplate.
 *
 * Layout:
 * - Back button
 * - Hero header (name, author, stats, badges, actions)
 * - Two-column grid (left sidebar + main content)
 *
 * @example Model Usage
 * ```tsx
 * <ArtifactDetailPageTemplate
 *   name={model.name}
 *   author={model.author}
 *   description={model.description}
 *   stats={[
 *     { icon: <Download />, value: model.downloads, label: 'downloads' },
 *     { icon: <Heart />, value: model.likes, label: 'likes' },
 *   ]}
 *   primaryAction={{
 *     label: 'Download Model',
 *     icon: <Download />,
 *     onClick: handleDownload
 *   }}
 *   leftSidebar={<ModelFilesList files={model.siblings} />}
 *   mainContent={<ModelDetailsCards model={model} />}
 * />
 * ```
 *
 * @example Worker Usage
 * ```tsx
 * <ArtifactDetailPageTemplate
 *   name={worker.name}
 *   description={worker.description}
 *   badges={[
 *     { label: `v${worker.version}`, variant: 'outline' },
 *     { label: worker.workerType, variant: 'default' },
 *   ]}
 *   primaryAction={{
 *     label: 'Install Worker',
 *     icon: <Download />,
 *     onClick: handleInstall
 *   }}
 *   mainContent={<WorkerDetailsCards worker={worker} />}
 * />
 * ```
 */
export function ArtifactDetailPageTemplate({
  name,
  description,
  author,
  primaryAction,
  secondaryAction,
  backButton,
  stats,
  badges,
  leftSidebar,
  mainContent,
  isLoading = false,
}: ArtifactDetailPageTemplateProps) {
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
      {backButton && (
        <Button variant="ghost" size="sm" onClick={backButton.onClick} className="mb-4">
          <ArrowLeft className="size-4 mr-2" />
          {backButton.label}
        </Button>
      )}

      {/* Hero Header */}
      <header className="space-y-6">
        <div className="space-y-4">
          {/* Artifact name and author */}
          <div>
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-2">{name}</h1>
            {author && (
              <p className="text-xl text-muted-foreground">
                by <span className="font-semibold">{author}</span>
              </p>
            )}
          </div>

          {/* Stats bar (downloads, likes, size, etc.) */}
          {stats && stats.length > 0 && (
            <div className="flex flex-wrap items-center gap-6 text-sm">
              {stats.map((stat, index) => (
                <div key={index} className="flex items-center gap-2">
                  <span className="text-muted-foreground">{stat.icon}</span>
                  <span className="font-semibold">
                    {typeof stat.value === 'number' ? stat.value.toLocaleString() : stat.value}
                  </span>
                  <span className="text-muted-foreground">{stat.label}</span>
                </div>
              ))}
            </div>
          )}

          {/* Badges (version, type, license, etc.) */}
          {badges && badges.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {badges.map((badge, index) => (
                <Badge key={index} variant={badge.variant || 'secondary'}>
                  {badge.label}
                </Badge>
              ))}
            </div>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap gap-3">
            {primaryAction && (
              <Button size="lg" onClick={primaryAction.onClick} disabled={primaryAction.disabled}>
                {primaryAction.icon}
                {primaryAction.label}
              </Button>
            )}
            {secondaryAction && (
              <Button variant="outline" size="lg" asChild>
                <a href={secondaryAction.href} target="_blank" rel="noopener noreferrer">
                  {secondaryAction.icon}
                  {secondaryAction.label}
                </a>
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left column - Sidebar (model files, worker info, etc.) */}
        {leftSidebar && <aside className="lg:col-span-1 space-y-6">{leftSidebar}</aside>}

        {/* Right column - Main content */}
        <article className={leftSidebar ? 'lg:col-span-2 space-y-6' : 'lg:col-span-3 space-y-6'}>
          {/* Description card */}
          <section>
            <Card>
              <CardHeader>
                <CardTitle>About</CardTitle>
              </CardHeader>
              <CardContent>
                <div 
                  className="text-muted-foreground leading-relaxed prose prose-sm dark:prose-invert max-w-none"
                  dangerouslySetInnerHTML={{ __html: description }}
                />
              </CardContent>
            </Card>
          </section>

          {/* Additional content sections */}
          {mainContent}
        </article>
      </div>
    </div>
  )
}
