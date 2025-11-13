// TEAM-464: Reusable markdown content renderer
// Uses react-markdown with existing BlogHeading, CodeBlock, and other UI components
// Converts HTML to markdown for consistent styling

'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import ReactMarkdown from 'react-markdown'
import type { Components } from 'react-markdown'
import rehypeRaw from 'rehype-raw'
import remarkGfm from 'remark-gfm'
import { BlogHeading } from '../BlogHeading'
import { CodeBlock } from '../CodeBlock'

export interface MarkdownContentProps {
  /** Markdown content string */
  markdown?: string
  /** HTML content string (for backwards compatibility with CivitAI) */
  html?: string
  /** Optional title for the card */
  title?: string
  /** Additional CSS classes */
  className?: string
  /** Show as card (default: true) */
  asCard?: boolean
}

/**
 * MarkdownContent - Renders markdown using existing UI components
 * 
 * Uses react-markdown with custom components:
 * - BlogHeading for h1-h6
 * - CodeBlock for code blocks
 * - Proper table styling
 * - GitHub Flavored Markdown support
 * 
 * @example
 * ```tsx
 * const readmeMarkdown = await getHuggingFaceModelReadme(modelId)
 * 
 * <MarkdownContent 
 *   markdown={readmeMarkdown} 
 *   title="Model Card"
 * />
 * ```
 * 
 * @example Without card wrapper
 * ```tsx
 * <MarkdownContent 
 *   markdown={readmeMarkdown} 
 *   asCard={false}
 * />
 * ```
 */
export function MarkdownContent({ 
  markdown,
  html,
  title, 
  className,
  asCard = true 
}: MarkdownContentProps) {
  // Use markdown if provided, otherwise use HTML
  let content = markdown || html
  
  // If no content, return null
  if (!content) {
    return null
  }

  // TEAM-464: Strip YAML frontmatter from markdown (it's redundant metadata)
  if (content.startsWith('---')) {
    const endOfYaml = content.indexOf('---', 3)
    if (endOfYaml !== -1) {
      content = content.substring(endOfYaml + 3).trim()
    }
  }
  
  // Custom components that use our existing UI library
  const components: Components = {
    // Headings - use BlogHeading component
    h1: ({ children }) => <BlogHeading level="h2" variant="gradient">{children}</BlogHeading>,
    h2: ({ children }) => <BlogHeading level="h2">{children}</BlogHeading>,
    h3: ({ children }) => <BlogHeading level="h3">{children}</BlogHeading>,
    h4: ({ children }) => <BlogHeading level="h4">{children}</BlogHeading>,
    h5: ({ children }) => <BlogHeading level="h5">{children}</BlogHeading>,
    h6: ({ children }) => <BlogHeading level="h6">{children}</BlogHeading>,
    
    // Code blocks - use CodeBlock component
    code: (props) => {
      const { node, inline, className, children, ...rest } = props as any
      const match = /language-(\w+)/.exec(className || '')
      const codeString = String(children).replace(/\n$/, '')
      
      if (!inline && match) {
        return (
          <div className="my-6">
            <CodeBlock
              code={codeString}
              {...(match[1] ? { language: match[1] } : {})}
              copyable={true}
            />
          </div>
        )
      }
      
      // Inline code
      return (
        <code 
          className="bg-muted/50 border border-border/50 rounded px-1.5 py-0.5 text-[13px] font-mono break-words"
          {...rest}
        >
          {children}
        </code>
      )
    },
    
    // Paragraphs
    p: ({ children }) => (
      <p className="mb-5 leading-[1.75] text-foreground">{children}</p>
    ),
    
    // Links
    a: ({ href, children }) => (
      <a 
        href={href}
        className="text-blue-400 underline decoration-blue-400/50 underline-offset-2 hover:text-blue-300 hover:decoration-blue-300 transition-colors"
        target={href?.startsWith('http') ? '_blank' : undefined}
        rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
      >
        {children}
      </a>
    ),
    
    // Lists
    ul: ({ children }) => (
      <ul className="my-5 space-y-2 list-disc pl-7">{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className="my-5 space-y-2 list-decimal pl-7">{children}</ol>
    ),
    li: ({ children }) => (
      <li className="leading-[1.75] text-foreground">{children}</li>
    ),
    
    // Tables
    table: ({ children }) => (
      <div className="my-6 overflow-x-auto">
        <table className="w-full border-collapse border border-border text-sm">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-muted/50">{children}</thead>
    ),
    th: ({ children }) => (
      <th className="border border-border px-4 py-3 text-left font-semibold text-foreground">
        {children}
      </th>
    ),
    td: ({ children }) => (
      <td className="border border-border px-4 py-3 text-foreground">
        {children}
      </td>
    ),
    tr: ({ children }) => (
      <tr className="border-b border-border hover:bg-muted/5">
        {children}
      </tr>
    ),
    
    // Blockquotes
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-l-blue-500 pl-4 py-1 my-5 bg-muted/20 rounded-r text-foreground/90">
        {children}
      </blockquote>
    ),
    
    // Horizontal rules
    hr: () => (
      <hr className="border-t-2 border-border my-10" />
    ),
    
    // Strong and emphasis
    strong: ({ children }) => (
      <strong className="font-semibold text-foreground">{children}</strong>
    ),
    em: ({ children }) => (
      <em className="italic text-foreground">{children}</em>
    ),
    
    // TEAM-501: Form inputs - mark as readOnly to prevent React warnings
    input: (props) => {
      const { type, checked, ...rest } = props as any
      if (type === 'checkbox') {
        return (
          <input
            type="checkbox"
            checked={checked}
            readOnly
            className="mr-2 cursor-default"
            {...rest}
          />
        )
      }
      return <input readOnly {...props} />
    },
  }

  const markdownContent = (
    <div className={cn('markdown-content overflow-hidden', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )

  if (!asCard) {
    return markdownContent
  }

  return (
    <Card>
      {title && (
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
      )}
      <CardContent>
        {markdownContent}
      </CardContent>
    </Card>
  )
}
