// TEAM-463: Premium trained words section for CivitAI models
// Beautiful tag display with copy functionality

'use client'

import { Badge, Card } from '@rbee/ui/atoms'
import { Sparkles, Copy, Check } from 'lucide-react'
import { useState } from 'react'
import { cn } from '@rbee/ui/utils'

export interface CivitAITrainedWordsProps {
  trainedWords?: string[]
  className?: string
}

export function CivitAITrainedWords({ trainedWords, className }: CivitAITrainedWordsProps) {
  const [copiedWord, setCopiedWord] = useState<string | null>(null)

  if (!trainedWords || trainedWords.length === 0) {
    return null
  }

  const copyWord = (word: string) => {
    navigator.clipboard.writeText(word)
    setCopiedWord(word)
    setTimeout(() => setCopiedWord(null), 2000)
  }

  const copyAll = () => {
    navigator.clipboard.writeText(trainedWords.join(', '))
    setCopiedWord('all')
    setTimeout(() => setCopiedWord(null), 2000)
  }

  return (
    <Card className={cn('p-6 space-y-4', className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="size-5 text-primary" />
          <h3 className="font-semibold text-lg">Trained Words</h3>
        </div>
        <button
          type="button"
          onClick={copyAll}
          className="text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
        >
          {copiedWord === 'all' ? (
            <>
              <Check className="size-3" />
              Copied!
            </>
          ) : (
            <>
              <Copy className="size-3" />
              Copy All
            </>
          )}
        </button>
      </div>

      <p className="text-sm text-muted-foreground">
        Use these trigger words in your prompts to activate the model:
      </p>

      <div className="flex flex-wrap gap-2">
        {trainedWords.map((word) => (
          <button
            key={word}
            type="button"
            onClick={() => copyWord(word)}
            className="group relative"
          >
            <Badge
              variant="secondary"
              className="font-mono cursor-pointer transition-all hover:bg-primary hover:text-primary-foreground pr-8"
            >
              {word}
              <span className="absolute right-2 top-1/2 -translate-y-1/2">
                {copiedWord === word ? (
                  <Check className="size-3" />
                ) : (
                  <Copy className="size-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                )}
              </span>
            </Badge>
          </button>
        ))}
      </div>
    </Card>
  )
}
