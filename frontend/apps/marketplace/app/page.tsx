// TEAM-405: Marketplace home page - using reusable rbee-ui components

import { Button } from '@rbee/ui/atoms/Button'
import { FeatureInfoCard } from '@rbee/ui/molecules/FeatureInfoCard'
import { IconPlate } from '@rbee/ui/molecules/IconPlate'
import { ArrowRight, Database, Search, Sparkles, Zap } from 'lucide-react'
import Link from 'next/link'

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16 max-w-7xl">
      {/* Hero Section */}
      <div className="max-w-4xl mx-auto text-center mb-20">
        <div className="mb-8 inline-flex justify-center">
          <IconPlate icon={<Sparkles />} size="xl" tone="primary" shape="rounded" />
        </div>

        <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">AI Model Marketplace</h1>

        <p className="text-xl md:text-2xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
          Discover and explore state-of-the-art language models from HuggingFace. Pre-rendered for blazing-fast
          performance and maximum SEO.
        </p>

        <div className="flex gap-4 justify-center">
          <Button size="lg" className="h-12 px-8 text-base" asChild>
            <Link href="/models">
              Browse Models
              <ArrowRight className="size-5 ml-2" />
            </Link>
          </Button>
        </div>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
        <FeatureInfoCard
          icon={Zap}
          title="Lightning Fast"
          body="Top 100 models pre-rendered at build time for instant loading and zero latency"
          tone="default"
          showBorder
        />

        <FeatureInfoCard
          icon={Search}
          title="SEO Optimized"
          body="Semantic HTML, structured data, and perfect metadata for maximum search visibility"
          tone="default"
          showBorder
        />

        <FeatureInfoCard
          icon={Database}
          title="Rich Metadata"
          body="Complete model information including downloads, likes, tags, and author details"
          tone="default"
          showBorder
        />
      </div>
    </div>
  )
}
