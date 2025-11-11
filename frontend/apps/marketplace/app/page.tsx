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
          Browse 1000s of LLMs from HuggingFace and image models from Civitai. Find compatible models, check hardware
          requirements, and download instantly.
        </p>

        <div className="flex gap-4 justify-center">
          <Button size="lg" className="h-12 px-8 text-base" asChild>
            <Link href="/models/huggingface">
              Browse LLMs
              <ArrowRight className="size-5 ml-2" />
            </Link>
          </Button>
          <Button size="lg" variant="outline" className="h-12 px-8 text-base" asChild>
            <Link href="/models/civitai">
              Browse Image Models
              <ArrowRight className="size-5 ml-2" />
            </Link>
          </Button>
        </div>
      </div>

      {/* Features */}
      <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
        <FeatureInfoCard
          icon={<Zap />}
          title="HuggingFace LLMs"
          body="Browse thousands of text generation models. Filter by size, license, and popularity. Perfect for local inference with rbee."
          tone="default"
          showBorder
        />

        <FeatureInfoCard
          icon={<Search />}
          title="Civitai Image Models"
          body="Discover Stable Diffusion checkpoints and LORAs. View previews, check compatibility, and download models for image generation."
          tone="default"
          showBorder
        />

        <FeatureInfoCard
          icon={<Database />}
          title="Worker Binaries"
          body="Pre-built inference workers for CPU, CUDA, ROCm, and Metal. One-click download and run compatible models locally."
          tone="default"
          showBorder
        />
      </div>
    </div>
  )
}
