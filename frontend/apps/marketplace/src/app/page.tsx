// TEAM-476: Marketplace homepage - simple and static
import Link from 'next/link'

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16">
      <div className="max-w-4xl mx-auto text-center space-y-12">
        {/* Header */}
        <div className="space-y-4">
          <h1 className="text-4xl font-bold">rbee Marketplace</h1>
          <p className="text-xl text-muted-foreground">
            Browse AI models and workers for your rbee cluster
          </p>
        </div>

        {/* Model Categories */}
        <div className="grid md:grid-cols-2 gap-8 mt-12">
          {/* CivitAI - Image Models */}
          <Link
            href="/models/civitai"
            className="group p-8 border rounded-lg hover:border-primary transition-colors"
          >
            <div className="space-y-4">
              <h2 className="text-2xl font-semibold group-hover:text-primary transition-colors">
                Image Models
              </h2>
              <p className="text-muted-foreground">
                Browse image generation models from CivitAI
              </p>
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                Continue to CivitAI
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>
          </Link>

          {/* HuggingFace - LLM Models */}
          <Link
            href="/models/huggingface"
            className="group p-8 border rounded-lg hover:border-primary transition-colors"
          >
            <div className="space-y-4">
              <h2 className="text-2xl font-semibold group-hover:text-primary transition-colors">
                LLM Models
              </h2>
              <p className="text-muted-foreground">
                Browse language models from HuggingFace
              </p>
              <div className="flex items-center gap-2 text-sm font-medium text-primary">
                Continue to HuggingFace
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}
