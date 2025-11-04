// TEAM-405: Marketplace home page
import Link from "next/link";
import { ArrowRight, Sparkles } from "lucide-react";

export default function Home() {
  return (
    <div className="font-sans min-h-screen">
      <main className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="max-w-4xl mx-auto text-center mb-16">
          <div className="mb-6">
            <Sparkles className="size-16 mx-auto text-primary" />
          </div>
          
          <h1 className="text-5xl font-bold mb-6">
            AI Model Marketplace
          </h1>
          
          <p className="text-xl text-muted-foreground mb-8">
            Discover and explore state-of-the-art language models from HuggingFace.
            Pre-rendered for blazing-fast performance and maximum SEO.
          </p>
          
          <div className="flex gap-4 justify-center">
            <Link
              href="/models"
              className="inline-flex items-center gap-2 px-8 py-4 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors font-semibold text-lg"
            >
              Browse Models
              <ArrowRight className="size-5" />
            </Link>
          </div>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
          <div className="p-6 border border-border rounded-lg">
            <h3 className="text-xl font-semibold mb-2">⚡ Lightning Fast</h3>
            <p className="text-muted-foreground">
              Top 100 models pre-rendered at build time for instant loading
            </p>
          </div>
          
          <div className="p-6 border border-border rounded-lg">
            <h3 className="text-xl font-semibold mb-2">🔍 SEO Optimized</h3>
            <p className="text-muted-foreground">
              Semantic HTML, structured data, and perfect metadata for search engines
            </p>
          </div>
          
          <div className="p-6 border border-border rounded-lg">
            <h3 className="text-xl font-semibold mb-2">📊 Rich Metadata</h3>
            <p className="text-muted-foreground">
              Complete model information including config, files, and examples
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
