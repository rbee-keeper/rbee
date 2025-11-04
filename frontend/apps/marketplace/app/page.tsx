// TEAM-405: Marketplace home page
import Link from "next/link";
import { Button } from "@rbee/ui/atoms/Button";
import { ArrowRight, Sparkles, Zap, Search, Database } from "lucide-react";

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16 max-w-7xl">
      {/* Hero Section */}
      <div className="max-w-4xl mx-auto text-center mb-20">
        <div className="mb-8 inline-flex items-center justify-center size-20 rounded-2xl bg-primary/10 ring-1 ring-primary/20">
          <Sparkles className="size-10 text-primary" />
        </div>
        
        <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">
          AI Model Marketplace
        </h1>
        
        <p className="text-xl md:text-2xl text-muted-foreground mb-10 max-w-2xl mx-auto leading-relaxed">
          Discover and explore state-of-the-art language models from HuggingFace.
          Pre-rendered for blazing-fast performance and maximum SEO.
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
        <div className="group p-8 rounded-xl border border-border bg-card hover:border-primary/50 transition-colors">
          <div className="mb-4 inline-flex items-center justify-center size-12 rounded-lg bg-primary/10 ring-1 ring-primary/20 group-hover:bg-primary/20 transition-colors">
            <Zap className="size-6 text-primary" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Lightning Fast</h3>
          <p className="text-muted-foreground leading-relaxed">
            Top 100 models pre-rendered at build time for instant loading and zero latency
          </p>
        </div>
        
        <div className="group p-8 rounded-xl border border-border bg-card hover:border-primary/50 transition-colors">
          <div className="mb-4 inline-flex items-center justify-center size-12 rounded-lg bg-primary/10 ring-1 ring-primary/20 group-hover:bg-primary/20 transition-colors">
            <Search className="size-6 text-primary" />
          </div>
          <h3 className="text-xl font-semibold mb-3">SEO Optimized</h3>
          <p className="text-muted-foreground leading-relaxed">
            Semantic HTML, structured data, and perfect metadata for maximum search visibility
          </p>
        </div>
        
        <div className="group p-8 rounded-xl border border-border bg-card hover:border-primary/50 transition-colors">
          <div className="mb-4 inline-flex items-center justify-center size-12 rounded-lg bg-primary/10 ring-1 ring-primary/20 group-hover:bg-primary/20 transition-colors">
            <Database className="size-6 text-primary" />
          </div>
          <h3 className="text-xl font-semibold mb-3">Rich Metadata</h3>
          <p className="text-muted-foreground leading-relaxed">
            Complete model information including downloads, likes, tags, and author details
          </p>
        </div>
      </div>
    </div>
  );
}
