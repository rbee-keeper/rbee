// TEAM-476: HuggingFace models listing page - STUB for MVP
export default function HuggingFaceModelsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">HuggingFace Models</h1>
          <p className="text-muted-foreground">
            Browse language models from HuggingFace
          </p>
        </div>

        {/* Stub Content */}
        <div className="border rounded-lg p-8 text-center space-y-4">
          <div className="text-6xl">🤗</div>
          <h2 className="text-xl font-semibold">Coming Soon</h2>
          <p className="text-muted-foreground max-w-md mx-auto">
            HuggingFace model listing will be implemented here. This page will display models in a
            table view with metrics and details.
          </p>
          <div className="pt-4">
            <code className="text-sm bg-muted px-3 py-1 rounded">
              fetchHuggingFaceModels() → Table View
            </code>
          </div>
        </div>
      </div>
    </div>
  )
}
