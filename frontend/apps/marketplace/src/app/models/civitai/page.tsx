// TEAM-476: CivitAI models listing page - STUB for MVP
export default function CivitAIModelsPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold">CivitAI Models</h1>
          <p className="text-muted-foreground">
            Browse image generation models from CivitAI
          </p>
        </div>

        {/* Stub Content */}
        <div className="border rounded-lg p-8 text-center space-y-4">
          <div className="text-6xl">🎨</div>
          <h2 className="text-xl font-semibold">Coming Soon</h2>
          <p className="text-muted-foreground max-w-md mx-auto">
            CivitAI model listing will be implemented here. This page will display models in a
            card grid view with preview images.
          </p>
          <div className="pt-4">
            <code className="text-sm bg-muted px-3 py-1 rounded">
              fetchCivitAIModels() → Card Grid View
            </code>
          </div>
        </div>
      </div>
    </div>
  )
}
