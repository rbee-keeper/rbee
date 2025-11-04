// TEAM-405: Marketplace Image Models page
// Browse and search CivitAI models (placeholder for future implementation)

export function MarketplaceImageModels() {
  return (
    <div className="h-full flex flex-col items-center justify-center p-6">
      <div className="max-w-md text-center space-y-4">
        <h1 className="text-2xl font-bold">Image Models</h1>
        <p className="text-muted-foreground">
          CivitAI integration coming soon. This will allow you to browse and download image generation models.
        </p>
        <div className="text-sm text-muted-foreground">
          <p>Planned features:</p>
          <ul className="list-disc list-inside mt-2 space-y-1">
            <li>Browse Stable Diffusion models</li>
            <li>Search by style and category</li>
            <li>Download and manage models</li>
            <li>Preview generated images</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
