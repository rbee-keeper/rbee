// TEAM-391: SD Worker UI main component
// Pattern: Simple stub for now, TEAM-399+ will implement full UI

import { useTextToImage } from '@rbee/sd-worker-react'
import { useState } from 'react'

function App() {
  const [prompt, setPrompt] = useState('')
  const { generate, isLoading, progress, result, error } = useTextToImage({
    baseUrl: 'http://localhost:8600',
    workerId: 'sd-worker-1',
  })

  const handleGenerate = () => {
    if (prompt) {
      generate({ prompt, steps: 20 })
    }
  }

  return (
    <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
      <h1>🎨 SD Worker UI</h1>
      <p>Stable Diffusion Worker - Text to Image</p>

      <div style={{ marginTop: '2rem' }}>
        <label htmlFor="prompt" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Prompt:
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="a photo of a cat"
          style={{
            width: '100%',
            padding: '0.5rem',
            minHeight: '100px',
            fontFamily: 'inherit',
          }}
        />
      </div>

      <button
        onClick={handleGenerate}
        disabled={isLoading || !prompt}
        style={{
          marginTop: '1rem',
          padding: '0.75rem 1.5rem',
          fontSize: '1rem',
          cursor: isLoading || !prompt ? 'not-allowed' : 'pointer',
          opacity: isLoading || !prompt ? 0.5 : 1,
        }}
      >
        {isLoading ? 'Generating...' : 'Generate Image'}
      </button>

      {progress && (
        <div style={{ marginTop: '1rem' }}>
          <p>
            Progress: {progress.step} / {progress.total} ({progress.percentage.toFixed(0)}%)
          </p>
          <div
            style={{
              width: '100%',
              height: '20px',
              backgroundColor: '#e0e0e0',
              borderRadius: '10px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progress.percentage}%`,
                height: '100%',
                backgroundColor: '#4caf50',
                transition: 'width 0.3s ease',
              }}
            />
          </div>
        </div>
      )}

      {error && (
        <div
          style={{
            marginTop: '1rem',
            padding: '1rem',
            backgroundColor: '#ffebee',
            color: '#c62828',
            borderRadius: '4px',
          }}
        >
          Error: {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      )}

      {result && (
        <div style={{ marginTop: '2rem' }}>
          <h2>Generated Image:</h2>
          <img
            src={result.imageBase64}
            alt="Generated"
            style={{ maxWidth: '100%', borderRadius: '8px', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}
          />
          <p style={{ marginTop: '0.5rem', fontSize: '0.875rem', color: '#666' }}>Seed: {result.seed}</p>
        </div>
      )}

      <div style={{ marginTop: '3rem', padding: '1rem', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
        <h3>📝 Status: Stub Implementation</h3>
        <p>
          This is a basic stub created by TEAM-391. The SDK and React hooks are in place but not yet connected to the
          backend.
        </p>
        <p>
          <strong>TEAM-399+</strong> will implement:
        </p>
        <ul>
          <li>Full text-to-image UI with parameter controls</li>
          <li>Image-to-image with upload</li>
          <li>Inpainting with canvas mask editor</li>
          <li>Image gallery with local storage</li>
          <li>Real backend integration via WASM SDK</li>
        </ul>
      </div>
    </div>
  )
}

export default App
