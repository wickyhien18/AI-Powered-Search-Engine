'use client'

import { useState } from 'react'

const API_URL = 'http://localhost:8000'

export default function Page() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSearch(e) {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 5 }),
      })

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`)
      }

      const data = await response.json()
      setResults(data.results)
    } catch (err) {
      // Most common cause here: FastAPI (uvicorn) isn't running,
      // or CORS is blocking the request — check the browser console for the exact error.
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 700, margin: '40px auto', fontFamily: 'sans-serif' }}>
      <h1>AI Search Engine</h1>

      <form onSubmit={handleSearch} style={{ display: 'flex', gap: 8 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search BBC news articles..."
          style={{ flex: 1, padding: 8, fontSize: 16 }}
        />
        <button type="submit" disabled={loading} style={{ padding: '8px 16px' }}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && (
        <p style={{ color: 'red', marginTop: 16 }}>
          Error: {error}
        </p>
      )}

      <div style={{ marginTop: 24 }}>
        {results.map((r, i) => (
          <div
            key={i}
            style={{
              border: '1px solid #ddd',
              borderRadius: 6,
              padding: 12,
              marginBottom: 12,
            }}
          >
            <div style={{ fontSize: 12, color: '#666', marginBottom: 4 }}>
              category: {r.category} · article #{r.article_id} · score: {r.score.toFixed(3)}
            </div>
            <div>{r.text}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
