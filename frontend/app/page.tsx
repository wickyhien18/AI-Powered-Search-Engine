'use client'

import { useState, useRef, useEffect, FormEvent } from 'react'
import { API_URL } from '../config/env'

interface SourceChunk {
  score: number
  text: string
  article_id: number
  category: string
}

interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
  sources?: SourceChunk[]
}

interface AskResponse {
  query: string
  answer: string
  sources: SourceChunk[]
}

// Pulled straight from the golden set in evaluate.py — these are the exact
// queries already verified against known-correct article_ids, so clicking
// one here demonstrates a case that's actually been measured, not a random example.
const QUICK_PROMPTS = [
  'kapranos',
  'why are musicians protesting',
  'movie release delayed',
]

export default function Page() {
  const [query, setQuery] = useState<string>('')
  const [turns, setTurns] = useState<ChatTurn[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to the newest turn whenever the conversation grows —
  // without this, a long conversation would stay scrolled at the top.
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns, loading])

  async function runQuery(text: string) {
    if (!text.trim() || loading) return

    const userTurn: ChatTurn = { role: 'user', content: text }
    const updatedTurns = [...turns, userTurn]
    setTurns(updatedTurns)
    setQuery('')
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          top_k: 5,
          history: turns.map((t) => ({ role: t.role, content: t.content })),
        }),
      })

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`)
      }

      const data: AskResponse = await response.json()
      setTurns([...updatedTurns, { role: 'assistant', content: data.answer, sources: data.sources }])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault()
    runQuery(query)
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="masthead-block">
          <h1 className="masthead">Wire Search</h1>
          <hr className="masthead-rule" />
        </div>

        <button className="new-search-btn" onClick={() => setTurns([])}>
          + New search
        </button>

        <div className="sidebar-quick-prompts">
          <div className="sidebar-label">Verified queries</div>
          {QUICK_PROMPTS.map((p) => (
            <button key={p} className="quick-prompt" onClick={() => runQuery(p)}>
              {p}
            </button>
          ))}
        </div>
      </aside>

      <div className="main">
        <div className="chat-scroll">
          {turns.length === 0 ? (
            <div className="empty-state">
              Search the BBC archive — 2,225 articles indexed.
              <br />
              Ask a question, or try a verified query from the sidebar.
            </div>
          ) : (
            <div className="chat-inner">
              {turns.map((turn, i) =>
                turn.role === 'user' ? (
                  <div key={i} className="turn-user">
                    <div className="label">query</div>
                    <div className="content">{turn.content}</div>
                  </div>
                ) : (
                  <div key={i} className="turn-assistant">
                    <div className="label">wire search</div>
                    <div className="content">{turn.content}</div>
                    {turn.sources && turn.sources.length > 0 && (
                      <details className="sources-toggle">
                        <summary>{turn.sources.length} sources</summary>
                        {turn.sources.map((s, j) => (
                          <div key={j} className="source-row">
                            <div className="source-meta">
                              article #{s.article_id} · {s.category} · score{' '}
                              <span className="score">{s.score.toFixed(3)}</span>
                            </div>
                            <div>{s.text}</div>
                          </div>
                        ))}
                      </details>
                    )}
                  </div>
                )
              )}
              {loading && <div className="loading-line">retrieving and generating — may take a while on CPU-only hardware...</div>}
              {error && <div className="error-line">Error: {error}</div>}
              <div ref={scrollRef} />
            </div>
          )}
        </div>

        <div className="input-bar">
          <form className="input-inner" onSubmit={handleSubmit}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask about the archive..."
            />
            <button type="submit" disabled={loading}>
              {loading ? 'Asking' : 'Ask'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
