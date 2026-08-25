"use client";

import { useState, FormEvent } from "react";
import { API_URL } from "../config/env";

// This shape must match exactly what main.py's /ask endpoint returns —
// if main.py's response shape changes and this type isn't updated,
// TypeScript will flag the mismatch at compile time instead of failing silently in the browser.
interface SourceChunk {
  score: number;
  text: string;
  article_id: number;
  category: string;
}

interface AskResponse {
  query: string;
  answer: string;
  sources: SourceChunk[];
}

export default function Page() {
  const [query, setQuery] = useState<string>("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<SourceChunk[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data: AskResponse = await response.json();
      setAnswer(data.answer);
      setSources(data.sources);
    } catch (err) {
      // Most common cause here: FastAPI (uvicorn) isn't running,
      // or CORS is blocking the request — check the browser console for the exact error.
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif" }}
    >
      <h1>AI Search Engine</h1>

      <form onSubmit={handleAsk} style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about BBC news articles..."
          style={{ flex: 1, padding: 8, fontSize: 16 }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{ padding: "8px 16px" }}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>

      {/* The LLM call is slow on CPU-only hardware — this note sets expectations
          instead of letting the UI look frozen during a 30+ second wait. */}
      {loading && (
        <p style={{ color: "#888", marginTop: 16, fontSize: 14 }}>
          Generating answer locally — this can take up to a minute on CPU-only
          hardware.
        </p>
      )}

      {error && <p style={{ color: "red", marginTop: 16 }}>Error: {error}</p>}

      {answer && (
        <div
          style={{
            marginTop: 24,
            padding: 16,
            background: "#f5f5f5",
            borderRadius: 8,
            lineHeight: 1.5,
          }}
        >
          {answer}
        </div>
      )}

      {sources.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <h3 style={{ fontSize: 14, color: "#666" }}>Sources</h3>
          {sources.map((s, i) => (
            <div
              key={i}
              style={{
                border: "1px solid #ddd",
                borderRadius: 6,
                padding: 12,
                marginBottom: 12,
              }}
            >
              <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>
                category: {s.category} · article #{s.article_id} · score:{" "}
                {s.score.toFixed(3)}
              </div>
              <div style={{ fontSize: 14 }}>{s.text}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
