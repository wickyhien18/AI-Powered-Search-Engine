"use client";

import { useState, FormEvent } from "react";
import { API_URL } from "../config/env";

interface SourceChunk {
  score: number;
  text: string;
  article_id: number;
  category: string;
}

// One turn in the visible conversation. "sources" only ever exists on
// assistant turns — a user turn never has retrieval results attached.
interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  sources?: SourceChunk[];
}

interface AskResponse {
  query: string;
  answer: string;
  sources: SourceChunk[];
}

export default function Page() {
  const [query, setQuery] = useState<string>("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!query.trim()) return;

    const userTurn: ChatTurn = { role: "user", content: query };

    // Show the user's own message immediately, before the network call resolves —
    // this is why we build `updatedTurns` here instead of waiting for the response.
    const updatedTurns = [...turns, userTurn];
    setTurns(updatedTurns);
    setQuery("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userTurn.content,
          top_k: 5,
          // Send everything EXCEPT the turn we just added locally — the backend's
          // `history` field means "everything before this question", and the
          // current question is sent separately as `query`.
          history: turns.map((t) => ({ role: t.role, content: t.content })),
        }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data: AskResponse = await response.json();

      const assistantTurn: ChatTurn = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };
      setTurns([...updatedTurns, assistantTurn]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        maxWidth: 700,
        margin: "40px auto",
        fontFamily: "sans-serif",
        background: "#1a1a1a",
        color: "#e0e0e0",
        minHeight: "100vh",
        padding: "0 16px",
      }}
    >
      <h1>AI Search Engine</h1>

      <div
        style={{
          marginTop: 24,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        {turns.map((turn, i) => (
          <div key={i}>
            {turn.role === "user" ? (
              <div style={{ fontWeight: 600 }}>You: {turn.content}</div>
            ) : (
              <>
                <div
                  style={{
                    padding: 16,
                    background: "#2a2a2a",
                    borderRadius: 8,
                    lineHeight: 1.5,
                    border: "1px solid #3a3a3a",
                  }}
                >
                  {turn.content}
                </div>
                {turn.sources && turn.sources.length > 0 && (
                  <details style={{ marginTop: 8, fontSize: 13 }}>
                    <summary style={{ cursor: "pointer", color: "#999" }}>
                      Sources ({turn.sources.length})
                    </summary>
                    {turn.sources.map((s, j) => (
                      <div
                        key={j}
                        style={{
                          border: "1px solid #3a3a3a",
                          borderRadius: 6,
                          padding: 10,
                          marginTop: 8,
                          background: "#222",
                        }}
                      >
                        <div style={{ color: "#999", marginBottom: 4 }}>
                          category: {s.category} · article #{s.article_id} ·
                          score: {s.score.toFixed(3)}
                        </div>
                        <div>{s.text}</div>
                      </div>
                    ))}
                  </details>
                )}
              </>
            )}
          </div>
        ))}
      </div>

      {loading && (
        <p style={{ color: "#999", marginTop: 16, fontSize: 14 }}>
          Generating answer locally — this can take a while on CPU-only
          hardware.
        </p>
      )}

      {error && (
        <p style={{ color: "#ff6b6b", marginTop: 16 }}>Error: {error}</p>
      )}

      <form
        onSubmit={handleAsk}
        style={{ display: "flex", gap: 8, marginTop: 24 }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question — follow-ups are understood..."
          style={{
            flex: 1,
            padding: 8,
            fontSize: 16,
            background: "#2a2a2a",
            color: "#e0e0e0",
            border: "1px solid #3a3a3a",
            borderRadius: 4,
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "8px 16px",
            background: "#3a3a3a",
            color: "#e0e0e0",
            border: "1px solid #4a4a4a",
            borderRadius: 4,
            cursor: "pointer",
          }}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>
    </div>
  );
}
