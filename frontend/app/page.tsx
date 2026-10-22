"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import { API_URL } from "../config/env";
import { ApiError, askQuestion, type Source } from "../utils/api";
import { hostname, isHttpUrl, linkCitations } from "../utils/citations";

interface ChatTurn {
  role: "user" | "assistant";
  content: string;
  // null for user messages and for old rows stored without sources
  sources?: Source[] | null;
}

interface ConversationSummary {
  id: number;
  title: string;
  created_at: string;
}

// Each example costs one real web search, exactly like a typed question.
const QUICK_PROMPTS = [
  "why are musicians protesting",
  "when was the song Hind's Hall by Macklemore released",
  "why did Daniel Ek step down as Spotify CEO",
];

function SourceList({ sources }: { sources: Source[] }) {
  return (
    <details className="sources-toggle">
      <summary>{sources.length} sources</summary>
      <ol className="source-list">
        {sources.map((s, i) => (
          <li key={i} className="source-item">
            <span className="source-number">{i + 1}</span>
            {isHttpUrl(s.url) ? (
              <a href={s.url} target="_blank" rel="noopener noreferrer">
                {s.title || s.url}
              </a>
            ) : (
              <span>{s.title || "Untitled source"}</span>
            )}
            <span className="source-domain">{hostname(s.url)}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

export default function Page() {
  const [query, setQuery] = useState<string>("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [cooldown, setCooldown] = useState<number>(0);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const wasCoolingDown = useRef<boolean>(false);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, loading]);

  // Per-minute rate limit: count down once per second, then clear the banner.
  useEffect(() => {
    if (cooldown > 0) {
      wasCoolingDown.current = true;
      const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
      return () => clearTimeout(t);
    }
    if (wasCoolingDown.current) {
      wasCoolingDown.current = false;
      setError(null);
    }
  }, [cooldown]);

  // Load the sidebar conversation list once, when the page first mounts.
  useEffect(() => {
    fetch(`${API_URL}/conversations`)
      .then((res) => res.json())
      .then((data) => setConversations(data.conversations))
      .catch(() => {
        // Silently ignore — sidebar list is a nice-to-have, not core functionality
      });
  }, []);

  function startNewSearch() {
    setTurns([]);
    setConversationId(null);
    setError(null);
  }

  // Plain GET that reads stored rows from SQLite: no search and no LLM call,
  // so reopening a past conversation costs nothing and is not rate limited.
  async function loadConversation(id: number) {
    setError(null);
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/conversations/${id}`);
      if (!response.ok)
        throw new Error(`Server responded with status ${response.status}`);
      const data: { messages: ChatTurn[] } = await response.json();
      setTurns(data.messages);
      setConversationId(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  async function runQuery(text: string) {
    // Example buttons call this directly, so the cooldown must be checked here
    // and not only on the submit button.
    if (!text.trim() || loading || cooldown > 0) return;

    const userTurn: ChatTurn = { role: "user", content: text };
    setTurns([...turns, userTurn]);
    setQuery("");
    setLoading(true);
    setError(null);

    try {
      const data = await askQuestion({
        query: text,
        top_k: 5,
        history: turns.map((t) => ({ role: t.role, content: t.content })),
        conversation_id: conversationId, // null on a brand-new search — server creates one
      });
      setTurns([
        ...turns,
        userTurn,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);

      // First message in a new conversation — now we know its id. Also
      // refresh the sidebar list so the new conversation shows up right away.
      if (conversationId === null) {
        setConversationId(data.conversation_id);
        fetch(`${API_URL}/conversations`)
          .then((res) => res.json())
          .then((d) => setConversations(d.conversations))
          .catch(() => {});
      }
    } catch (err) {
      // The question was not answered, so it must not stay in the chat: the
      // server stored nothing, and the next request would otherwise send two
      // user messages in a row as history. Put the text back so it can be resent.
      setTurns(turns);
      setQuery(text);
      if (err instanceof ApiError) {
        setError(err.message);
        if (
          err.kind === "rate_limited" &&
          err.limit === "ip_minute" &&
          err.retryAfter
        ) {
          setCooldown(err.retryAfter);
        }
      } else {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    runQuery(query);
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="masthead-block">
          <h1 className="masthead">Wire Search</h1>
          <hr className="masthead-rule" />
        </div>

        <button className="new-search-btn" onClick={startNewSearch}>
          + New search
        </button>

        <div className="sidebar-quick-prompts">
          <div className="sidebar-label">Try asking</div>
          {QUICK_PROMPTS.map((p) => (
            <button
              key={p}
              className="quick-prompt"
              onClick={() => runQuery(p)}
            >
              {p}
            </button>
          ))}
        </div>

        {conversations.length > 0 && (
          <div className="sidebar-quick-prompts">
            <div className="sidebar-label">Past conversations</div>
            {conversations.map((c) => (
              <button
                key={c.id}
                className="quick-prompt"
                onClick={() => loadConversation(c.id)}
                style={
                  c.id === conversationId
                    ? { color: "#d9a441", borderLeftColor: "#d9a441" }
                    : undefined
                }
              >
                {c.title}
              </button>
            ))}
          </div>
        )}
      </aside>

      <div className="main">
        <div className="chat-scroll">
          {turns.length === 0 ? (
            <div className="empty-state">
              Ask anything. Answers are grounded in live web sources, with [n]
              citations you can click.
              <br />
              Try an example from the sidebar, or reopen a past conversation.
            </div>
          ) : (
            <div className="chat-inner">
              {turns.map((turn, i) =>
                turn.role === "user" ? (
                  <div key={i} className="turn-user">
                    <div className="label">query</div>
                    <div className="content">{turn.content}</div>
                  </div>
                ) : (
                  <div key={i} className="turn-assistant">
                    <div className="label">wire search</div>
                    <div className="content">
                      {/* The answer is model output built from web pages, so images are
                          never rendered (a remote image URL can leak data) and links open
                          in a new tab. react-markdown does not render raw HTML by default. */}
                      <ReactMarkdown
                        disallowedElements={["img"]}
                        components={{
                          a: ({ href, children }) => {
                            const label = [children].flat().join("");
                            const isCitation = /^\[\d{1,2}\]$/.test(label);
                            return (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={
                                  isCitation ? "citation" : "answer-link"
                                }
                              >
                                {children}
                              </a>
                            );
                          },
                        }}
                      >
                        {linkCitations(turn.content, turn.sources ?? [])}
                      </ReactMarkdown>
                    </div>
                    {turn.sources && turn.sources.length > 0 && (
                      <SourceList sources={turn.sources} />
                    )}
                  </div>
                ),
              )}
              {loading && (
                <div className="loading-line">
                  searching the web and writing an answer...
                </div>
              )}
              <div ref={scrollRef} />
            </div>
          )}
        </div>

        {/* Outside the chat area on purpose: when the first question fails the
            chat is empty again, and an error inside it would never be shown. */}
        {error && (
          <div className="error-banner" role="alert">
            {error}
          </div>
        )}

        <div className="input-bar">
          <form className="input-inner" onSubmit={handleSubmit}>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything..."
            />
            <button type="submit" disabled={loading || cooldown > 0}>
              {cooldown > 0
                ? `Try again in ${cooldown}s`
                : loading
                  ? "Asking"
                  : "Ask"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
