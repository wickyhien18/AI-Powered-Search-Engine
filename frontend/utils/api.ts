import { API_URL } from '../config/env'

export type Source = {
  title: string
  url: string
  snippet: string
  score: number | null
}

export type AskResponse = {
  query: string
  search_query: string
  answer: string
  sources: Source[]
  conversation_id: number
}

export type AskPayload = {
  query: string
  top_k: number
  history: { role: 'user' | 'assistant'; content: string }[]
  conversation_id: number | null
}

export type ApiErrorKind = 'rate_limited' | 'unavailable' | 'upstream' | 'network' | 'unknown'

export class ApiError extends Error {
  constructor(
    public kind: ApiErrorKind,
    message: string,
    public retryAfter?: number,
    public limit?: string,
  ) {
    super(message)
  }
}

// Backend error shapes:
//   429 -> detail = { code, limit, retry_after, message }   (object)
//   503 -> detail = { code, message }                        (object)
//   502 -> detail = "text"                                   (string)
//   422 -> detail = [ ... ]                                  (list, FastAPI validation)
// Reading detail as a string unconditionally would show "[object Object]".
export async function askQuestion(payload: AskPayload): Promise<AskResponse> {
  let res: Response
  try {
    res = await fetch(`${API_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new ApiError('network', 'Cannot reach the server. Check your connection and try again.')
  }

  if (res.ok) return res.json()

  const body = await res.json().catch(() => null)
  const detail = body?.detail

  if (res.status === 429 && detail && typeof detail === 'object') {
    throw new ApiError(
      'rate_limited',
      detail.message ?? 'Too many requests. Please try again later.',
      typeof detail.retry_after === 'number' ? detail.retry_after : undefined,
      detail.limit,
    )
  }
  if (res.status === 503) {
    throw new ApiError('unavailable', 'The service is temporarily unavailable. Please try again later.')
  }
  if (res.status === 502) {
    throw new ApiError(
      'upstream',
      typeof detail === 'string' ? detail : 'The search or the model failed. Please try again.',
    )
  }
  throw new ApiError(
    'unknown',
    typeof detail === 'string' ? detail : `Request failed (${res.status}).`,
  )
}
