import type { Source } from './api'

// Only http(s) URLs are ever turned into links. Old conversations stored in
// the database may hold sources without a url, and a URL must never be a
// javascript: or data: link.
export function isHttpUrl(url: unknown): url is string {
  return typeof url === 'string' && /^https?:\/\//.test(url)
}

export function hostname(url: unknown): string {
  if (!isHttpUrl(url)) return ''
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return ''
  }
}

// Turns "[2]" in the answer into a markdown link to sources[1].
// [n] is 1-based and matches the backend numbering (sources are renumbered
// after filtering, so [1] is always sources[0]). A number with no matching
// source (invented by the model) stays as plain text.
export function linkCitations(answer: string, sources: Source[]): string {
  return answer.replace(/\[(\d{1,2})\]/g, (match, n: string) => {
    const source = sources[Number(n) - 1]
    if (!source || !isHttpUrl(source.url)) return match
    const safeUrl = source.url
      .replace(/ /g, '%20')
      .replace(/\(/g, '%28')
      .replace(/\)/g, '%29')
    return `[\\[${n}\\]](${safeUrl})`
  })
}
