import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { Lora, IBM_Plex_Mono } from 'next/font/google'
import './globals.css'

// Two deliberate typefaces: a serif for the masthead/identity, a monospace
// used ONLY for metadata readouts (category, score, article id) — everything
// else stays on the system sans stack. Exposed as CSS variables so globals.css
// can reference them without every component needing to import font objects.
const lora = Lora({ subsets: ['latin'], variable: '--font-lora' })
const plexMono = IBM_Plex_Mono({ subsets: ['latin'], weight: ['400', '500'], variable: '--font-mono' })

export const metadata: Metadata = {
  title: 'Wire Search — BBC Archive',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${lora.variable} ${plexMono.variable}`}>
      <body>{children}</body>
    </html>
  )
}
