import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Providers } from "@/components/Providers";
import { THEME_INIT_SCRIPT } from "@/components/ThemeProvider";

import "./globals.css";

export const metadata: Metadata = {
  title: "ResearchPilot",
  description: "A multi-agent research copilot that writes cited, fact-checked literature reviews from real papers.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Applies the saved theme before the first paint (no light/dark flash). */}
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
