import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const instrumentSerif = localFont({
  src: "./fonts/instrument-serif-latin.woff2",
  variable: "--font-display",
  weight: "400",
  display: "swap",
});

const dmSans = localFont({
  src: "./fonts/dm-sans-latin.woff2",
  variable: "--font-body",
  weight: "100 1000",
  display: "swap",
});

const ibmPlexMono = localFont({
  src: [
    {
      path: "./fonts/ibm-plex-mono-400-latin.woff2",
      weight: "400",
      style: "normal",
    },
    {
      path: "./fonts/ibm-plex-mono-500-latin.woff2",
      weight: "500",
      style: "normal",
    },
  ],
  variable: "--font-mono",
  display: "swap",
});

const siteUrl = "https://memory.agentclash.dev";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "Agentic Memory Playground",
  description:
    "A real playground for semantic and episodic memory built on the Agentic Memory Python backend.",
  openGraph: {
    type: "website",
    url: siteUrl,
    siteName: "Agentic Memory Playground",
    title: "Agentic Memory Playground",
    description:
      "Store semantic memories, ingest episodic media, query mixed retrieval, and inspect temporal memory behavior.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Agentic Memory Playground",
    description:
      "A real playground for semantic and episodic memory built on the Agentic Memory Python backend.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${instrumentSerif.variable} ${dmSans.variable} ${ibmPlexMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
