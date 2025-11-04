import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "rbee Model Marketplace - AI Language Models",
    template: "%s | rbee Marketplace"
  },
  description: "Browse and discover AI language models for your projects. Pre-rendered static pages for optimal SEO and performance.",
  keywords: ["AI", "language models", "LLM", "machine learning", "marketplace"],
  authors: [{ name: "rbee" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://marketplace.rbee.ai",
    siteName: "rbee Model Marketplace",
    title: "rbee Model Marketplace - AI Language Models",
    description: "Browse and discover AI language models for your projects",
  },
  twitter: {
    card: "summary_large_image",
    title: "rbee Model Marketplace",
    description: "Browse and discover AI language models",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
