import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "J.A.R.V.I.S.",
  description: "Locally hosted, privacy-first agentic AI assistant with an audio-reactive holographic interface.",
};

export const viewport: Viewport = {
  themeColor: "#04060a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
