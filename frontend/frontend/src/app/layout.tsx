import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppChrome } from "@/components/app-chrome";

export const metadata: Metadata = {
  title: { default: "PolicyLens — Consultation Research", template: "%s · PolicyLens" },
  description: "A read-only research interface to saved Canadian federal consultation research artifacts.",
  applicationName: "PolicyLens",
  robots: { index: false, follow: false }
};
export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#143454", colorScheme: "light dark" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en" suppressHydrationWarning><body><AppChrome>{children}</AppChrome></body></html>;
}
