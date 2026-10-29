"use client";
import type { ReactNode } from "react";
import { ThemeProvider } from "@/components/theme/theme-provider";
import { GovHeader } from "@/components/navigation/gov-header";

export function AppChrome({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <div className="min-h-screen bg-[#f8f9fa] text-slate-900">
        <GovHeader />
        {children}
      </div>
    </ThemeProvider>
  );
}
