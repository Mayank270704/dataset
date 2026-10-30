"use client";
import type { ReactNode } from "react";
import { ThemeProvider } from "@/components/theme/theme-provider";
import { AuthProvider } from "@/lib/auth-context";
import { GovHeader } from "@/components/navigation/gov-header";

export function AppChrome({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider>
      <AuthProvider>
        <div className="min-h-screen bg-[#f8f9fa] text-slate-900">
          <GovHeader />
          {children}
        </div>
      </AuthProvider>
    </ThemeProvider>
  );
}
