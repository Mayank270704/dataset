"use client";
import Link from "next/link";
import { useTheme } from "@/components/theme/theme-provider";
import { readHistory, readSaved } from "@/lib/policy-library";
import { useCallback } from "react";
import { usePolicyLibrary } from "@/lib/policy-library";

export default function ProfilePage() {
  const { theme, toggleTheme } = useTheme();
  const countsReader = useCallback(() => [readSaved().length, readHistory().length] as const, []);
  const [savedCount, historyCount] = usePolicyLibrary(countsReader, [0, 0]);
  return <main className="mx-auto max-w-3xl px-4 py-8 pb-24"><h1 className="font-display text-3xl font-extrabold">Preferences</h1><p className="mt-2 text-sm text-slate-600">This portal does not have user accounts. These settings apply only to this browser; no profile identity is assumed.</p>
    <section className="mt-6 rounded-xl border border-slate-200 bg-white p-5"><h2 className="text-lg font-semibold">Appearance</h2><p className="mt-1 text-sm text-slate-600">Choose the color theme used on this device.</p><button onClick={toggleTheme} className="mt-4 rounded-lg border px-4 py-2 text-sm font-semibold">Switch to {theme === "dark" ? "light" : "dark"} theme</button></section>
    <section className="mt-4 rounded-xl border border-slate-200 bg-white p-5"><h2 className="text-lg font-semibold">On this browser</h2><div className="mt-3 flex flex-wrap gap-4 text-sm"><Link className="text-blue-800 hover:underline" href="/saved">{savedCount} saved {savedCount === 1 ? "policy" : "policies"}</Link><Link className="text-blue-800 hover:underline" href="/history">{historyCount} history {historyCount === 1 ? "item" : "items"}</Link></div><p className="mt-3 text-xs text-slate-500">Saved policies and browsing history remain in local browser storage and may be removed by clearing site data.</p></section>
    <p className="mt-5 text-xs text-slate-500">Account sign-in and registration are not configured. <Link href="/login" className="underline">Account access details</Link>.</p>
  </main>;
}
