"use client";
import Link from "next/link";
import { useCallback } from "react";
import { Clock3, Trash2 } from "lucide-react";
import { officialPolicies } from "@/features/feed/demo-data";
import { clearHistory, readHistory, usePolicyLibrary } from "@/lib/policy-library";

export default function HistoryPage() {
  const reader = useCallback(() => readHistory(), []);
  const history = usePolicyLibrary(reader, []);
  const rows = history.map(item => ({ item, policy: officialPolicies.find(p => p.slug === item.slug) }));
  return <main className="mx-auto max-w-4xl px-4 py-8 pb-24">
    <div className="flex items-start justify-between gap-4"><div><h1 className="font-display text-3xl font-extrabold">Your history</h1><p className="mt-2 text-sm text-slate-600">Recently viewed policy pages on this browser. History is stored locally and is not tied to an account.</p></div><button onClick={clearHistory} disabled={!history.length} className="inline-flex min-h-10 items-center gap-2 rounded-lg border px-3 text-sm disabled:opacity-40"><Trash2 size={15}/>Clear history</button></div>
    {rows.length ? <ul className="mt-6 space-y-3">{rows.map(({ item, policy }) => <li key={`${item.kind}-${item.slug}`} className="rounded-xl border border-slate-200 bg-white p-4">{policy ? <Link className="font-semibold text-[#143454] hover:underline" href={`/policy/${policy.slug}`}>{policy.title}</Link> : <p className="font-semibold text-slate-800">Consultation inference completed</p>}<p className="mt-1 text-xs text-slate-500"><Clock3 className="mr-1 inline" size={13}/>{new Date(item.visitedAt).toLocaleString()} · {item.kind === "policy" ? "Policy page" : "Inference run (comment text is not stored)"}</p></li>)}</ul> : <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center"><p className="font-semibold">No history yet</p><p className="mt-1 text-sm text-slate-600">Viewed policy pages and successful inference runs are recorded in this browser. Comment text is never stored here.</p><Link href="/discover" className="mt-4 inline-block text-sm font-semibold text-blue-800 hover:underline">Explore policies</Link></div>}
  </main>;
}
