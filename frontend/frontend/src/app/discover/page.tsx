"use client";
import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { officialPolicies } from "@/features/feed/demo-data";
import { PolicyCard } from "@/features/feed/policy-card";

export default function DiscoverPage() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All categories");
  const categories = ["All categories", ...Array.from(new Set(officialPolicies.map(p => p.category)))];
  const results = useMemo(() => officialPolicies.filter(p => {
    const q = query.trim().toLocaleLowerCase();
    return (category === "All categories" || p.category === category) && (!q || `${p.title} ${p.summary} ${p.ministry} ${p.category}`.toLocaleLowerCase().includes(q));
  }), [query, category]);
  return <main className="mx-auto max-w-4xl px-4 py-6 pb-24 sm:py-8">
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-4 sm:p-5">
      <h1 className="font-display text-2xl font-extrabold text-[#171717]">Explore policies</h1>
      <p className="mt-1.5 text-sm leading-relaxed text-[#4b5563]">Search the official policy and programme records currently available in this portal.</p>
      <div className="mt-4 flex flex-col gap-2 sm:flex-row">
        <label className="relative flex-1"><Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"/><input value={query} onChange={e => setQuery(e.target.value)} maxLength={200} aria-label="Search policies" placeholder="Policy, programme, ministry…" className="w-full rounded-lg border border-slate-300 py-2 pl-9 pr-3 text-sm"/></label>
        <select value={category} onChange={e => setCategory(e.target.value)} aria-label="Filter by category" className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm">{categories.map(x => <option key={x}>{x}</option>)}</select>
      </div>
      <p className="mt-3 text-xs text-slate-500">{results.length} of {officialPolicies.length} available records</p>
    </section>
    <div className="mt-4 space-y-4">{results.length ? results.map(p => <PolicyCard key={p.slug} post={p}/>) : <p className="rounded-xl border border-dashed p-8 text-center text-sm text-slate-600">No policies match this search.</p>}</div>
  </main>;
}
