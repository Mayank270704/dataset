"use client";
import Link from "next/link";
import { useCallback } from "react";
import { Bookmark } from "lucide-react";
import { officialPolicies } from "@/features/feed/demo-data";
import { PolicyCard } from "@/features/feed/policy-card";
import { readSaved, usePolicyLibrary } from "@/lib/policy-library";

export default function SavedPage() {
  const reader = useCallback(() => readSaved(), []);
  const saved = usePolicyLibrary(reader, []);
  const policies = officialPolicies.filter(p => saved.includes(p.slug));
  return <main className="mx-auto max-w-4xl px-4 py-8 pb-24"><h1 className="font-display text-3xl font-extrabold">Saved policies</h1><p className="mt-2 text-sm text-slate-600">Your saved items are stored in this browser only.</p>
    {policies.length ? <div className="mt-5 space-y-4">{policies.map(post => <PolicyCard key={post.slug} post={post}/>)}</div> : <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center"><Bookmark className="mx-auto text-slate-400"/><p className="mt-2 font-semibold">No saved policies</p><p className="mt-1 text-sm text-slate-600">Use “Save policy” on an item to keep it here.</p><Link href="/discover" className="mt-4 inline-block text-sm font-semibold text-blue-800 hover:underline">Explore policies</Link></div>}
  </main>;
}
