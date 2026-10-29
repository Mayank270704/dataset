"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Envelope, type Group } from "@/lib/api";
export default function GroupsPage() {
  const [groups, setGroups] = useState<Envelope<Group[]> | null>(null); const [error, setError] = useState("");
  useEffect(() => { api<Envelope<Group[]>>("/api/groups").then(setGroups).catch(e => setError(e.message)); }, []);
  return <main className="mx-auto max-w-6xl px-5 py-9 pb-24"><h1 className="text-3xl font-bold">Exploratory concern groups</h1><p className="mt-2 text-sm text-slate-600">Automated semantic clusters. Labels and descriptions are unvalidated; review source comments before interpretation.</p>{error && <p role="alert" className="mt-4 text-sm text-red-700">{error}</p>}{!groups && !error && <p className="mt-5 text-sm">Loading saved groups…</p>}<div className="mt-5 grid gap-3 md:grid-cols-2">{groups?.data.map(g => <Link href={`/groups/${g.cluster_id}`} key={g.cluster_id} className="rounded-xl border border-slate-200 bg-white p-4 hover:border-blue-700"><div className="flex justify-between gap-3"><h2 className="font-semibold text-slate-900">{g.short_label || `Group ${g.cluster_id}`}</h2><span className="text-xs text-slate-500">{g.tier}</span></div><p className="mt-2 line-clamp-3 text-sm text-slate-600">{g.description}</p><p className="mt-3 text-xs text-slate-500">{g.entries.toLocaleString()} entries · {g.independent_texts.toLocaleString()} independent texts · group {g.cluster_id}</p></Link>)}</div>{groups?.data.length === 0 && <p className="mt-5 text-sm">No verified group results available.</p>}</main>;
}
