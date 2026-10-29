"use client";
import { FormEvent, useEffect, useState } from "react";
import { api, type DatasetPage } from "@/lib/api";

export default function DatasetPageView() {
  const [query, setQuery] = useState(""); const [page, setPage] = useState<DatasetPage | null>(null); const [error, setError] = useState("");
  async function search(e?: FormEvent) { e?.preventDefault(); setError(""); try { setPage(await api<DatasetPage>(`/api/dataset?limit=50&q=${encodeURIComponent(query)}`)); } catch (x) { setError((x as Error).message); } }
  useEffect(() => { api<DatasetPage>("/api/dataset?limit=50").then(setPage).catch(e => setError(e.message)); }, []);
  return <main className="mx-auto max-w-5xl px-5 py-9 pb-24"><h1 className="text-3xl font-bold">Dataset explorer</h1><p className="mt-2 text-sm text-slate-600">Original comment text with source dataset, source record, and corpus identifiers.</p><form className="mt-5 flex gap-2" onSubmit={search}><input value={query} onChange={e => setQuery(e.target.value)} maxLength={200} placeholder="Search comment text" className="min-w-0 flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"/><button className="rounded-lg bg-blue-900 px-4 py-2 text-sm font-semibold text-white">Search</button></form>
    {error && <p role="alert" className="mt-4 text-sm text-red-700">{error}</p>}{!page && !error && <p className="mt-5 text-sm">Loading corpus…</p>}{page && <><p className="mt-4 text-xs text-slate-500">{page.total.toLocaleString()} matches · {page.scope}</p><div className="mt-4 space-y-3">{page.data.map(row => <article key={row.corpus_id} className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-sm leading-6 text-slate-800">{row.text}</p><p className="mt-3 border-t pt-2 text-xs text-slate-500">Corpus {row.corpus_id} · {row.source_dataset} · source record {row.source_record_id}{row.language ? ` · ${row.language}` : ""}{row.cluster_id !== undefined ? ` · group ${row.cluster_id}` : ""}</p></article>)}</div>{!page.data.length && <p className="mt-6 text-sm text-slate-500">No matching comments.</p>}</>}</main>;
}
