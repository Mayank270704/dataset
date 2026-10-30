"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type DemandsPage, type DemandCandidate } from "@/lib/api";

const FLAG_LABELS: Record<string, string> = {
  weak_cohesion: "weak cohesion",
  moderately_unstable: "moderately unstable",
  unstable: "unstable",
  mixed: "mixed content",
  no_opinion_pattern: "no-opinion pattern",
  single_field: "single field",
  single_instrument: "single instrument",
  channel_concentrated: "channel concentrated",
};

function CaveatBanner({ caveats }: { caveats?: string[] }) {
  if (!caveats?.length) return null;
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-950">
      <p className="font-semibold mb-1">Research caveats — read before interpreting</p>
      <ul className="list-disc list-inside space-y-0.5">
        {caveats.map((c, i) => <li key={i}>{c}</li>)}
      </ul>
    </div>
  );
}

function DemandCard({ d }: { d: DemandCandidate }) {
  const insufficient = d.insufficient_evidence;
  return (
    <Link
      href={`/demands/${encodeURIComponent(d.candidate_id)}`}
      className={`block rounded-2xl border bg-white p-5 transition-shadow hover:shadow-md ${insufficient ? "border-slate-200 opacity-75" : "border-slate-200 hover:border-blue-700"}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h2 className="font-semibold text-slate-900 leading-5">
          {d.short_label || d.candidate_id}
        </h2>
        <div className="flex flex-wrap gap-1">
          {d.has_dissenting_evidence && (
            <span className="rounded-full border border-orange-200 bg-orange-50 px-2 py-0.5 text-[10px] font-medium text-orange-800">dissenting evidence</span>
          )}
          {d.insufficient_evidence && (
            <span className="rounded-full border border-slate-300 bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-600">insufficient evidence</span>
          )}
          {d.quality_flags.map(f => (
            <span key={f} className="rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] text-slate-500">{FLAG_LABELS[f] || f}</span>
          ))}
        </div>
      </div>

      <p className="mt-2 text-sm leading-5 text-slate-700 line-clamp-2">{d.demand_text}</p>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
        <span>{d.entries.toLocaleString()} entries</span>
        <span>{d.independent_texts.toLocaleString()} independent texts</span>
        <span className="capitalize">{d.state.replace(/_/g, " ")}</span>
        <span className="capitalize text-amber-700">{d.human_review_status}</span>
      </div>
    </Link>
  );
}

export default function DemandsPage() {
  const [result, setResult] = useState<DemandsPage | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<DemandsPage>("/api/demands?limit=100")
      .then(setResult)
      .catch((e: Error) => setError(e.message));
  }, []);

  const sufficient = result?.data.filter(d => !d.insufficient_evidence) ?? [];
  const insufficient = result?.data.filter(d => d.insufficient_evidence) ?? [];

  return (
    <main className="mx-auto max-w-6xl px-5 py-9 pb-24">
      <p className="text-xs font-bold uppercase tracking-[.18em] text-blue-800">PolicyLens · research workspace</p>
      <h1 className="mt-2 text-3xl font-bold text-slate-900">Collective-demand candidates</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
        Extractive demand candidates derived from verified concern groups. Each candidate is assembled from
        the group&apos;s representative comments and extractive description. These are{" "}
        <strong>generated candidates, not validated demands</strong>. Human review is pending for all.
      </p>

      {error && (
        <p role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-800">
          Demand generation unavailable: {error}. Start the API with <code>python -m app.api</code>.
        </p>
      )}

      {!result && !error && (
        <p className="mt-5 text-sm text-slate-500">Loading demand candidates…</p>
      )}

      {result && (
        <>
          <div className="mt-4">
            <CaveatBanner caveats={result.caveats} />
          </div>

          <div className="mt-5 flex flex-wrap gap-4 text-sm text-slate-600">
            <span className="rounded-lg bg-slate-100 px-3 py-1.5">
              <strong>{result.total}</strong> total candidates
            </span>
            <span className="rounded-lg bg-emerald-50 px-3 py-1.5 text-emerald-800">
              <strong>{sufficient.length}</strong> with sufficient evidence
            </span>
            <span className="rounded-lg bg-amber-50 px-3 py-1.5 text-amber-800">
              <strong>{insufficient.length}</strong> flagged insufficient evidence
            </span>
            <span className="rounded-lg bg-slate-100 px-3 py-1.5">
              Generation: <code className="text-xs">{result.kind}</code>
            </span>
          </div>

          {sufficient.length > 0 && (
            <section className="mt-8">
              <h2 className="text-lg font-semibold text-slate-900">Candidates with sufficient evidence</h2>
              <p className="mt-1 text-xs text-slate-500">Ordered by exploratory priority rank · extractive · not human-validated</p>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {sufficient.map(d => <DemandCard key={d.candidate_id} d={d} />)}
              </div>
            </section>
          )}

          {insufficient.length > 0 && (
            <section className="mt-10">
              <h2 className="text-lg font-semibold text-slate-700">Candidates flagged: insufficient evidence</h2>
              <p className="mt-1 text-xs text-slate-500">
                These groups meet the candidate tier but fall below the independent-text threshold. Listed for completeness.
              </p>
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {insufficient.map(d => <DemandCard key={d.candidate_id} d={d} />)}
              </div>
            </section>
          )}

          {result.data.length === 0 && (
            <p className="mt-6 text-sm text-slate-500">No demand candidates were generated from the verified concern groups.</p>
          )}

          <p className="mt-8 text-xs text-slate-400">{result.scope}</p>
        </>
      )}
    </main>
  );
}
