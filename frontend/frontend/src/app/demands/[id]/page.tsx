"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type DemandCandidate, type Envelope } from "@/lib/api";

const FLAG_LABELS: Record<string, string> = {
  weak_cohesion: "Weak cohesion",
  moderately_unstable: "Moderately unstable",
  unstable: "Unstable",
  mixed: "Mixed content",
  no_opinion_pattern: "No-opinion pattern",
  single_field: "Single field",
  single_instrument: "Single instrument",
  channel_concentrated: "Channel concentrated",
};

type DetailResponse = Envelope<DemandCandidate>;

export default function DemandDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const [candidateId, setCandidateId] = useState("");
  const [result, setResult] = useState<DetailResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    params.then(p => setCandidateId(p.id));
  }, [params]);

  useEffect(() => {
    if (!candidateId) return;
    api<DetailResponse>(`/api/demands/${encodeURIComponent(candidateId)}`)
      .then(setResult)
      .catch((e: Error) => setError(e.message));
  }, [candidateId]);

  const d = result?.data;

  return (
    <main className="mx-auto max-w-4xl px-5 py-9 pb-24">
      <Link href="/demands" className="text-sm text-blue-800 hover:underline">← All demand candidates</Link>

      {error && (
        <p role="alert" className="mt-5 rounded-lg bg-red-50 p-3 text-sm text-red-800">{error}</p>
      )}
      {!d && !error && (
        <p className="mt-5 text-sm text-slate-500">Loading candidate…</p>
      )}

      {d && (
        <>
          {/* Header */}
          <div className="mt-5">
            <p className="text-xs font-semibold uppercase tracking-widest text-blue-800">
              Demand candidate · {d.candidate_id}
            </p>
            <h1 className="mt-2 text-3xl font-bold text-slate-900">
              {d.short_label || d.candidate_id}
            </h1>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-slate-300 bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-700">
                {d.state.replace(/_/g, " ")}
              </span>
              <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-800">
                Human review: {d.human_review_status}
              </span>
              {d.has_dissenting_evidence && (
                <span className="rounded-full border border-orange-300 bg-orange-50 px-2.5 py-0.5 text-xs font-medium text-orange-800">
                  ⚠ dissenting evidence present
                </span>
              )}
              {d.insufficient_evidence && (
                <span className="rounded-full border border-red-300 bg-red-50 px-2.5 py-0.5 text-xs font-medium text-red-700">
                  ⚠ insufficient evidence
                </span>
              )}
            </div>
          </div>

          {/* Caveats */}
          {result?.caveats && (
            <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-950">
              <p className="font-semibold mb-1">Research caveats</p>
              <ul className="list-disc list-inside space-y-0.5">
                {result.caveats.map((c, i) => <li key={i}>{c}</li>)}
              </ul>
            </div>
          )}

          {/* Insufficient-evidence warning */}
          {d.insufficient_evidence && (
            <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
              <strong>Insufficient evidence:</strong> {d.insufficient_reason}
            </div>
          )}

          {/* Demand text */}
          <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-6">
            <h2 className="text-xs font-bold uppercase tracking-widest text-slate-500">
              Extractive demand text <span className="font-normal normal-case">(not generated; extracted from concern group)</span>
            </h2>
            <p className="mt-3 text-lg leading-7 text-slate-800">{d.demand_text}</p>
            <p className="mt-3 text-xs text-slate-500">
              Generation method: <code>{d.generation_method}</code>
            </p>
          </section>

          {/* Statistics */}
          <section className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              ["Entries", d.entries.toLocaleString()],
              ["Unique texts", d.unique_texts.toLocaleString()],
              ["Independent texts", d.independent_texts.toLocaleString()],
              ["Source datasets", d.source_datasets.length.toString()],
            ].map(([label, value]) => (
              <div key={label} className="rounded-xl border border-slate-100 bg-slate-50 p-3 text-center">
                <div className="text-xl font-bold text-slate-900">{value}</div>
                <div className="mt-0.5 text-xs text-slate-500">{label}</div>
              </div>
            ))}
          </section>

          {/* Language breakdown */}
          {d.language_counts && Object.keys(d.language_counts).length > 0 && (
            <section className="mt-4 flex flex-wrap gap-3">
              {Object.entries(d.language_counts).map(([lang, count]) => (
                <span key={lang} className="rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-600">
                  {lang}: <strong>{count}</strong>
                </span>
              ))}
            </section>
          )}

          {/* Quality flags */}
          {d.quality_flags.length > 0 && (
            <section className="mt-4">
              <p className="text-xs font-semibold text-slate-500 mb-2">Quality flags from concern group</p>
              <div className="flex flex-wrap gap-2">
                {d.quality_flags.map(f => (
                  <span key={f} className="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-0.5 text-xs text-slate-600">
                    {FLAG_LABELS[f] || f}
                  </span>
                ))}
              </div>
            </section>
          )}

          {/* Evidence */}
          <section className="mt-8">
            <h2 className="text-xl font-semibold text-slate-900">Supporting evidence</h2>
            <p className="mt-1 text-xs text-slate-500">{d.evidence_coverage}</p>
            {d.has_dissenting_evidence && (
              <p className="mt-2 text-xs text-orange-700">
                ⚠ Quality flags indicate this group may contain mixed or dissenting content. Interpret evidence carefully.
              </p>
            )}

            {d.evidence && d.evidence.length > 0 ? (
              <div className="mt-4 space-y-3">
                {d.evidence.map((e, i) => (
                  <article
                    key={`${e.corpus_id}-${i}`}
                    className="rounded-xl border border-slate-200 bg-white p-4"
                  >
                    <p className="text-sm leading-6 text-slate-800">&ldquo;{e.quote}&rdquo;</p>
                    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 border-t pt-2 text-xs text-slate-500">
                      <span>corpus {e.corpus_id}</span>
                      <span>{e.source_dataset}</span>
                      <span>record {e.source_record_id}</span>
                      {e.language && <span>{e.language}</span>}
                      {e.question_slug && <span>question: {e.question_slug}</span>}
                      {e.selection_reason && (
                        <span className="italic">{e.selection_reason}</span>
                      )}
                      <Link
                        href={`/dataset`}
                        className="text-blue-700 hover:underline"
                      >
                        View in dataset →
                      </Link>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="mt-4 text-sm text-slate-500">No representative evidence available for this candidate.</p>
            )}
          </section>

          {/* Provenance */}
          <section className="mt-8 rounded-xl border border-slate-100 bg-slate-50 p-4 text-xs text-slate-500">
            <p className="font-semibold text-slate-700 mb-2">Provenance and traceability</p>
            <p>Candidate ID: <code>{d.candidate_id}</code> · Cluster: {d.cluster_id}</p>
            <p className="mt-1">Source datasets: {d.source_datasets.join(", ") || "—"}</p>
            <p className="mt-1">Source corpus IDs: {(d.evidence || []).map(e => e.corpus_id).join(", ")}</p>
            <p className="mt-2">{result?.scope || d.scope}</p>
          </section>

          {/* Navigation */}
          <div className="mt-6 flex gap-4 text-sm">
            <Link href="/demands" className="text-blue-800 hover:underline">← All candidates</Link>
            <Link href={`/groups/${d.cluster_id}`} className="text-blue-800 hover:underline">View concern group {d.cluster_id}</Link>
          </div>
        </>
      )}
    </main>
  );
}
