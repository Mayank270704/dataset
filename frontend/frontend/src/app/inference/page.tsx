"use client";
import { FormEvent, useState } from "react";
import { postApi } from "@/lib/api";
import { recordVisit } from "@/lib/policy-library";

type V1Prediction = {
  text: string;
  input_status: string;
  stance_suggestion: string | null;
  stance_margin: number | null;
  sentiment_suggestion: string | null;
  sentiment_margin: number | null;
  stance_label_source: string;
  concern_group: null;
  concern_group_status: string;
  priority: null;
  priority_status: string;
};
type V1Response = {
  component: "consultation_inference";
  kind: string;
  model: string;
  human_validated: false;
  label_provenance: string;
  caveats: string[];
  data: V1Prediction[];
};

type V2Prediction = {
  text: string;
  input_status: string;
  final_stance: string | null;
  stance_raw: string | null;
  stance_margin: number | null;
  sentiment_prediction: string | null;
  sentiment_model_source: string | null;
  prediction_method: string | null;
  stance_label_source: string | null;
  disclaimer: string;
};
type V2Response = {
  component: "final_stance_inference";
  kind: string;
  model: string;
  human_validated: false;
  pipeline: string;
  label_provenance: string;
  caveats: string[];
  data: V2Prediction[];
};

type InferenceResponse = V1Response | V2Response;
type ContextStagingRow = {
  item_index: number;
  input_status: string;
  final_stance: "Support" | "Neutral" | "Unclear" | null;
  stance_margin: number | null;
  question_missing: boolean;
  context_missing: boolean;
  language_status: string;
  disclaimer: string;
};
type ContextStagingResponse = {
  component: "context_staging_inference";
  kind: string;
  candidate: string;
  human_validated: false;
  labels_supported: string[];
  language_policy: string;
  caveats: string[];
  data: ContextStagingRow[];
};
type FiveClassStagingRow = {
  item_index: number;
  input_status: string;
  final_stance: "Support" | "Neutral" | "Oppose" | "Mixed" | "Unclear" | null;
  stance_margin: number | null;
  question_missing: boolean;
  context_missing: boolean;
  disclaimer: string;
};
type FiveClassStagingResponse = {
  component: "five_class_stance_staging_inference";
  kind: string;
  candidate: string;
  human_validated: false;
  labels_supported: string[];
  caveats: string[];
  data: FiveClassStagingRow[];
};

const STANCE_COLORS: Record<string, string> = {
  Support: "bg-emerald-50 text-emerald-800 border-emerald-200",
  Oppose: "bg-red-50 text-red-700 border-red-200",
  Mixed: "bg-amber-50 text-amber-800 border-amber-200",
  Neutral: "bg-slate-100 text-slate-700 border-slate-200",
  Unclear: "bg-slate-100 text-slate-500 border-slate-200",
};
const SENTIMENT_COLORS: Record<string, string> = {
  Positive: "bg-emerald-50 text-emerald-700 border-emerald-200",
  Neutral: "bg-slate-100 text-slate-700 border-slate-200",
  Negative: "bg-red-50 text-red-700 border-red-200",
  positive: "bg-emerald-50 text-emerald-700 border-emerald-200",
  negative: "bg-red-50 text-red-700 border-red-200",
};

const EXAMPLES = [
  "We need stronger transparency requirements for government open data.",
  "This policy is a complete waste of money and should be scrapped.",
  "The meeting schedule was published on the department website.",
];

export default function InferencePage() {
  const [texts, setTexts] = useState("");
  const [model, setModel] = useState<"linsvm" | "logreg">("linsvm");
  const [includeOpposeHeuristic, setIncludeOpposeHeuristic] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [stagingQuestion, setStagingQuestion] = useState("");
  const [stagingContext, setStagingContext] = useState("");
  const [stagingComment, setStagingComment] = useState("");
  const [stagingLanguage, setStagingLanguage] = useState("");
  const [stagingResult, setStagingResult] = useState<ContextStagingResponse | null>(null);
  const [stagingError, setStagingError] = useState("");
  const [stagingLoading, setStagingLoading] = useState(false);
  const [fiveQuestion, setFiveQuestion] = useState("");
  const [fiveContext, setFiveContext] = useState("");
  const [fiveComment, setFiveComment] = useState("");
  const [fiveResult, setFiveResult] = useState<FiveClassStagingResponse | null>(null);
  const [fiveError, setFiveError] = useState("");
  const [fiveLoading, setFiveLoading] = useState(false);

  async function runInference() {
    const lines = texts
      .split("\n")
      .map(l => l.trim())
      .filter(Boolean)
      .slice(0, 16);
    if (!lines.length) {
      setError("Enter at least one comment before analyzing.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body: Record<string, unknown> = { model, texts: lines };
      if (includeOpposeHeuristic) body.pipeline = "v2";
      const res = await postApi<InferenceResponse>("/api/inference/consultation", body);
      setResult(res);
      recordVisit({ slug: "consultation-analysis", kind: "inference", visitedAt: new Date().toISOString() });
    } catch (x) {
      setError((x as Error).message || "The inference request failed. The API may be unavailable.");
    } finally {
      setLoading(false);
    }
  }

  function submit(e: FormEvent) {
    e.preventDefault();
    void runInference();
  }

  async function runContextStaging(e: FormEvent) {
    e.preventDefault();
    setStagingLoading(true);
    setStagingError("");
    setStagingResult(null);
    try {
      const response = await postApi<ContextStagingResponse>("/api/inference/stance/context-staging", {
        items: [{
          question: stagingQuestion,
          context: stagingContext,
          comment: stagingComment,
          ...(stagingLanguage.trim() ? { language: stagingLanguage.trim() } : {}),
        }],
      });
      setStagingResult(response);
    } catch (x) {
      setStagingError((x as Error).message || "The context staging request failed.");
    } finally {
      setStagingLoading(false);
    }
  }

  async function runFiveClassStaging(e: FormEvent) {
    e.preventDefault();
    setFiveLoading(true);
    setFiveError("");
    setFiveResult(null);
    try {
      const response = await postApi<FiveClassStagingResponse>("/api/inference/stance/five-class-staging", {
        items: [{ question: fiveQuestion, context: fiveContext, comment: fiveComment }],
      });
      setFiveResult(response);
    } catch (x) {
      setFiveError((x as Error).message || "The five-class staging request failed.");
    } finally {
      setFiveLoading(false);
    }
  }

  const lineCount = texts.split("\n").filter(l => l.trim()).length;

  return (
    <main className="mx-auto max-w-5xl px-5 py-9 pb-24">
      <p className="text-xs font-bold uppercase tracking-[.18em] text-blue-800">PolicyLens · provisional inference</p>
      <h1 className="mt-2 text-3xl font-bold text-slate-900">Comment stance &amp; sentiment analysis</h1>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
        PolicyLens studies how people respond to Canadian federal Open Government consultations.
        This page runs the project&apos;s saved TF-IDF baseline models directly on text you enter
        below. Results are <strong>provisional suggestions</strong>, not ground truth — the
        training targets are the repository reference labels; results outside this dataset are not established.
      </p>

      {/* Mandatory caveats — always visible */}
      <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-4 text-xs leading-5 text-amber-950">
        <p className="font-semibold mb-1">Critical limitations — must read before use</p>
        <ul className="list-disc list-inside space-y-0.5">
          <li><strong>Standard model:</strong> outputs Support, Neutral, or Unclear only — it cannot predict Oppose.</li>
          <li><strong>Supported stance classes:</strong> this model recognizes Support, Neutral and Unclear. Oppose and Mixed are unavailable because the saved model has no reliable training examples for them.</li>
          <li>Evaluation measures agreement with the repository reference labels; it does not establish performance on other consultation sources.</li>
          <li>Decision margins are not probabilities and are not calibrated confidence scores.</li>
          <li>A previously recorded failure classified a strongly opposing statement as Support.</li>
          <li>Models were trained on ~400 Canadian consultation comments; coverage of other domains/languages is unknown.</li>
          <li>Sentiment (tone) and stance (position) are distinct constructs; do not conflate them.</li>
          <li>Reported evaluation scores do not establish human-validated accuracy.</li>
          <li>Input text is processed in memory only and is not stored by this endpoint.</li>
        </ul>
      </div>

      {/* Form */}
      <form onSubmit={submit} className="mt-6 rounded-2xl border border-slate-200 bg-white p-5">
        <label className="block text-sm font-semibold text-slate-700 mb-2">
          Comment text (one per line, max 16, max 5000 chars each)
        </label>
        <textarea
          value={texts}
          onChange={e => setTexts(e.target.value)}
          rows={6}
          maxLength={80000}
          placeholder={"Enter one consultation comment per line…\nExample: We need stronger transparency requirements.\nExample: This policy seems reasonable to me."}
          className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm font-mono text-slate-800 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <p className="mt-1 text-xs text-slate-500">{lineCount} / 16 comments entered</p>

        {/* Clearly-labelled examples */}
        <div className="mt-2 flex flex-wrap gap-2">
          <span className="text-xs text-slate-500">Examples (click to insert):</span>
          {EXAMPLES.map(ex => (
            <button
              key={ex}
              type="button"
              onClick={() => setTexts(prev => (prev.trim() ? `${prev.trim()}\n${ex}` : ex))}
              className="rounded-full border border-slate-300 bg-slate-50 px-3 py-1 text-xs text-slate-600 hover:bg-slate-100"
              title="Example comment, not a real submission"
            >
              {ex.length > 40 ? `${ex.slice(0, 40)}…` : ex}
            </button>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap items-end gap-4">
          <div>
            <label className="block text-xs font-semibold text-slate-600 mb-1">Model</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value as "linsvm" | "logreg")}
              className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              <option value="linsvm">Linear SVM (default)</option>
              <option value="logreg">Logistic Regression</option>
            </select>
          </div>
          <label className="flex items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={includeOpposeHeuristic}
              onChange={e => setIncludeOpposeHeuristic(e.target.checked)}
              className="h-4 w-4 rounded border-slate-300"
            />
            Use direct three-class stance inference (sentiment is not used for stance)
          </label>
          <button
            type="submit"
            disabled={loading || !lineCount}
            className="rounded-lg bg-blue-900 px-5 py-2 text-sm font-semibold text-white disabled:opacity-60"
          >
            {loading ? "Running inference…" : "Analyze"}
          </button>
        </div>
      </form>

      {error && (
        <div role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-800">
          <p>{error}</p>
          <button
            type="button"
            onClick={() => void runInference()}
            disabled={loading || !lineCount}
            className="mt-2 rounded-lg border border-red-300 bg-white px-3 py-1 text-xs font-semibold text-red-800 hover:bg-red-50 disabled:opacity-60"
          >
            Retry
          </button>
        </div>
      )}

      {result && result.component === "consultation_inference" && (
        <>
          <div className="mt-6 rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-900">
            <strong>Provisional suggestions only.</strong> {result.label_provenance}
          </div>

          {result.caveats && (
            <div className="mt-3 rounded-lg border border-amber-100 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
              {result.caveats.map((c, i) => <p key={i}>· {c}</p>)}
            </div>
          )}

          <section className="mt-5 space-y-4">
            <h2 className="text-lg font-semibold text-slate-900">
              Results — model: <code className="text-sm">{result.model}</code>
            </h2>
            {result.data.map((row, i) => (
              <article key={i} className="rounded-2xl border border-slate-200 bg-white p-5">
                <p className="text-sm leading-6 text-slate-800 font-mono">&ldquo;{row.text}&rdquo;</p>

                {row.input_status !== "ok" ? (
                  <p className="mt-2 text-xs text-slate-500">Input status: {row.input_status} — no prediction made</p>
                ) : (
                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border p-3">
                      <p className="text-xs font-semibold text-slate-500 mb-2">Stance suggestion</p>
                      <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${STANCE_COLORS[row.stance_suggestion || ""] || "bg-slate-50 text-slate-600 border-slate-200"}`}>
                        {row.stance_suggestion ?? "—"}
                      </span>
                      <p className="mt-1 text-[10px] text-slate-400">Labels: Support · Neutral · Unclear only. Cannot predict Oppose.</p>
                    </div>

                    <div className="rounded-xl border p-3">
                      <p className="text-xs font-semibold text-slate-500 mb-2">Sentiment suggestion</p>
                      <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${SENTIMENT_COLORS[row.sentiment_suggestion || ""] || "bg-slate-50 text-slate-600 border-slate-200"}`}>
                        {row.sentiment_suggestion ?? "—"}
                      </span>
                      <p className="mt-1 text-[10px] text-slate-400">Tone only — distinct from stance.</p>
                    </div>
                  </div>
                )}

                <div className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">
                  <p>Concern group: <em>{row.concern_group_status}</em></p>
                  <p>Priority: <em>{row.priority_status}</em></p>
                </div>
              </article>
            ))}
          </section>
        </>
      )}

      {result && result.component === "final_stance_inference" && (
        <>
          <div className="mt-6 rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-900">
            <strong>Provisional suggestions only.</strong> {result.label_provenance} · pipeline: {result.pipeline}
          </div>

          {result.caveats && (
            <div className="mt-3 rounded-lg border border-amber-100 bg-amber-50 p-3 text-xs leading-5 text-amber-900">
              {result.caveats.map((c, i) => <p key={i}>· {c}</p>)}
            </div>
          )}

          <section className="mt-5 space-y-4">
            <h2 className="text-lg font-semibold text-slate-900">
              Results (direct three-class stance) — model: <code className="text-sm">{result.model}</code>
            </h2>
            {result.data.map((row, i) => (
              <article key={i} className="rounded-2xl border border-slate-200 bg-white p-5">
                <p className="text-sm leading-6 text-slate-800 font-mono">&ldquo;{row.text}&rdquo;</p>

                {row.input_status !== "ok" ? (
                  <p className="mt-2 text-xs text-slate-500">Input status: {row.input_status} — no prediction made</p>
                ) : (
                  <div className="mt-4">
                    <div className="rounded-xl border p-3">
                      <p className="text-xs font-semibold text-slate-500 mb-2">Final stance</p>
                      <span className={`rounded-full border px-3 py-1 text-sm font-semibold ${STANCE_COLORS[row.final_stance || ""] || "bg-slate-50 text-slate-600 border-slate-200"}`}>
                        {row.final_stance ?? "—"}
                      </span>
                      <p className="mt-2 text-xs text-slate-500">Suggestion trained against repository stance reference labels; external performance is not established.</p>
                    </div>
                    <p className="mt-2 rounded-lg bg-amber-50 p-2 text-[11px] leading-5 text-amber-900">{row.disclaimer}</p>
                  </div>
                )}
              </article>
            ))}
          </section>
        </>
      )}

      <section className="mt-10 rounded-2xl border-2 border-dashed border-indigo-300 bg-indigo-50/40 p-5">
        <p className="text-xs font-bold uppercase tracking-wide text-indigo-800">Separate staging prototype</p>
        <h2 className="mt-1 text-xl font-bold text-slate-900">Question and context aware stance</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Uses a versioned staging candidate with Support, Neutral, and Unclear outputs. It does not replace the production model. Language metadata is accepted for testing but ignored by the model.
        </p>
        <form onSubmit={runContextStaging} className="mt-4 grid gap-4">
          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            Consultation question (optional, max 2,000 characters)
            <textarea value={stagingQuestion} onChange={e => setStagingQuestion(e.target.value)} maxLength={2000} rows={2} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal" />
          </label>
          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            Additional context (optional, max 5,000 characters)
            <textarea value={stagingContext} onChange={e => setStagingContext(e.target.value)} maxLength={5000} rows={3} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal" />
          </label>
          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            Comment (max 5,000 characters; fewer than 3 non-space characters returns no prediction)
            <textarea value={stagingComment} onChange={e => setStagingComment(e.target.value)} maxLength={5000} rows={4} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono font-normal" />
          </label>
          <label className="grid max-w-sm gap-1 text-sm font-semibold text-slate-700">
            Language metadata (optional; ignored; unsupported values are not model features)
            <input value={stagingLanguage} onChange={e => setStagingLanguage(e.target.value)} maxLength={100} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal" placeholder="e.g. Klingon" />
          </label>
          <button type="submit" disabled={stagingLoading} className="w-fit rounded-lg bg-indigo-800 px-5 py-2 text-sm font-semibold text-white disabled:opacity-60">
            {stagingLoading ? "Running staging model…" : "Run staging prototype"}
          </button>
        </form>
        {stagingError && <p role="alert" className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-800">{stagingError}</p>}
        {stagingResult && stagingResult.data.map(row => (
          <article key={row.item_index} className="mt-4 rounded-xl border border-indigo-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-700">Staging result</p>
            {row.input_status === "ok" ? (
              <p className="mt-2 inline-block rounded-full border border-indigo-200 bg-indigo-50 px-3 py-1 font-semibold text-indigo-900">{row.final_stance}</p>
            ) : (
              <p className="mt-2 text-sm text-slate-600">No stance prediction: comment is empty or too short.</p>
            )}
            <p className="mt-2 text-xs text-slate-600">Question missing: {row.question_missing ? "yes" : "no"}; context missing: {row.context_missing ? "yes" : "no"}; language: {row.language_status}.</p>
            <p className="mt-2 text-xs text-slate-500">{row.disclaimer}</p>
          </article>
        ))}
      </section>

      <section className="mt-8 rounded-2xl border-2 border-dashed border-violet-300 bg-violet-50/40 p-5">
        <p className="text-xs font-bold uppercase tracking-wide text-violet-800">Five-class synthetic staging candidate</p>
        <h2 className="mt-1 text-xl font-bold text-slate-900">Expanded stance schema (staging only)</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Candidate {"stance_context_wordchar_5class_synth_v1"} adds Oppose and Mixed using explicitly synthetic training examples. It is not human validated; current original grouped evaluation has no Mixed examples and only one Oppose record, so those classes&apos; real-world performance is unknown. Production remains on its existing model.
        </p>
        <form onSubmit={runFiveClassStaging} className="mt-4 grid gap-4">
          <label className="grid gap-1 text-sm font-semibold text-slate-700">Consultation question (optional)
            <textarea value={fiveQuestion} onChange={e => setFiveQuestion(e.target.value)} maxLength={2000} rows={2} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal" />
          </label>
          <label className="grid gap-1 text-sm font-semibold text-slate-700">Additional context (optional)
            <textarea value={fiveContext} onChange={e => setFiveContext(e.target.value)} maxLength={5000} rows={3} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-normal" />
          </label>
          <label className="grid gap-1 text-sm font-semibold text-slate-700">Comment (required; fewer than 3 non-space characters returns no prediction)
            <textarea value={fiveComment} onChange={e => setFiveComment(e.target.value)} maxLength={5000} rows={4} className="rounded-lg border border-slate-300 bg-white px-3 py-2 font-mono font-normal" />
          </label>
          <button type="submit" disabled={fiveLoading} className="w-fit rounded-lg bg-violet-800 px-5 py-2 text-sm font-semibold text-white disabled:opacity-60">
            {fiveLoading ? "Running five-class staging model…" : "Run five-class staging prototype"}
          </button>
        </form>
        {fiveError && <p role="alert" className="mt-3 rounded-lg bg-red-50 p-3 text-sm text-red-800">{fiveError}</p>}
        {fiveResult?.caveats.map(caveat => <p key={caveat} className="mt-2 text-xs leading-5 text-amber-900">{caveat}</p>)}
        {fiveResult?.data.map(row => (
          <article key={row.item_index} className="mt-4 rounded-xl border border-violet-200 bg-white p-4">
            <p className="text-sm font-semibold text-slate-700">Staging result</p>
            {row.input_status === "ok" ? (
              <p className={`mt-2 inline-block rounded-full border px-3 py-1 font-semibold ${STANCE_COLORS[row.final_stance || ""] || "bg-slate-50 text-slate-700 border-slate-200"}`}>{row.final_stance}</p>
            ) : <p className="mt-2 text-sm text-slate-600">No stance prediction: comment is empty or too short.</p>}
            <p className="mt-2 text-xs text-slate-600">Question missing: {row.question_missing ? "yes" : "no"}; context missing: {row.context_missing ? "yes" : "no"}.</p>
            <p className="mt-2 text-xs text-slate-500">{row.disclaimer}</p>
          </article>
        ))}
      </section>
    </main>
  );
}
