"use client";

import React, { useState } from "react";
import { Send, AlertTriangle, MessageSquare, Bot, Loader2 } from "lucide-react";
import { postApi } from "@/lib/api";

type StanceSuggestionResult = {
  final_stance: string | null;
  stance_raw: string | null;
  sentiment_prediction: string | null;
  prediction_method: string | null;
  sentiment_model_source: string | null;
  disclaimer: string;
};
type V2InferenceResponse = {
  component: "final_stance_inference";
  model: "linsvm";
  human_validated: false;
  pipeline: "FinalStancePredictor (direct three-class stance)";
  data: Array<StanceSuggestionResult & { input_status: "ok" | "too_short_or_empty" }>;
};

const STANCE_LABELS = new Set(["Support", "Neutral", "Unclear"]);

function isV2InferenceResponse(value: unknown): value is V2InferenceResponse {
  if (!value || typeof value !== "object") return false;
  const response = value as Partial<V2InferenceResponse>;
  if (
    response.component !== "final_stance_inference" ||
    response.model !== "linsvm" ||
    response.human_validated !== false ||
    response.pipeline !== "FinalStancePredictor (direct three-class stance)" ||
    !Array.isArray(response.data) ||
    response.data.length !== 1
  ) return false;

  const row = response.data[0] as Partial<V2InferenceResponse["data"][number]> | undefined;
  return Boolean(
    row &&
    (row.input_status === "ok" || row.input_status === "too_short_or_empty") &&
    (row.final_stance === null || (typeof row.final_stance === "string" && STANCE_LABELS.has(row.final_stance))) &&
    (row.stance_raw === null || (typeof row.stance_raw === "string" && STANCE_LABELS.has(row.stance_raw))) &&
    row.sentiment_prediction === null && row.sentiment_model_source === null &&
    (row.prediction_method === null || row.prediction_method === "stance_model_direct") &&
  typeof row.disclaimer === "string" && row.disclaimer.includes("reference labels")
  );
}

interface ShareOpinionFormProps {
  policySlug: string;
  policyTitle: string;
  policyId?: string;
  openedAt?: string;
  commentingDeadline?: string;
  onSubmissionSuccess?: (comment: { id: string; text: string; stakeholder: string }) => void;
}

export function ShareOpinionForm({
  policySlug,
  policyTitle,
  policyId,
  openedAt,
  commentingDeadline,
  onSubmissionSuccess
}: ShareOpinionFormProps) {
  const [stakeholder, setStakeholder] = useState("Citizen / Community Member");
  const [commentText, setCommentText] = useState("");
  const [submissionResult, setSubmissionResult] = useState<{
    status: "idle" | "local" | "error";
    message: string;
  }>({ status: "idle", message: "" });
  const [stanceSuggestion, setStanceSuggestion] = useState<StanceSuggestionResult | null>(null);
  const [stanceLoading, setStanceLoading] = useState(false);
  const [stanceError, setStanceError] = useState("");

  // Time limit removed: anyone can comment at anytime
  const isDeadlinePassed = false;
  const isFormDisabled = false;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = commentText.trim();

    if (!trimmed) {
      setSubmissionResult({ status: "error", message: "Enter a comment before requesting a suggestion." });
      return;
    }
    if (trimmed.length < 10) {
      setSubmissionResult({ status: "error", message: "Please write at least 10 characters." });
      return;
    }

    // This comment is shown for this page session only; it is not sent to the Ministry or
    // saved to a database. It IS sent, in-memory only, to the PolicyLens inference API below so
    // the saved stance/sentiment model can return a provisional suggestion.
    const id = `session-${(policyId || policySlug).replace(/[^a-zA-Z0-9-]/g, "-")}-${Date.now().toString(36)}`;
    onSubmissionSuccess?.({ id, text: trimmed, stakeholder });
    setSubmissionResult({
      status: "local",
      message: "Preview only: this comment is shown for this page session. It was not sent to the Ministry or saved to a database."
    });
    setCommentText("");
    setStanceSuggestion(null);
    setStanceError("");
    setStanceLoading(true);
    try {
      const raw = await postApi<unknown>("/api/inference/stance", {
        model: "linsvm",
        texts: [trimmed],
      });
      if (!isV2InferenceResponse(raw)) {
        throw new Error("The inference API returned an unexpected response.");
      }
      const row = raw.data[0];
      if (row.input_status === "ok") {
        setStanceSuggestion({
          final_stance: row.final_stance,
          stance_raw: row.stance_raw,
          sentiment_prediction: row.sentiment_prediction,
          prediction_method: row.prediction_method,
          sentiment_model_source: row.sentiment_model_source,
          disclaimer: row.disclaimer,
        });
      } else {
        setStanceError("The model could not produce a suggestion for this comment.");
      }
    } catch (err) {
      setStanceError((err as Error).message || "The inference API is unavailable.");
    } finally {
      setStanceLoading(false);
    }
  };

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-3 border-b border-[#f1f3f5]">
        <div>
          <h2 className="font-display text-[19px] sm:text-[21px] font-bold text-[#171717] flex items-center gap-2">
            <MessageSquare size={19} className="text-[#143454]" />
            <span>Share Your Opinion</span>
          </h2>
          <p className="text-[13px] sm:text-[14px] text-[#4b5563] mt-0.5">
            Comment on &ldquo;{policyTitle}&rdquo;. Public opinion submission is open at anytime.
          </p>
        </div>
        <div className="flex flex-col sm:items-end gap-0.5 shrink-0">
          <span className="text-[12px] text-[#16834A] bg-[#f0fdf4] px-2.5 py-1 rounded-md border border-[#bbf7d0] font-semibold flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-full bg-[#16834A]" aria-hidden="true" />
            Open for comments
          </span>
          <span className="text-[11px] text-[#64748b]">Public participation</span>
        </div>
      </div>

      <p className="text-[12px] leading-relaxed text-[#666666]">
        PolicyLens is an independent frontend preview. Comments are not official Government feedback, are not sent to MHA, and are not saved.
      </p>
      <p className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-[12px] leading-relaxed text-amber-950">
        Model scope: this provisional classifier was trained on historical Canadian consultation comments; its tone signal comes from a Twitter-trained model. It has not been validated for Indian policies or users and is not official or verified analysis.
      </p>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-1.5">
          <label htmlFor="stakeholder-select" className="block text-[13px] sm:text-[14px] font-semibold text-[#171717]">
            Stakeholder / Community Perspective (optional context)
          </label>
          <select
            id="stakeholder-select"
            value={stakeholder}
            onChange={(e) => setStakeholder(e.target.value)}
            disabled={isFormDisabled}
            className="w-full rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-2.5 text-[13px] text-[#171717] outline-none focus:border-[#143454] cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <option>Citizen / Community Member</option>
            <option>Small Business / MSME Representative</option>
            <option>Farmer / Agricultural Producer</option>
            <option>Academic / Policy Researcher</option>
            <option>Public Sector / Healthcare Worker</option>
            <option>Non-Profit / Civil Society Organization</option>
          </select>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-[13px]">
            <label htmlFor="comment-text" className="font-semibold text-[#171717]">Your Comment</label>
            <span className="text-[12px] text-[#6b7280]">{commentText.length} / 2,000 characters (min. 10)</span>
          </div>
          <textarea
            id="comment-text"
            rows={4}
            maxLength={2000}
            value={commentText}
            onChange={(e) => setCommentText(e.target.value)}
            disabled={isFormDisabled}
            placeholder={isFormDisabled ? "Commenting is closed for this policy." : "Write your opinion, concern, or suggested improvement…"}
            className="w-full rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-3 text-[13px] sm:text-[14px] text-[#171717] outline-none focus:border-[#E87518] focus:bg-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed leading-relaxed"
            required={!isFormDisabled}
          />
        </div>

        {submissionResult.status !== "idle" && (
          <div className={`rounded-lg border p-3.5 text-[13px] flex items-start gap-2 ${
            submissionResult.status === "local"
              ? "border-[#e5e7eb] bg-[#f8fafc] text-[#475569]"
              : "border-[#fca5a5] bg-[#fef2f2] text-[#991b1b]"
          }`}>
            {submissionResult.status === "local" ? <MessageSquare size={15} /> : <AlertTriangle size={15} />}
            <span>{submissionResult.message}</span>
          </div>
        )}

        {stanceLoading && (
          <div className="rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-3.5 text-[13px] text-[#4b5563] flex items-center gap-2">
            <Loader2 size={15} className="animate-spin text-[#143454]" />
            <span>Running the saved three-class stance model…</span>
          </div>
        )}

        {stanceError && (
          <div className="rounded-lg border border-[#fca5a5] bg-[#fef2f2] p-3.5 text-[13px] text-[#991b1b] flex items-start gap-2">
            <AlertTriangle size={15} className="shrink-0 mt-0.5" />
            <span>Stance suggestion unavailable: {stanceError}</span>
          </div>
        )}

        {stanceSuggestion && (
          <div className="rounded-lg border border-[#d6e0ea] bg-[#f0f4f8] p-3.5 text-[13px] text-[#143454] space-y-1.5">
            <div className="flex items-center gap-1.5 font-bold">
              <Bot size={15} className="text-[#E87518]" />
              <span>Model stance suggestion (provisional)</span>
            </div>
            <p>
              Stance: <strong>{stanceSuggestion.final_stance ?? "—"}</strong>
            </p>
            <p className="text-[11px] leading-relaxed text-[#475569]">{stanceSuggestion.disclaimer}</p>
          </div>
        )}

        <div className="flex items-center justify-end pt-1">
          <button
            type="submit"
            disabled={isFormDisabled || stanceLoading || commentText.trim().length < 10}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#143454] px-4 py-2.5 text-[13px] font-bold text-white shadow-xs hover:bg-[#0e2a47] disabled:opacity-50 disabled:cursor-not-allowed transition-colors cursor-pointer"
          >
            <Send size={14} /><span>Add Preview Comment</span>
          </button>
        </div>
      </form>
    </section>
  );
}
