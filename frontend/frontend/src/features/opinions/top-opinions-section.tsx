"use client";

import React, { useState } from "react";
import { Sparkles, Award, ExternalLink, HelpCircle, AlertCircle, CheckCircle2, User } from "lucide-react";
import type { CommentItem, StanceType } from "./types";

function getStanceBadge(stance: StanceType) {
  switch (stance) {
    case "SUPPORT":
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-[#e6f4ea] px-2 py-0.5 text-[11px] font-bold text-[#16834A]">
          <CheckCircle2 size={11} /> Support
        </span>
      );
    case "OPPOSE":
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-[#fee2e2] px-2 py-0.5 text-[11px] font-bold text-[#991b1b]">
          <AlertCircle size={11} /> Oppose
        </span>
      );
    case "NEUTRAL":
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-[#f1f5f9] px-2 py-0.5 text-[11px] font-bold text-[#475569]">
          Neutral
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded-md bg-[#fff7ed] px-2 py-0.5 text-[11px] font-bold text-[#E87518]">
          <HelpCircle size={11} /> Unclear
        </span>
      );
  }
}

export function TopOpinionsSection({ comments }: { comments: CommentItem[] }) {
  const [viewMode, setViewMode] = useState<"top10" | "top1">("top10");

  const top10Comments =
    comments.filter((c) => c.isTopInformative).length > 0
      ? comments.filter((c) => c.isTopInformative)
      : comments.slice(0, 10);
  const displayedComments = viewMode === "top1" ? top10Comments.slice(0, 1) : top10Comments;

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-4">
      {/* Title & View Mode Toggle */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#f1f3f5]">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#143454] bg-[#e8eef5] mb-1">
            <Sparkles size={12} className="text-[#E87518]" />
            Post-Deadline NLP Analysis
          </div>
          <h2 className="font-display text-[19px] sm:text-[21px] font-bold text-[#171717]">
            Analyzed Public Opinions
          </h2>
          <p className="text-[13px] text-[#4b5563] mt-0.5">
            Key consultation submissions analyzed and ranked by policy demand clarity and viewpoint diversity.
          </p>
        </div>

        {/* Top 10 vs Top 1 Toggle */}
        <div className="inline-flex rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-1 text-[13px] self-start sm:self-center">
          <button
            type="button"
            onClick={() => setViewMode("top10")}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              viewMode === "top10"
                ? "bg-[#143454] text-white shadow-2xs"
                : "text-[#666666] hover:text-[#171717]"
            }`}
          >
            Top 10 Diverse Opinions
          </button>
          <button
            type="button"
            onClick={() => setViewMode("top1")}
            className={`px-3 py-1.5 rounded-md font-semibold transition-all cursor-pointer ${
              viewMode === "top1"
                ? "bg-[#E87518] text-white shadow-2xs"
                : "text-[#666666] hover:text-[#171717]"
            }`}
          >
            Top 1 Highest-Ranked
          </button>
        </div>
      </div>

      {/* List of Highlighted Comment Cards */}
      <div className="space-y-4">
        {displayedComments.map((comment, index) => (
          <div
            key={comment.id}
            className={`rounded-xl border p-4 sm:p-5 transition-shadow bg-white ${
              index === 0
                ? "border-[#E87518]/50 shadow-xs ring-1 ring-[#E87518]/20"
                : "border-[#e5e7eb] hover:shadow-xs"
            }`}
          >
            {/* Top Row: Rank Badge, Stance, Confidence & Stakeholder */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-[#f1f3f5] text-xs">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1 font-bold text-xs px-2 py-0.5 rounded bg-[#143454] text-white">
                  <Award size={11} className="text-[#E87518]" />
                  Rank #{index + 1}
                </span>
                {getStanceBadge(comment.stance)}
                <span className="text-[11px] font-semibold text-[#143454] bg-[#f1f5f9] px-2 py-0.5 rounded">
                  {comment.policyTopic.replace("_", " ")}
                </span>
              </div>

              <div className="flex items-center gap-2 text-[11px] text-[#666666]">
                {comment.confidenceScore && (
                  <span className="font-semibold text-[#16834A] bg-[#e6f4ea] px-1.5 py-0.5 rounded">
                    {Math.round(comment.confidenceScore * 100)}% Confidence
                  </span>
                )}
                <span className="flex items-center gap-1">
                  <User size={11} className="text-[#9ca3af]" />
                  {comment.stakeholder}
                </span>
              </div>
            </div>

            {/* Original Comment Text */}
            <div className="mt-3">
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#9ca3af] block mb-1">
                Verbatim Public Feedback
              </span>
              <blockquote className="text-[14px] leading-relaxed text-[#171717] font-medium bg-[#fafbfc] p-3 rounded-lg border-l-3 border-[#143454]">
                &ldquo;{comment.originalText}&rdquo;
              </blockquote>
            </div>

            {/* AI Synthesized Analysis Grid */}
            <div className="mt-3.5 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-[#f8fafc] p-3.5 rounded-lg border border-[#e5e7eb]">
              {/* Main Argument / Concern Highlight */}
              <div className="space-y-1">
                <span className="font-bold text-[#143454] uppercase tracking-wider text-[10px] block">
                  Core Argument / Concern:
                </span>
                <p className="text-[#334155] leading-relaxed">
                  {comment.mainArgument}
                </p>
              </div>

              {/* Action Requested or Suggested */}
              <div className="space-y-1">
                <span className="font-bold text-[#143454] uppercase tracking-wider text-[10px] block">
                  Action Requested for Policymakers:
                </span>
                <p className="text-[#334155] leading-relaxed">
                  {comment.suggestedAction || (
                    <span className="italic text-[#9ca3af]">No specific solution requested (Grievance/Concern only)</span>
                  )}
                </p>
              </div>

              {/* Why This Comment Matters */}
              <div className="md:col-span-2 pt-2 border-t border-[#e2e8f0] space-y-1">
                <span className="font-bold text-[#16834A] uppercase tracking-wider text-[10px] block">
                  Why this comment matters:
                </span>
                <p className="text-[#475569] leading-relaxed">
                  {comment.whyItMatters}
                </p>
              </div>
            </div>

            {/* Metadata Footer */}
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#666666]">
              <span className="truncate max-w-md">
                Source: {comment.sourceDocument}
              </span>
              <a
                href={comment.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-[#E87518] inline-flex items-center gap-1 font-medium"
              >
                View Consultation Record <ExternalLink size={10} />
              </a>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
