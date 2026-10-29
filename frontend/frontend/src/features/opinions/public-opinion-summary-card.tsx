"use client";

import React from "react";
import { Info, BarChart3, ExternalLink, ShieldCheck } from "lucide-react";
import type { OpinionSummary } from "./types";

export function PublicOpinionSummaryCard({ summary }: { summary: OpinionSummary }) {
  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[#f1f3f5]">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#143454] bg-[#e8eef5] mb-1">
            <BarChart3 size={12} className="text-[#E87518]" />
            AI-Powered Public Opinion Synthesis
          </div>
          <h2 className="font-display text-[18px] sm:text-[20px] font-bold text-[#171717]">
            Public Opinion Distribution
          </h2>
        </div>
        <div className="flex items-center gap-2 text-xs text-[#666666]">
          <span className="font-semibold text-[#171717]">{summary.totalAnalyzed}</span>
          <span>analyzed comments</span>
        </div>
      </div>

      {/* Tricolour-inspired Visual Distribution Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-[#666666]">
          <span>Stance Breakdown</span>
          <span className="text-[11px] font-medium text-[#16834A]">
            {summary.supportPercent}% Support • {summary.opposePercent}% Oppose
          </span>
        </div>

        <div className="h-4 w-full rounded-full overflow-hidden flex bg-[#f1f5f9] shadow-inner" aria-label="Visual Stance Distribution">
          {/* Support - Indian Green */}
          <div
            style={{ width: `${summary.supportPercent}%` }}
            className="h-full bg-[#16834A] transition-all duration-500"
            title={`Support: ${summary.supportCount} comments (${summary.supportPercent}%)`}
          />
          {/* Oppose - Dark Navy Blue */}
          <div
            style={{ width: `${summary.opposePercent}%` }}
            className="h-full bg-[#143454] transition-all duration-500"
            title={`Oppose: ${summary.opposeCount} comments (${summary.opposePercent}%)`}
          />
          {/* Neutral - Slate / Off-white Neutral */}
          <div
            style={{ width: `${summary.neutralPercent}%` }}
            className="h-full bg-[#94a3b8] transition-all duration-500"
            title={`Neutral: ${summary.neutralCount} comments (${summary.neutralPercent}%)`}
          />
          {/* Unclear - Saffron Accent */}
          <div
            style={{ width: `${summary.unclearPercent}%` }}
            className="h-full bg-[#E87518] transition-all duration-500"
            title={`Unclear: ${summary.unclearCount} comments (${summary.unclearPercent}%)`}
          />
        </div>

        {/* Legend Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2">
          {/* Support */}
          <div className="rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-2.5">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-[#16834A]" />
              <span className="text-xs font-semibold text-[#171717]">Support</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-lg font-bold text-[#16834A]">{summary.supportCount}</span>
              <span className="text-xs font-medium text-[#666666]">{summary.supportPercent}%</span>
            </div>
            <span className="text-[10px] text-[#666666] block leading-tight mt-0.5">Constructive support</span>
          </div>

          {/* Oppose */}
          <div className="rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-2.5">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-[#143454]" />
              <span className="text-xs font-semibold text-[#171717]">Oppose</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-lg font-bold text-[#143454]">{summary.opposeCount}</span>
              <span className="text-xs font-medium text-[#666666]">{summary.opposePercent}%</span>
            </div>
            <span className="text-[10px] text-[#666666] block leading-tight mt-0.5">Concerns &amp; criticisms</span>
          </div>

          {/* Neutral */}
          <div className="rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-2.5">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-[#94a3b8]" />
              <span className="text-xs font-semibold text-[#171717]">Neutral</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-lg font-bold text-[#334155]">{summary.neutralCount}</span>
              <span className="text-xs font-medium text-[#666666]">{summary.neutralPercent}%</span>
            </div>
            <span className="text-[10px] text-[#666666] block leading-tight mt-0.5">Informational / mixed</span>
          </div>

          {/* Unclear */}
          <div className="rounded-lg border border-[#e5e7eb] bg-[#f8fafc] p-2.5">
            <div className="flex items-center gap-1.5 mb-1">
              <span className="w-2.5 h-2.5 rounded-full bg-[#E87518]" />
              <span className="text-xs font-semibold text-[#171717]">Unclear</span>
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-lg font-bold text-[#E87518]">{summary.unclearCount}</span>
              <span className="text-xs font-medium text-[#666666]">{summary.unclearPercent}%</span>
            </div>
            <span className="text-[10px] text-[#666666] block leading-tight mt-0.5">Insufficient evidence</span>
          </div>
        </div>
      </div>

      {/* Dataset Provenance & Non-Representative Disclaimer */}
      <div className="rounded-lg border border-[#f1f3f5] bg-[#fafbfc] p-3 text-[11px] leading-relaxed text-[#666666] space-y-1">
        <div className="flex items-center gap-1.5 font-medium text-[#171717]">
          <ShieldCheck size={13} className="text-[#16834A]" />
          <span>Dataset Source: {summary.datasetSource}</span>
          <a
            href={summary.sourceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[#143454] hover:text-[#E87518] inline-flex items-center gap-0.5 ml-1"
          >
            Source <ExternalLink size={9} />
          </a>
        </div>
        <p className="flex items-start gap-1.5 text-[#555555]">
          <Info size={13} className="text-[#9ca3af] shrink-0 mt-0.5" />
          <span>{summary.disclaimer}</span>
        </p>
      </div>
    </section>
  );
}
