"use client";

import React from "react";
import { Lightbulb, ShieldAlert, CheckCircle } from "lucide-react";
import type { ImprovementSuggestion } from "./types";

export function PolicyImprovementSuggestions({
  suggestions
}: {
  suggestions: ImprovementSuggestion[];
}) {
  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-[#f8fafc] p-5 sm:p-6 shadow-2xs space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 pb-3 border-b border-[#e2e8f0]">
        <div>
          <div className="inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#16834A] bg-[#e6f4ea] mb-1">
            <Lightbulb size={12} className="text-[#E87518]" />
            Synthesis for Policymakers
          </div>
          <h2 className="font-display text-[18px] sm:text-[20px] font-bold text-[#171717]">
            What Could Be Improved?
          </h2>
          <p className="text-xs text-[#666666] mt-0.5">
            Key operational improvements synthesized from recurring citizen grievances and feedback.
          </p>
        </div>

        {/* AI Disclaimer Badge */}
        <div className="inline-flex items-center gap-1.5 rounded-lg border border-[#e5e7eb] bg-white px-2.5 py-1.5 text-[11px] text-[#666666]">
          <ShieldAlert size={13} className="text-[#E87518] shrink-0" />
          <span>AI-generated derived synthesis • Not official government policy</span>
        </div>
      </div>

      {/* Grid of Actionable Suggestions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {suggestions.map((item, idx) => (
          <div
            key={idx}
            className="rounded-xl border border-[#e2e8f0] bg-white p-4 shadow-2xs flex flex-col justify-between space-y-3"
          >
            <div className="space-y-2">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#143454] bg-[#e8eef5] px-2 py-0.5 rounded inline-block">
                {item.category}
              </span>
              <div>
                <span className="text-[10px] font-bold uppercase text-[#9ca3af] block">Public Concern:</span>
                <p className="text-xs text-[#444444] leading-relaxed mt-0.5 font-medium">
                  {item.concernIdentified}
                </p>
              </div>
            </div>

            <div className="pt-2.5 border-t border-[#f1f3f5] space-y-1 bg-[#f9fafb] p-3 rounded-lg">
              <span className="text-[10px] font-bold uppercase text-[#16834A] flex items-center gap-1">
                <CheckCircle size={11} /> Suggested Action:
              </span>
              <p className="text-xs text-[#171717] font-semibold leading-relaxed">
                {item.actionableSuggestion}
              </p>
              <span className="text-[10px] text-[#666666] block pt-1">
                Supported by {item.supportingCommentsCount} verified comments
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
