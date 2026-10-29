"use client";

import React, { useState, useMemo } from "react";
import {
  MessageSquare,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Bot,
  ExternalLink,
  User,
  CheckCircle2,
  AlertCircle,
  HelpCircle
} from "lucide-react";
import type { CommentItem, StanceType } from "./types";

function getStanceBadge(stance: StanceType) {
  switch (stance) {
    case "SUPPORT":
      return (
        <span className="inline-flex items-center gap-1 rounded bg-[#e6f4ea] px-2 py-0.5 text-[10px] font-bold text-[#16834A]">
          <CheckCircle2 size={10} /> Support
        </span>
      );
    case "OPPOSE":
      return (
        <span className="inline-flex items-center gap-1 rounded bg-[#fee2e2] px-2 py-0.5 text-[10px] font-bold text-[#991b1b]">
          <AlertCircle size={10} /> Oppose
        </span>
      );
    case "NEUTRAL":
      return (
        <span className="inline-flex items-center gap-1 rounded bg-[#f1f5f9] px-2 py-0.5 text-[10px] font-bold text-[#475569]">
          Neutral
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1 rounded bg-[#fff7ed] px-2 py-0.5 text-[10px] font-bold text-[#E87518]">
          <HelpCircle size={10} /> Unclear
        </span>
      );
  }
}

export function AllCommentsList({
  comments,
  isAnalysisReady = false
}: {
  comments: CommentItem[];
  isAnalysisReady?: boolean;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [sortOrder, setSortOrder] = useState<"top" | "newest" | "oldest">("top");
  const [visibleCount, setVisibleCount] = useState<number>(10);
  const [expandedAnalysisIds, setExpandedAnalysisIds] = useState<Record<string, boolean>>({});
  const [expandedTextIds, setExpandedTextIds] = useState<Record<string, boolean>>({});

  const sortedComments = useMemo(() => {
    const list = [...comments];
    if (sortOrder === "top") {
      list.sort((a, b) => b.rankingScore - a.rankingScore);
    } else if (sortOrder === "newest") {
      list.sort((a, b) => b.id.localeCompare(a.id));
    } else {
      list.sort((a, b) => a.id.localeCompare(b.id));
    }
    return list;
  }, [comments, sortOrder]);

  const displayedList = sortedComments.slice(0, visibleCount);

  const toggleAnalysis = (id: string) => {
    setExpandedAnalysisIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const toggleText = (id: string) => {
    setExpandedTextIds((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <section className="rounded-xl border border-[#e5e7eb] bg-white p-5 sm:p-6 shadow-2xs space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#f1f3f5]">
        <button
          type="button"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="flex items-center gap-2 group text-left cursor-pointer"
          aria-expanded={!isCollapsed}
          aria-label="Toggle All Comments"
        >
          <MessageSquare size={18} className="text-[#143454]" />
          <h2 className="font-display text-[18px] sm:text-[20px] font-bold text-[#171717] group-hover:text-[#143454] transition-colors">
            All Comments ({comments.length})
          </h2>
          <span className="p-1 rounded-md text-[#666666] group-hover:text-[#143454] group-hover:bg-[#f1f3f5] transition-colors">
            {isCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
          </span>
        </button>

        {!isCollapsed && comments.length > 0 && (
          <div className="flex items-center gap-2 text-xs">
            <ArrowUpDown size={13} className="text-[#666666]" />
            <span className="text-[#666666] font-medium">Sort by:</span>
            <div className="inline-flex rounded-lg border border-[#e5e7eb] bg-[#f9fafb] p-0.5">
              <button
                type="button"
                onClick={() => setSortOrder("top")}
                className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer ${
                  sortOrder === "top" ? "bg-[#143454] text-white shadow-xs" : "text-[#666666] hover:text-[#171717]"
                }`}
              >Top comments</button>
              <button
                type="button"
                onClick={() => setSortOrder("newest")}
                className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer ${
                  sortOrder === "newest" ? "bg-[#143454] text-white shadow-xs" : "text-[#666666] hover:text-[#171717]"
                }`}
              >Newest first</button>
              <button
                type="button"
                onClick={() => setSortOrder("oldest")}
                className={`px-2.5 py-1 rounded text-xs font-semibold cursor-pointer ${
                  sortOrder === "oldest" ? "bg-[#143454] text-white shadow-xs" : "text-[#666666] hover:text-[#171717]"
                }`}
              >Oldest first</button>
            </div>
          </div>
        )}
      </div>

      {!isCollapsed && (
        <>
          {comments.length === 0 ? (
            <div className="py-6 text-center text-[13px] sm:text-[14px] text-[#666666]">
              No preview comments have been added in this browser session yet.
            </div>
          ) : (
            <div className="divide-y divide-[#f1f3f5]">
              {displayedList.map((item) => {
                const isTextLong = item.originalText.length > 200;
                const isTextExpanded = expandedTextIds[item.id];
                const isAnalysisExpanded = expandedAnalysisIds[item.id];
                const isSessionPreview = item.id.startsWith("session-");

                return (
                  <div key={item.id} className="py-4 first:pt-0 last:pb-0 space-y-2.5">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-[13px]">
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-full bg-[#e8eef5] flex items-center justify-center text-[#143454] shrink-0 font-bold text-[13px]">
                          <User size={15} />
                        </div>
                        <div>
                          <span className="font-semibold text-[#171717] block leading-tight text-[13px] sm:text-[14px]">{item.stakeholder}</span>
                          <span className="text-[11px] text-[#9ca3af]">ID: {item.id} • {item.timestamp}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {isAnalysisReady && getStanceBadge(item.stance)}
                        {isSessionPreview && (
                          <span className="text-[11px] text-[#475569] bg-[#f1f5f9] px-2 py-0.5 rounded">
                            Session preview · not saved
                          </span>
                        )}
                        {item.policyTopic && isAnalysisReady && (
                          <span className="text-[11px] text-[#666666] bg-[#f8fafc] px-2 py-0.5 rounded border border-[#e5e7eb]">
                            {item.policyTopic.replace("_", " ")}
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="text-[14px] sm:text-[15px] leading-relaxed text-[#334155] pl-10.5">
                      <p>{isTextLong && !isTextExpanded ? `${item.originalText.slice(0, 200)}...` : item.originalText}</p>
                      {isTextLong && (
                        <button type="button" onClick={() => toggleText(item.id)} className="mt-1 text-[13px] font-semibold text-[#143454] hover:text-[#E87518] cursor-pointer">
                          {isTextExpanded ? "Show less" : "Read more"}
                        </button>
                      )}
                    </div>

                    <div className="pl-10.5 flex items-center gap-4 text-[13px] pt-1">
                      {isAnalysisReady && (
                        <button type="button" onClick={() => toggleAnalysis(item.id)} className="inline-flex items-center gap-1 font-semibold text-[#143454] hover:text-[#E87518] transition-colors cursor-pointer">
                          <Bot size={14} className="text-[#E87518]" />
                          <span>{isAnalysisExpanded ? "Hide AI Analysis" : "View AI Analysis"}</span>
                          {isAnalysisExpanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </button>
                      )}
                      {item.sourceUrl ? (
                        <a href={item.sourceUrl} target="_blank" rel="noopener noreferrer" className="text-[#9ca3af] hover:text-[#666666] inline-flex items-center gap-1 text-[12px]">
                          <span>Source Record</span><ExternalLink size={10} />
                        </a>
                      ) : (
                        <span className="text-[12px] text-[#9ca3af]">No external source · frontend preview</span>
                      )}
                    </div>

                    {isAnalysisReady && isAnalysisExpanded && (
                      <div className="ml-10.5 mt-2 rounded-lg border border-[#e2e8f0] bg-[#f8fafc] p-3.5 text-[13px] space-y-2.5 animate-fadeIn">
                        <div className="flex items-center justify-between text-[12px] font-semibold text-[#143454] border-b border-[#e2e8f0] pb-1.5">
                          <span className="flex items-center gap-1.5"><Bot size={14} className="text-[#E87518]" />Model Analysis &amp; Attribution Breakdown</span>
                          {item.confidenceScore && (
                            <span className="text-[#16834A] bg-[#e6f4ea] px-1.5 py-0.5 rounded">
                              Confidence: {Math.round(item.confidenceScore * 100)}%
                            </span>
                          )}
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[13px]">
                          <div><strong className="text-[11px] uppercase text-[#666666] block">Main Grievance / Concern:</strong><p className="text-[#171717]">{item.mainArgument}</p></div>
                          <div><strong className="text-[11px] uppercase text-[#666666] block">Action Requested:</strong><p className="text-[#171717]">{item.suggestedAction || "None specified"}</p></div>
                          <div className="sm:col-span-2"><strong className="text-[11px] uppercase text-[#666666] block">Attribution &amp; Safeguard Decision:</strong><span className="inline-block mt-0.5 text-[12px] font-mono bg-white px-2 py-0.5 rounded border border-[#e2e8f0] text-[#143454]">{item.demandStatus} • {item.attributionTier}</span></div>
                          <div className="sm:col-span-2"><strong className="text-[11px] uppercase text-[#666666] block">Evidence &amp; Reasoning:</strong><p className="text-[#555555]">{item.evidenceReasoning}</p></div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {visibleCount < comments.length && (
            <div className="pt-4 border-t border-[#f1f3f5] text-center">
              <button type="button" onClick={() => setVisibleCount((prev) => Math.min(prev + 10, comments.length))} className="inline-flex items-center gap-2 rounded-lg bg-[#143454] px-5 py-2.5 text-[13px] font-semibold text-white shadow-xs hover:bg-[#0e2a47] transition-colors cursor-pointer">
                <span>Load More Comments ({comments.length - visibleCount} remaining)</span><ChevronDown size={14} />
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}
