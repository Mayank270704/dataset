"use client";

import React, { useState, useEffect } from "react";
import { AuthenticatedCommentForm } from "./authenticated-comment-form";
import { ShareOpinionForm } from "./share-opinion-form";
import { AllCommentsList } from "./all-comments-list";
import type { ConsultationStatus } from "../feed/types";
import type { CommentItem } from "./types";
import { Loader2, MessageSquare } from "lucide-react";

interface PublicOpinionSectionProps {
  policySlug: string;
  policyTitle: string;
  policyId?: string;
  openedAt?: string;
  commentingDeadline?: string;
  initialConsultationStatus?: ConsultationStatus;
}

interface ApiComment {
  id: string;
  policySlug: string;
  stakeholder: string;
  originalText: string;
  status: string;
  createdAt: string;
  authorName: string;
}

/**
 * PublicOpinionSection combines:
 * 1. AuthenticatedCommentForm — real backend, authenticated, moderated comments.
 * 2. ShareOpinionForm — session-only inference demo (shown only when window is open).
 * 3. A list of approved comments fetched from the real backend.
 */
export function PublicOpinionSection({
  policySlug,
  policyTitle,
  policyId,
  openedAt,
  commentingDeadline,
}: PublicOpinionSectionProps) {
  // Session-only preview comments (for the inference demo)
  const [sessionComments, setSessionComments] = useState<CommentItem[]>([]);

  // Real approved comments from the backend
  const [realComments, setRealComments] = useState<CommentItem[]>([]);
  const [commentsLoading, setCommentsLoading] = useState(true);

  // Time limit removed for now: commenting is open at anytime for all policies
  const isWindowOpen = true;

  // Fetch real approved comments
  useEffect(() => {
    let cancelled = false;
    async function load() {
      setCommentsLoading(true);
      try {
        const res = await fetch(
          `/api/policies/${encodeURIComponent(policySlug)}/comments`,
          { cache: "no-store" }
        );
        if (!res.ok || cancelled) return;
        const data = await res.json();
        const mapped: CommentItem[] = (data.comments || []).map(
          (c: ApiComment, idx: number) => ({
            id: c.id,
            originalText: c.originalText,
            stance: "UNCLEAR" as const,
            policyTopic: c.status,
            mainArgument: "",
            suggestedAction: "",
            evidenceReasoning: "",
            whyItMatters: "",
            demandStatus: "UNCLEAR_OR_INSUFFICIENT" as const,
            attributionTier: c.stakeholder ? "ORGANIZATION" : ("NONE_UNSTATED" as const),
            stakeholder: c.stakeholder,
            sourceDocument: `Submitted by ${c.authorName}`,
            sourceUrl: "",
            timestamp: new Date(c.createdAt).toLocaleString("en-IN", {
              day: "numeric",
              month: "short",
              year: "numeric",
            }),
            rankingScore: 0,
            rank: idx + 1,
            isTopInformative: false,
          })
        );
        if (!cancelled) setRealComments(mapped);
      } catch {
        // non-critical
      } finally {
        if (!cancelled) setCommentsLoading(false);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [policySlug]);

  const handleNewSessionComment = (newSub: {
    id: string;
    text: string;
    stakeholder: string;
  }) => {
    const previewComment: CommentItem = {
      id: newSub.id,
      originalText: newSub.text,
      stance: "UNCLEAR",
      policyTopic: "PENDING_AI_ANALYSIS",
      mainArgument: "",
      suggestedAction: "",
      evidenceReasoning: "",
      whyItMatters: "",
      demandStatus: "UNCLEAR_OR_INSUFFICIENT",
      attributionTier: "NONE_UNSTATED",
      stakeholder: newSub.stakeholder,
      sourceDocument: "Temporary browser-session preview",
      sourceUrl: "",
      timestamp: "Just now",
      rankingScore: 0,
      rank: sessionComments.length + 1,
      isTopInformative: false,
    };
    setSessionComments((prev) => [previewComment, ...prev]);
  };

  const allDisplayedComments = [...sessionComments, ...realComments];

  return (
    <section className="mt-8 space-y-6">
      {/* 1. Real authenticated comment form */}
      <AuthenticatedCommentForm policySlug={policySlug} policyTitle={policyTitle} />

      {/* 2. Inference demo (session-only) — shown only when window is open */}
      {isWindowOpen && (
        <div className="space-y-2">
          <p className="text-[11px] font-semibold uppercase tracking-widest text-[#6b7280] px-1">
            Research / Inference Preview
          </p>
          <ShareOpinionForm
            policySlug={policySlug}
            policyTitle={policyTitle}
            policyId={policyId}
            openedAt={openedAt}
            commentingDeadline={commentingDeadline}
            onSubmissionSuccess={handleNewSessionComment}
          />
        </div>
      )}

      {/* 3. Real approved comments from backend */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <MessageSquare size={15} className="text-[#143454]" />
          <h3 className="font-display text-[16px] font-bold text-[#171717]">
            Community Comments
          </h3>
          {commentsLoading && (
            <Loader2 size={13} className="animate-spin text-[#6b7280]" />
          )}
        </div>

        {commentsLoading ? (
          <p className="text-[13px] text-[#6b7280]">Loading comments…</p>
        ) : realComments.length === 0 && sessionComments.length === 0 ? (
          <p className="text-[13px] text-[#6b7280] py-4 text-center">
            No published comments yet. Be the first to share your opinion.
          </p>
        ) : (
          <AllCommentsList comments={allDisplayedComments} isAnalysisReady={false} />
        )}
      </div>
    </section>
  );
}
