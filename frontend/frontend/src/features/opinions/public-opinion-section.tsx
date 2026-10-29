"use client";

import React, { useState } from "react";
import { ShareOpinionForm } from "./share-opinion-form";
import { AllCommentsList } from "./all-comments-list";
import type { ConsultationStatus } from "../feed/types";
import type { CommentItem } from "./types";

interface PublicOpinionSectionProps {
  policySlug: string;
  policyTitle: string;
  policyId?: string;
  openedAt?: string;
  commentingDeadline?: string;
  initialConsultationStatus?: ConsultationStatus;
}

export function PublicOpinionSection({
  policySlug,
  policyTitle,
  policyId,
  openedAt,
  commentingDeadline
}: PublicOpinionSectionProps) {
  // Only temporary comments entered in this browser session are shown in this
  // frontend-only phase. Real analysed comments are hidden until backend integration.
  const [sessionComments, setSessionComments] = useState<CommentItem[]>([]);

  const handleNewComment = (newSub: { id: string; text: string; stakeholder: string }) => {
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
      isTopInformative: false
    };
    setSessionComments((prev) => [previewComment, ...prev]);
  };

  return (
    <section className="mt-8 space-y-6">
      <ShareOpinionForm
        policySlug={policySlug}
        policyTitle={policyTitle}
        policyId={policyId}
        openedAt={openedAt}
        commentingDeadline={commentingDeadline}
        onSubmissionSuccess={handleNewComment}
      />
      <AllCommentsList comments={sessionComments} isAnalysisReady={false} />
    </section>
  );
}
