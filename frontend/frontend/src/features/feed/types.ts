export type PolicyCategory =
  | "Environment"
  | "Public services"
  | "Economy"
  | "Health"
  | "Digital government"
  | "Education";

export type ConsultationStatus =
  | "open"
  | "closed_analysis_pending"
  | "closed_analysis_running"
  | "closed_analysis_completed"
  | "closed_analysis_failed";

export type PolicyPost = {
  id: string;
  slug: string;
  title: string;
  summary: string;
  category: PolicyCategory;
  ministry: string;
  publishedDate: string; // Verified release / launch date from official records
  sourceName: string;
  sourceUrl: string;
  statusLabel: string;
  verifiedOfficial: boolean;
  language: string;
  keyHighlights: string[];
  department?: string;
  schemeType?: string;
  openedAt?: string; // ISO 8601 opening timestamp for consultation
  commentingDeadline?: string; // ISO 8601 format (openedAt + 4 days)
  consultationStatus?: ConsultationStatus;
};
