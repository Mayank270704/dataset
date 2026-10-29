export type StanceType = "SUPPORT" | "OPPOSE" | "NEUTRAL" | "UNCLEAR";

export type DemandStatus =
  | "EXPLICIT_DEMAND"
  | "IMPLICIT_DEMAND"
  | "CONCERN_ONLY"
  | "UNCLEAR_OR_INSUFFICIENT";

export type AttributionTier =
  | "RESPONDENT_EXPLICIT_DEMAND"
  | "MODEL_INFERRED_RECOMMENDATION"
  | "ANALYST_PROPOSED_ACTION"
  | "NONE_UNSTATED";

export interface CommentItem {
  id: string;
  originalText: string;
  stance: StanceType;
  policyTopic: string;
  mainArgument: string;
  suggestedAction: string;
  evidenceReasoning: string;
  whyItMatters: string;
  confidenceScore?: number;
  demandStatus: DemandStatus;
  attributionTier: AttributionTier;
  stakeholder: string;
  sourceDocument: string;
  sourceUrl: string;
  timestamp: string;
  rankingScore: number;
  rank: number;
  isTopInformative: boolean;
}

export interface OpinionSummary {
  totalAnalyzed: number;
  supportCount: number;
  supportPercent: number;
  opposeCount: number;
  opposePercent: number;
  neutralCount: number;
  neutralPercent: number;
  unclearCount: number;
  unclearPercent: number;
  datasetSource: string;
  sourceUrl: string;
  disclaimer: string;
}

export interface ImprovementSuggestion {
  category: string;
  concernIdentified: string;
  actionableSuggestion: string;
  supportingCommentsCount: number;
}
