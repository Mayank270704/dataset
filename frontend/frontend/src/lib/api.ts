export type Envelope<T> = {
  component: string;
  kind?: string;
  scope?: string;
  caveats?: string[];
  human_validated?: boolean;
  data: T;
  total?: number;
  offset?: number;
  limit?: number;
};
export type ResearchStatus = {
  all_verified: boolean;
  scope: string;
  human_validated: false;
  components: Record<string, { component: string; status: string; problems: string[] }>;
};
export type Group = {
  cluster_id: number;
  tier: string;
  short_label: string | null;
  description: string;
  flags: string[];
  entries: number;
  unique_texts: number;
  independent_texts: number;
  language_counts: Record<string, number>;
};
export type Evidence = {
  corpus_id: number;
  source_dataset: string;
  source_record_id: string;
  text: string;
  cluster_id?: number;
  language?: string;
  state?: string;
  question_slug?: string;
};
export type DatasetPage = {
  component: string;
  scope: string;
  total: number;
  offset: number;
  limit: number;
  data: Evidence[];
};
export type Stats = {
  corpus_entries: number;
  concern_groups: number;
  ranked_groups: number;
  demand_candidates?: number;
  top_exploratory_group: { short_label: string; exploratory_score: number } | null;
  scope: string;
  human_validated: false;
};
export type DemandEvidence = {
  corpus_id: number;
  source_dataset: string;
  source_record_id: string;
  question_slug: string;
  language: string;
  quote: string;
  selection_reason: string;
};
export type DemandCandidate = {
  candidate_id: string;
  cluster_id: number;
  short_label: string;
  demand_text: string;
  entries: number;
  unique_texts: number;
  independent_texts: number;
  language_counts: Record<string, number>;
  source_datasets: string[];
  state: string;
  human_review_status: string;
  generation_method: string;
  quality_flags: string[];
  has_dissenting_evidence: boolean;
  insufficient_evidence: boolean;
  insufficient_reason: string;
  evidence_coverage: string;
  evidence?: DemandEvidence[];
  scope?: string;
  disclaimer?: string;
};
export type DemandsPage = Envelope<DemandCandidate[]> & { total: number; offset: number; limit: number };

const API_BASE = (process.env.NEXT_PUBLIC_POLICYLENS_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Research API returned ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function postApi<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
