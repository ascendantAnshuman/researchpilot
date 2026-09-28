/**
 * TypeScript mirrors of the FastAPI response models (backend/app/schemas.py).
 * Keep these in sync when the backend schemas change.
 */

export type SessionStatus = "pending" | "running" | "completed" | "failed";
export type SupportLabel = "supported" | "partial" | "unsupported";
export type SectionTheme = "methods" | "datasets" | "results" | "open_problems" | "other";
export type ClaimAction = "kept" | "removed" | "flagged";
export type ExportFormat = "markdown" | "pdf";

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
}

export interface Paper {
  paper_id: string;
  sources: string[];
  title: string;
  authors: string[];
  year: number | null;
  published: string | null;
  venue: string | null;
  abstract: string;
  url: string | null;
  pdf_url: string | null;
  doi: string | null;
  arxiv_id: string | null;
  s2_id: string | null;
  citation_count: number;
  source_rank: number;
  relevance: number;
  score: number;
}

export interface PaperSummary {
  problem: string;
  method: string;
  datasets: string[];
  metrics: string[];
  key_results: string[];
  limitations: string[];
}

export interface PaperDigest {
  paper: Paper;
  summary: PaperSummary | null;
  full_text: boolean;
  chunk_count: number;
  note: string | null;
}

export interface Reference {
  number: number;
  paper_id: string;
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  url: string | null;
}

export interface ReportSection {
  heading: string;
  theme: SectionTheme;
  paragraphs: string[];
}

export interface ComparisonRow {
  reference: number;
  title: string;
  year: number | null;
  method: string;
  dataset: string;
  key_result: string;
}

export interface EvidenceSnippet {
  chunk_id: string;
  paper_id: string;
  title: string;
  section: string;
  page: number;
  text: string;
}

export interface VerifiedClaim {
  claim_id: string;
  text: string;
  section: string;
  sentence: string;
  label: SupportLabel;
  rationale: string;
  evidence: EvidenceSnippet[];
  action: ClaimAction;
}

export interface StageMetrics {
  seconds: number;
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
}

export interface RunMetrics {
  llm_model: string;
  critic_enabled: boolean;
  revisions: number;
  total_seconds: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  stages: Record<string, StageMetrics>;
  papers_found: number;
  papers_full_text: number;
  claims_total: number;
  claims_supported: number;
  claims_partial: number;
  claims_unsupported: number;
  faithfulness: number | null;
}

export interface ReportSentence {
  section: string;
  text: string;
  chunk_ids: string[];
}

export interface ResearchReport {
  question: string;
  title: string;
  executive_summary: string;
  sections: ReportSection[];
  comparison_table: ComparisonRow[];
  references: Reference[];
  papers: PaperDigest[];
  verification: VerifiedClaim[];
  sentences: ReportSentence[];
  markdown: string;
  metrics: RunMetrics;
  warnings: string[];
}

export interface SessionSummary {
  id: string;
  question: string;
  title: string | null;
  status: SessionStatus;
  current_stage: string | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  metrics: RunMetrics | null;
}

export interface SessionDetail extends SessionSummary {
  report: ResearchReport | null;
}

export interface ResearchStarted {
  session: SessionSummary;
  events_url: string;
}

export interface ResearchOptions {
  question: string;
  enable_critic?: boolean;
  top_n?: number;
}

export interface ChatCitation {
  number: number;
  paper_id: string;
  title: string;
  chunk_id: string;
  section: string;
  page: number;
  snippet: string;
}

export interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: ChatCitation[];
  created_at: string;
}

export interface ChatExchange {
  question: ChatMessage;
  answer: ChatMessage;
}

export type ProgressStatus = "started" | "progress" | "completed" | "warning" | "error";

export interface ProgressEvent {
  stage: string;
  status: ProgressStatus;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
}

/** Messages on the research SSE stream (see backend/app/routes/research.py). */
export type StreamMessage =
  | { event: "progress"; data: ProgressEvent }
  | { event: "queued" | "running"; data: { message: string; timestamp: string } }
  | {
      event: "done";
      data: {
        session_id: string;
        status: "completed";
        title: string | null;
        faithfulness?: number | null;
        cost_usd?: number;
        timestamp: string;
      };
    }
  | { event: "error"; data: { session_id: string; message: string; timestamp: string } };
