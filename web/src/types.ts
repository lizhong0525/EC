export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  text: string;
  pending?: boolean;
}

export interface RelationshipState {
  trust: number;
  dependency: number;
  offline_help_willingness: number;
}

export interface SessionState {
  session_id: string;
  phase: string;
  risk_level: string;
  relationship: RelationshipState;
  handoff_status: string;
  memory_enabled: boolean;
  proactive_enabled: boolean;
  turn_index: number;
  updated_at: string;
}

export interface SessionView {
  session_id: string;
  state: SessionState;
  model: string | null;
  research_mode: boolean;
  participant_type: "synthetic_case" | "adult_research_tester";
  synthetic_case_id: string | null;
}

export interface ModelSnapshot {
  source: "snapshot" | "runtime" | "unavailable";
  active_model: string | null;
  detail?: string | null;
  provider_location: "local" | "remote" | "unavailable";
}

export interface ModelCallUsage {
  model: string;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  estimated_cost_cny: number;
  retry_index: number;
  cache_hit: boolean;
}

export interface DecisionView {
  primary_action: string;
  escalation_level: string;
  escalation_target: string;
  risk_estimate: {
    depression_worsening: number;
    immediate_safety_risk: number;
    dependency_risk: number;
  };
  uncertainty: number;
  evidence_codes: string[];
  reason_codes: string[];
  safety_passed: boolean;
  used_fallback: boolean;
  audit_codes: string[];
  model_calls: ModelCallUsage[];
  policy_version: string;
  prompt_version: string | null;
}

export interface TurnResult {
  session: SessionView;
  response: string;
  decision: DecisionView;
}

export interface ProactiveMessage {
  prompt_id: string;
  text: string;
  sent_at: string;
  used_fallback: boolean;
}

export interface StoredMessage {
  message_id: string;
  ordinal: number;
  role: "USER" | "COMPANION" | "STUDENT_SIMULATOR" | "SYSTEM";
  content: string;
  simulated_time: string;
  created_at: string;
}

export interface MemoryRecord {
  memory_id: string;
  session_id: string;
  kind: string;
  content: string;
  confidence: number;
  created_at: string;
  expires_at: string | null;
  user_authorized: boolean;
}

export interface TraceRecord {
  trace_id: string;
  step_id: number;
  selected_action: string;
  response_text: string;
  safety_audit: {
    passed: boolean;
    violation_codes: string[];
    used_fallback: boolean;
  };
  model_usage?: ModelCallUsage[];
}

export interface BenchmarkRun {
  schema_version: "1.0";
  run_id: string;
  directory_name: string;
  track: "A" | "B" | "C";
  created_at: string;
  status: "in_progress" | "complete";
  completed_unit_count: number;
  planned_unit_count: number;
  methods: string[];
  report_available: boolean;
}

export interface BenchmarkRunCatalog {
  schema_version: "1.0";
  runs: BenchmarkRun[];
  ignored_directory_names: string[];
}

export interface BenchmarkReportDocument {
  schema_version: string;
  track?: "B" | "C";
  run_id: string | null;
  completed_unit_count: number;
  planned_unit_count: number | null;
  complete: boolean;
  rows: Array<Record<string, unknown>>;
}

export interface BenchmarkReport {
  schema_version: "1.0";
  run: BenchmarkRun;
  report: BenchmarkReportDocument;
}

export interface BenchmarkStartRequest {
  track: "A" | "B" | "C";
  case_limit: number;
  policies: string[];
  split: "dev" | "public_test";
}

export interface BenchmarkJob {
  job_id: string;
  run_id: string;
  status: "queued" | "running" | "reporting" | "complete" | "failed" | "interrupted";
  configuration: BenchmarkStartRequest;
  execution: "local_rules";
  model_calls: 0;
  error: string | null;
  run: BenchmarkRun | null;
}
