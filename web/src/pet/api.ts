export interface PetSettings {
  pet_name: string;
  personality_prompt: string;
  deepseek_base_url: "https://api.deepseek.com" | "https://api.deepseek.com/v1";
  deepseek_model: "deepseek-flash" | "deepseek-v4-pro" | "deepseek-chat" | "deepseek-reasoner";
  deepseek_thinking: boolean;
  temperature: number;
  max_tokens: number;
  context_turns: number;
  timeout_seconds: number;
  cloud_consent: boolean;
  qwen_enabled: boolean;
  qwen_base_url: string;
  qwen_model: string;
  assessment_timeout_seconds: number;
  jev_enabled: boolean;
  jev_model: string;
  jev_confidence: number;
  pet_scale: number;
  animation_enabled: boolean;
  always_on_top: boolean;
}

export interface SettingsView {
  current_session_id?: string | null;
  settings: PetSettings;
  deepseek_key_configured: boolean;
  jev_key_configured: boolean;
  credential_storage: string;
  warning: string | null;
}

export interface MessagingSettings {
  qq: { enabled: boolean; app_id: string; client_secret?: string; intents: number };
  wechat: { enabled: boolean; token?: string; app_id: string; app_secret?: string; allow_ip_cidr: string };
  personal_wechat: { enabled: boolean; bridge_token?: string; account_wxid: string; allowed_chats: string[]; allow_groups: boolean; test_account_confirmed: boolean };
  callback_host: string;
  callback_port: number;
  per_user_per_minute: number;
  max_inflight: number;
}

export interface MessagingSettingsView {
  settings: MessagingSettings;
  qq_secret_configured: boolean;
  wechat_token_configured: boolean;
  wechat_secret_configured: boolean;
  personal_wechat_token_configured: boolean;
  credential_storage: string;
  deepseek_key_configured: boolean;
  cloud_consent: boolean;
}

export interface Stage {
  name: string;
  state: string;
  detail: string;
}

export interface StoredMessage {
  message_id: string;
  role: string;
  content: string;
}

export interface SessionActivation {
  session_id: string;
  messages: StoredMessage[];
  draft?: string;
  attention: boolean;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface SessionPage {
  items: SessionSummary[];
  total: number;
  has_more: boolean;
}

export interface PetTurn {
  result: { response: string; decision: { used_fallback: boolean } };
  stages: Stage[];
  mode: "safety_fallback" | "deepseek" | "recovered";
  attention: boolean;
  recovered?: boolean;
}

let token = "";

export async function request<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  const response = await fetch(`/api/pet${path}`, {
    method,
    headers: { "Content-Type": "application/json", "X-Pet-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = typeof data.detail === "string" ? data.detail : "参数无效或请求失败，请检查设置。";
    throw new Error(detail);
  }
  return (response.status === 204 ? undefined : await response.json()) as T;
}

export async function bootstrap(): Promise<SettingsView> {
  const data = await request<SettingsView & { token: string }>("/bootstrap", undefined, "GET");
  token = data.token;
  return data;
}

export interface DesktopBridge {
  toggleChat(): void;
  hideChat(): void;
  openSettings(): void;
  openGames(): void;
  openReport(reportId: string): void;
  movePet(dx: number, dy: number, screenX: number, screenY: number): void;
  drag(active: boolean, screenX?: number, screenY?: number): void;
  configure(settings: Pick<PetSettings, "pet_scale" | "always_on_top">): void;
  activity(state: string): void;
  quit(): void;
  onActivity(callback: (state: string) => void): () => void;
  onSettings(callback: () => void): () => void;
}

declare global {
  interface Window {
    petDesktop?: DesktopBridge;
  }
}
