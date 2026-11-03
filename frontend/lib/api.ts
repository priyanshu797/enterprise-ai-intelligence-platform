// Typed API client for the Enterprise AI Intelligence Platform backend.
// Reads the JWT from localStorage and attaches it as a Bearer token.

// In production set NEXT_PUBLIC_API_URL to the deployed backend (e.g.
// https://api.example.com). In dev, the Next.js dev server rewrites /api/* to
// the backend, so a relative base works.
const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "") + "/api/v1";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string>),
  };
  if (init.body && !(init.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (token) headers["Authorization"] = `Bearer ${token}`;

  // Use a longer timeout for agent calls (they can take 60-90s with Gemini)
  const timeout = path.includes("/agent") ? 180000 : 45000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
  } catch (err: any) {
    clearTimeout(timer);
    if (err?.name === "AbortError") {
      throw new ApiError(408, "Request timed out. The agent is taking too long — try a simpler query.");
    }
    throw new ApiError(0, err?.message || "Network error — is the backend running?");
  }
  clearTimeout(timer);

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      detail = await res.text().catch(() => res.statusText);
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return null as T;
  return res.json();
}

// ---------------- Types ----------------
export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface Token {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface Doc {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  title: string;
  status: string;
  error_message: string;
  num_chunks: number;
  content_hash: string;
  created_at: string;
  processed_at: string | null;
}

export interface SourceRef {
  document_id: string;
  document_filename: string;
  chunk_index: number;
  text: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceRef[];
  conversation_id: string;
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  cached: boolean;
  confidence: number;
  answer_status: "answered" | "insufficient" | "conflicting" | "unsupported";
}

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  role: string;
  content: string;
  sources: any[];
  provider: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  created_at: string;
}

export interface AgentStep {
  step_index: number;
  tool_name: string;
  tool_input: Record<string, any>;
  tool_output: string;
  success: boolean;
  error_message: string;
  latency_ms: number;
}

export interface AgentRun {
  id: string;
  query: string;
  status: string;
  final_answer: string;
  total_tokens: number;
  latency_ms: number;
  error_message: string;
  created_at: string;
  completed_at: string | null;
  steps: AgentStep[];
}

export interface DashboardStats {
  total_documents: number;
  total_chunks: number;
  total_conversations: number;
  total_messages: number;
  total_agent_runs: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_cost_usd: number;
  cache_hits: number;
  avg_latency_ms: number;
  documents_by_status: Record<string, number>;
  tokens_by_provider: Record<string, number>;
  recent_activity: any[];
}

// ---------------- API calls ----------------
export const api = {
  // Auth
  register: (email: string, password: string, full_name: string) =>
    apiFetch<User>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, full_name }),
    }),
  login: (email: string, password: string) =>
    apiFetch<Token>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
  me: () => apiFetch<User>("/auth/me"),

  // Documents
  listDocuments: (status?: string) =>
    apiFetch<Doc[]>(`/documents${status ? `?status_filter=${status}` : ""}`),
  uploadDocument: (file: File, title?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return apiFetch<Doc>("/documents", { method: "POST", body: form });
  },
  getChunks: (id: string) => apiFetch<any[]>(`/documents/${id}/chunks`),
  getJob: (id: string) => apiFetch<any>(`/documents/${id}/job`),
  deleteDocument: (id: string) =>
    apiFetch<void>(`/documents/${id}`, { method: "DELETE" }),

  // Chat
  chat: (query: string, conversation_id?: string, filters?: Record<string, any>) =>
    apiFetch<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ query, conversation_id, filters }),
    }),

  // Conversations
  listConversations: () => apiFetch<Conversation[]>("/conversations"),
  getMessages: (id: string) => apiFetch<Message[]>(`/conversations/${id}`),
  deleteConversation: (id: string) =>
    apiFetch<void>(`/conversations/${id}`, { method: "DELETE" }),

  // Agent
  runAgent: (query: string, conversation_id?: string) =>
    apiFetch<AgentRun>("/agent/run", {
      method: "POST",
      body: JSON.stringify({ query, conversation_id }),
    }),
  listRuns: (limit = 20) => apiFetch<AgentRun[]>(`/agent/runs?limit=${limit}`),
  getRun: (id: string) => apiFetch<AgentRun>(`/agent/runs/${id}`),

  // Stats
  dashboard: () => apiFetch<DashboardStats>("/stats/dashboard"),
  usage: (limit = 50) => apiFetch<any[]>(`/stats/usage?limit=${limit}`),
  activity: (limit = 50) => apiFetch<any[]>(`/stats/activity?limit=${limit}`),
  health: () => apiFetch<any>("/stats/health"),
};
