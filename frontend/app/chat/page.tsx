"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import RequireAuth from "@/components/RequireAuth";
import { Badge } from "@/components/UI";
import { api, ChatResponse, Conversation, Message } from "@/lib/api";

export default function ChatPage() {
  return <RequireAuth><ChatView /></RequireAuth>;
}

// ── Markdown renderer ────────────────────────────────────────────────────────
// Maps markdown elements to styled HTML so responses look polished.
const MarkdownComponents = {
  p: ({ children }: any) => (
    <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>
  ),
  strong: ({ children }: any) => (
    <strong className="font-semibold text-white">{children}</strong>
  ),
  em: ({ children }: any) => (
    <em className="italic text-slate-300">{children}</em>
  ),
  ul: ({ children }: any) => (
    <ul className="my-2 space-y-1 pl-4">{children}</ul>
  ),
  ol: ({ children }: any) => (
    <ol className="my-2 space-y-1 pl-4 list-decimal">{children}</ol>
  ),
  li: ({ children }: any) => (
    <li className="ml-2 text-sm leading-relaxed list-disc list-inside marker:text-indigo-400">
      {children}
    </li>
  ),
  h1: ({ children }: any) => (
    <h1 className="text-base font-bold text-white mt-3 mb-1.5">{children}</h1>
  ),
  h2: ({ children }: any) => (
    <h2 className="text-sm font-bold text-white mt-3 mb-1">{children}</h2>
  ),
  h3: ({ children }: any) => (
    <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-1">{children}</h3>
  ),
  code: ({ inline, children }: any) =>
    inline ? (
      <code className="bg-slate-700/60 text-indigo-300 text-xs px-1.5 py-0.5 rounded font-mono">
        {children}
      </code>
    ) : (
      <pre className="bg-slate-900/60 rounded-xl p-3 my-2 overflow-x-auto">
        <code className="text-xs text-slate-300 font-mono">{children}</code>
      </pre>
    ),
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-2 border-indigo-500/50 pl-3 my-2 text-slate-400 italic">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="border-slate-700 my-3" />,
  a: ({ children, href }: any) => (
    <a href={href} className="text-indigo-400 hover:underline" target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

// Render text content — markdown for AI, plain for user
function MessageContent({ role, content }: { role: string; content: string }) {
  if (role === "user") {
    return <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>;
  }
  return (
    <div className="text-sm text-slate-200">
      <ReactMarkdown components={MarkdownComponents}>{content}</ReactMarkdown>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
function ChatView() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId]   = useState<string | null>(null);
  const [messages, setMessages]   = useState<Message[]>([]);
  const [input, setInput]         = useState("");
  const [busy, setBusy]           = useState(false);
  const [error, setError]         = useState("");
  const [lastResp, setLastResp]   = useState<ChatResponse | null>(null);
  const endRef   = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const loadConvs = () =>
    api.listConversations().then(setConversations).catch(() => {});

  useEffect(() => { loadConvs(); }, []);
  useEffect(() => {
    if (!activeId) { setMessages([]); return; }
    api.getMessages(activeId).then(setMessages).catch(() => setMessages([]));
  }, [activeId]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const send = async () => {
    if (!input.trim() || busy) return;
    const q = input.trim();
    setInput(""); setBusy(true); setError("");
    const tempId = `temp-${Date.now()}`;
    setMessages(m => [...m, {
      id: tempId, role: "user", content: q,
      sources: [], provider: "", model: "",
      prompt_tokens: 0, completion_tokens: 0, latency_ms: 0,
      created_at: new Date().toISOString(),
    }]);
    try {
      const resp = await api.chat(q, activeId ?? undefined);
      setLastResp(resp);
      setActiveId(resp.conversation_id);
      loadConvs();
      api.getMessages(resp.conversation_id).then(setMessages);
    } catch (err: any) {
      setError(err.message);
      setMessages(m => m.filter(x => x.id !== tempId));
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const newConversation = () => {
    setActiveId(null);
    setMessages([]);
    setLastResp(null);
    setError("");
  };

  return (
    <div className="flex h-screen overflow-hidden">

      {/* ── Conversation sidebar ── */}
      <div className="w-60 flex-shrink-0 bg-slate-900/80 border-r border-slate-800 flex flex-col">
        <div className="flex items-center justify-between px-4 py-4 border-b border-slate-800">
          <span className="text-sm font-semibold text-white">Conversations</span>
          <button onClick={newConversation}
            className="text-xs text-indigo-400 hover:text-indigo-300 bg-indigo-500/10 hover:bg-indigo-500/20 rounded-lg px-2 py-1 transition-all">
            + New
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 py-2 space-y-0.5">
          {conversations.length === 0 && (
            <p className="text-xs text-slate-500 px-3 py-6 text-center">
              No conversations yet.<br />Ask your first question!
            </p>
          )}
          {conversations.map(c => (
            <button key={c.id} onClick={() => setActiveId(c.id)}
              className={`w-full text-left rounded-xl px-3 py-2.5 transition-all group ${
                activeId === c.id
                  ? "bg-indigo-500/20 ring-1 ring-indigo-500/30 text-white"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}>
              <p className="text-sm truncate font-medium">{c.title}</p>
              <p className="text-xs text-slate-600 mt-0.5">
                {new Date(c.updated_at).toLocaleDateString()}
              </p>
            </button>
          ))}
        </div>
      </div>

      {/* ── Main chat area ── */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0f1117]">

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          {messages.length === 0 && !busy && (
            <EmptyState onSuggest={(s) => { setInput(s); inputRef.current?.focus(); }} />
          )}

          {messages.map((m, i) => (
            <ChatMessage key={m.id + i} message={m} />
          ))}

          {/* Typing indicator */}
          {busy && <TypingIndicator />}

          <div ref={endRef} />
        </div>

        {/* ── Status bar ── */}
        {lastResp && (
          <div className="px-6 py-2 border-t border-slate-800/60 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
            <Badge status={lastResp.answer_status}>{lastResp.answer_status}</Badge>
            <span className="flex items-center gap-1">
              <span className="text-slate-600">Confidence</span>
              <span className="text-slate-300 font-medium">
                {(lastResp.confidence * 100).toFixed(0)}%
              </span>
            </span>
            <span className="text-slate-700">·</span>
            <span>{lastResp.sources.length} sources</span>
            <span className="text-slate-700">·</span>
            <span>{lastResp.latency_ms}ms</span>
            <span className="text-slate-700">·</span>
            <span>{lastResp.prompt_tokens + lastResp.completion_tokens} tokens</span>
            <span className="text-slate-700">·</span>
            <span className="text-indigo-400">{lastResp.provider}</span>
            {lastResp.cached && (
              <span className="text-emerald-400 flex items-center gap-1">
                ⚡ cached
              </span>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="px-6 py-2 border-t border-slate-800/60">
            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-xl px-3 py-2">
              <span className="mt-0.5 flex-shrink-0">⚠</span>
              <span>{error}</span>
            </div>
          </div>
        )}

        {/* ── Input ── */}
        <div className="p-4 border-t border-slate-800">
          <div className="flex gap-3 items-end bg-slate-800/60 border border-slate-700 rounded-2xl px-4 py-3
                          focus-within:border-indigo-500/60 focus-within:ring-2 focus-within:ring-indigo-500/10 transition-all">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder="Ask about your documents… (Enter to send, Shift+Enter for new line)"
              rows={1}
              className="flex-1 bg-transparent text-sm text-slate-200 placeholder-slate-500 outline-none resize-none max-h-40"
            />
            <button
              onClick={send}
              disabled={busy || !input.trim()}
              className="flex-shrink-0 w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center transition-all active:scale-95">
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-white">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            </button>
          </div>
          <p className="text-center text-xs text-slate-700 mt-2">
            Answers are grounded in your uploaded documents only
          </p>
        </div>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function EmptyState({ onSuggest }: { onSuggest: (s: string) => void }) {
  const suggestions = [
    "What are the main topics in my documents?",
    "Summarize the key findings.",
    "What are the most common issues mentioned?",
    "What recommendations are made?",
  ];
  return (
    <div className="flex flex-col items-center justify-center h-full py-16 text-center">
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-glow mb-5">
        <svg viewBox="0 0 24 24" fill="none" className="w-8 h-8 text-white">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-white mb-2">Ask about your documents</h2>
      <p className="text-sm text-slate-400 max-w-sm mb-8">
        Upload documents first, then ask anything. The AI retrieves relevant passages and cites sources.
      </p>
      <div className="grid grid-cols-1 gap-2 w-full max-w-xs">
        {suggestions.map(s => (
          <button key={s} onClick={() => onSuggest(s)}
            className="text-left text-sm text-slate-400 hover:text-white bg-slate-800/50 hover:bg-slate-700/60
                       border border-slate-700/50 hover:border-indigo-500/30 rounded-xl px-4 py-2.5 transition-all">
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function ChatMessage({ message: m }: { message: Message }) {
  const isUser = m.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"} items-start animate-slide-up`}>

      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold mt-0.5
                       ${isUser ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-300"}`}>
        {isUser ? "U" : "AI"}
      </div>

      {/* Bubble + sources */}
      <div className={`flex flex-col gap-2 ${isUser ? "items-end" : "items-start"} max-w-[80%]`}>

        {/* Content bubble */}
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-indigo-600 text-white rounded-tr-sm"
            : "bg-slate-800 text-slate-200 rounded-tl-sm border border-slate-700/40"
        }`}>
          <MessageContent role={m.role} content={m.content} />
        </div>

        {/* Source citations — below the bubble, not inside */}
        {!isUser && m.sources && m.sources.length > 0 && (
          <div className="w-full bg-slate-800/40 border border-slate-700/30 rounded-xl p-3 space-y-1.5">
            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Sources
            </p>
            {m.sources.map((s: any, idx: number) => (
              <div key={idx} className="flex items-center gap-2 text-xs">
                <span className="w-5 h-5 rounded-md bg-indigo-500/20 flex items-center justify-center
                                 text-indigo-400 font-bold text-[10px] flex-shrink-0">
                  {idx + 1}
                </span>
                <span className="text-slate-300 font-medium truncate flex-1">
                  {s.document_filename}
                </span>
                <span className="text-slate-600 text-[10px]">chunk {s.chunk_index}</span>
                <span className="text-slate-500 bg-slate-700/50 px-1.5 py-0.5 rounded text-[10px] font-medium">
                  {(s.score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
            {/* Meta row */}
            <div className="pt-1.5 mt-1 border-t border-slate-700/30 flex items-center gap-2 text-[10px] text-slate-600">
              <span>{m.provider}</span>
              {m.provider && <span>·</span>}
              <span>{m.prompt_tokens + m.completion_tokens} tokens</span>
              {m.latency_ms > 0 && <><span>·</span><span>{m.latency_ms}ms</span></>}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 items-start animate-slide-up">
      <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs text-slate-300 flex-shrink-0">
        AI
      </div>
      <div className="bg-slate-800 border border-slate-700/40 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
          <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
          <span className="w-2 h-2 rounded-full bg-indigo-400 typing-dot" />
        </div>
      </div>
    </div>
  );
}
