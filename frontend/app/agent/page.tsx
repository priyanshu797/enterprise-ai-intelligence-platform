"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import RequireAuth from "@/components/RequireAuth";
import { Badge, Card, PageHeader, Spinner } from "@/components/UI";
import { api, AgentRun } from "@/lib/api";

// ── Markdown components ───────────────────────────────────────────────────────
const MD: Record<string, any> = {
  p: ({ children }: any) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  strong: ({ children }: any) => <strong className="font-semibold text-white">{children}</strong>,
  em: ({ children }: any) => <em className="italic text-slate-300">{children}</em>,
  ul: ({ children }: any) => <ul className="my-2 space-y-1 pl-5 list-disc marker:text-indigo-400">{children}</ul>,
  ol: ({ children }: any) => <ol className="my-2 space-y-1 pl-5 list-decimal">{children}</ol>,
  li: ({ children }: any) => <li className="text-sm leading-relaxed">{children}</li>,
  h1: ({ children }: any) => <h1 className="text-base font-bold text-white mt-4 mb-2">{children}</h1>,
  h2: ({ children }: any) => <h2 className="text-sm font-bold text-white mt-3 mb-1.5">{children}</h2>,
  h3: ({ children }: any) => <h3 className="text-sm font-semibold text-slate-200 mt-2 mb-1">{children}</h3>,
  code: ({ inline, children }: any) =>
    inline
      ? <code className="bg-slate-700/60 text-indigo-300 text-xs px-1.5 py-0.5 rounded font-mono">{children}</code>
      : <pre className="bg-slate-900/60 rounded-xl p-3 my-2 overflow-x-auto"><code className="text-xs text-slate-300 font-mono">{children}</code></pre>,
  hr: () => <hr className="border-slate-700 my-3" />,
};

// ── Tool config ───────────────────────────────────────────────────────────────
// Each entry maps a backend tool_name to its display metadata.
// The agent auto-selects tools based on the question — see decision rules below.
const TOOLS: Record<string, { color: string; bg: string; icon: string; label: string; description: string }> = {
  knowledge_base_search: {
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    icon: "🔍",
    label: "Knowledge Base Search",
    description: "Searches your uploaded documents first",
  },
  web_search: {
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    icon: "🌐",
    label: "Web Search",
    description: "Searches the web when KB has no results or real-time data is needed",
  },
  analyze_document: {
    color: "text-purple-400",
    bg: "bg-purple-500/10",
    icon: "🔬",
    label: "Document Analysis",
    description: "Deep-dives into a specific document via LLM analysis",
  },
  generate_report: {
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
    icon: "📊",
    label: "Report Generation",
    description: "Synthesises gathered findings into a structured Markdown report",
  },
};

const EXAMPLES = [
  "Find the most common issues in my documents and suggest solutions.",
  "Search the web for the latest AI trends and generate a report.",
  "Analyze my uploaded documents and create a structured summary report.",
  "What key topics are covered across all my uploaded documents?",
];

// ── Page ──────────────────────────────────────────────────────────────────────
export default function AgentPage() {
  return <RequireAuth><AgentView /></RequireAuth>;
}

function AgentView() {
  const [query, setQuery]   = useState("");
  const [busy, setBusy]     = useState(false);
  const [run, setRun]       = useState<AgentRun | null>(null);
  const [error, setError]   = useState("");
  const [elapsed, setElapsed] = useState(0);

  // Show elapsed time while agent is running
  useEffect(() => {
    if (!busy) { setElapsed(0); return; }
    const t = setInterval(() => setElapsed(s => s + 1), 1000);
    return () => clearInterval(t);
  }, [busy]);

  const submit = async (q?: string) => {
    const question = (q ?? query).trim();
    if (!question || busy) return;
    setBusy(true); setError(""); setRun(null);
    try {
      const result = await api.runAgent(question);
      setRun(result);
      if (q) setQuery(q);
      else setQuery("");
    } catch (err: any) {
      setError(err.detail ?? err.message ?? "Unknown error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="p-6 max-w-5xl animate-fade-in">
      <PageHeader
        title="AI Agent"
        sub="4-tool smart agent — auto-selects between KB search, web search, document analysis, and report generation"
      />

      {/* ── Tool legend ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-5">
        {Object.entries(TOOLS).map(([key, tool]) => (
          <div key={key} className={`rounded-xl border border-slate-700/30 px-3 py-2.5 ${tool.bg}`}>
            <div className="flex items-center gap-1.5 mb-1">
              <span className="text-sm">{tool.icon}</span>
              <span className={`text-xs font-semibold ${tool.color}`}>{tool.label}</span>
            </div>
            <p className="text-[10px] text-slate-500 leading-relaxed">{tool.description}</p>
          </div>
        ))}
      </div>

      {/* ── Input card ── */}
      <Card className="p-5 mb-5">
        <textarea
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(); }}
          placeholder="Describe a task — e.g. 'Find the main failure causes and generate a report'"
          rows={3}
          disabled={busy}
          className="inp resize-none mb-4 disabled:opacity-60"
        />
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => submit()}
            disabled={busy || !query.trim()}
            className="btn-primary min-w-[130px]"
          >
            {busy ? (
              <><Spinner size={4} /> Running… {elapsed}s</>
            ) : (
              <>
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd"/>
                </svg>
                Run Agent
              </>
            )}
          </button>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map(ex => (
              <button
                key={ex}
                onClick={() => submit(ex)}
                disabled={busy}
                className="text-xs text-slate-500 hover:text-indigo-400 bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-1.5 transition-colors disabled:opacity-40"
              >
                {ex.slice(0, 45)}…
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* ── Error ── */}
      {error && (
        <div className="mb-5 flex items-start gap-3 bg-red-500/10 border border-red-500/20 rounded-2xl px-4 py-3">
          <span className="text-red-400 mt-0.5 flex-shrink-0 text-lg">⚠</span>
          <div>
            <p className="text-sm font-medium text-red-400">Agent failed</p>
            <p className="text-xs text-red-400/70 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* ── Loading ── */}
      {busy && <AgentProgress elapsed={elapsed} />}

      {/* ── Results ── */}
      {run && !busy && <AgentResults run={run} />}
    </div>
  );
}

// ── Loading state ─────────────────────────────────────────────────────────────
function AgentProgress({ elapsed }: { elapsed: number }) {
  const steps = [
    { label: "Searching knowledge base…", after: 0 },
    { label: "Analyzing documents…",      after: 12 },
    { label: "Synthesizing final answer…",after: 28 },
  ];
  const current = steps.filter(s => elapsed >= s.after).length - 1;
  // Estimated time based on Gemini typical latency
  const estSeconds = 45;

  return (
    <Card className="p-8">
      <div className="flex flex-col items-center gap-5 text-center">
        {/* Animated icon */}
        <div className="relative">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-glow">
            <svg viewBox="0 0 24 24" fill="none" className="w-8 h-8 text-white animate-spin"
              style={{ animationDuration: "3s" }}>
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"
                stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-indigo-500 flex items-center justify-center">
            <span className="text-[9px] text-white font-bold">{elapsed}s</span>
          </div>
        </div>

        <div>
          <p className="text-white font-semibold text-base">Agent is working…</p>
          <p className="text-sm text-slate-400 mt-1">
            {steps[Math.max(0, current)]?.label ?? "Processing…"}
          </p>
        </div>

        {/* Progress bar */}
        <div className="w-full max-w-xs bg-slate-800 rounded-full h-1.5 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-1000"
            style={{ width: `${Math.min(95, (elapsed / estSeconds) * 100)}%` }}
          />
        </div>

        {/* Step indicators */}
        <div className="flex items-center gap-6">
          {steps.map((s, i) => (
            <div key={s.label} className={`flex items-center gap-1.5 text-xs transition-colors ${
              i <= current ? "text-indigo-400" : "text-slate-600"
            }`}>
              <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                i < current ? "bg-indigo-500 text-white" :
                i === current ? "bg-indigo-500/30 ring-2 ring-indigo-500 text-indigo-300" :
                "bg-slate-800 text-slate-600"
              }`}>{i + 1}</span>
              <span className="hidden sm:block">
                {["Search", "Analyze", "Synthesize"][i]}
              </span>
            </div>
          ))}
        </div>

        <p className="text-xs text-slate-600">
          Gemini takes ~30–60 seconds · please wait…
          {elapsed > 50 && <span className="text-amber-500"> (almost done)</span>}
        </p>
      </div>
    </Card>
  );
}

// ── Results ───────────────────────────────────────────────────────────────────
function AgentResults({ run }: { run: AgentRun }) {
  return (
    <div className="space-y-4 animate-slide-up">

      {/* Summary card */}
      <Card className="p-5">
        <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
          <div className="flex items-center gap-2">
            <Badge status={run.status}>{run.status}</Badge>
            <span className="text-xs text-slate-500">
              {run.steps.length} tool call{run.steps.length !== 1 ? "s" : ""}
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span>{(run.latency_ms / 1000).toFixed(1)}s</span>
            <span className="text-slate-700">·</span>
            <span>{run.total_tokens.toLocaleString()} tokens</span>
          </div>
        </div>
        <p className="text-sm text-slate-400 italic">"{run.query}"</p>
      </Card>

      {/* Execution steps */}
      {run.steps.length > 0 && (
        <Card className="overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-700/50 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Execution Steps</h2>
            <span className="text-xs text-slate-500">{run.steps.length} steps</span>
          </div>
          <div className="p-4 space-y-2">
            {run.steps.map((s, i) => {
              const tool = TOOLS[s.tool_name] ?? { color: "text-slate-400", bg: "bg-slate-800/60", icon: "🔧", label: s.tool_name };
              return (
                <details key={i} className="group rounded-xl border border-slate-700/30 overflow-hidden">
                  <summary className={`flex items-center gap-3 px-4 py-3 cursor-pointer list-none ${tool.bg} hover:brightness-110 transition-all`}>
                    <span className="text-base flex-shrink-0">{tool.icon}</span>
                    <div className="flex-1 min-w-0">
                      <span className={`text-sm font-semibold ${tool.color}`}>{tool.label}</span>
                      <span className="ml-2 text-xs text-slate-600">#{s.step_index}</span>
                    </div>
                    <div className="flex items-center gap-3 flex-shrink-0">
                      {s.success ? (
                        <span className="text-xs text-emerald-400 flex items-center gap-1">
                          <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5"><path fillRule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z" clipRule="evenodd"/></svg>
                          OK
                        </span>
                      ) : (
                        <span className="text-xs text-red-400">✗ Failed</span>
                      )}
                      {s.latency_ms > 0 && (
                        <span className="text-xs text-slate-600">{(s.latency_ms / 1000).toFixed(1)}s</span>
                      )}
                      <svg viewBox="0 0 20 20" fill="currentColor"
                        className="w-4 h-4 text-slate-600 group-open:rotate-180 transition-transform duration-200">
                        <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd"/>
                      </svg>
                    </div>
                  </summary>
                  <div className="px-4 pb-4 pt-3 border-t border-slate-800/60 bg-slate-900/40">
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Input</p>
                        <pre className="text-xs text-slate-400 bg-slate-950/60 border border-slate-800 rounded-xl p-3 overflow-x-auto whitespace-pre-wrap">
                          {JSON.stringify(s.tool_input, null, 2)}
                        </pre>
                      </div>
                      <div>
                        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-2">Output</p>
                        <OutputViewer output={s.tool_output} />
                      </div>
                    </div>
                    {!s.success && s.error_message && (
                      <div className="mt-3 flex items-start gap-2 bg-red-500/10 border border-red-500/20 rounded-xl px-3 py-2">
                        <span className="text-red-400 text-xs mt-0.5">⚠</span>
                        <span className="text-xs text-red-400">{s.error_message}</span>
                      </div>
                    )}
                  </div>
                </details>
              );
            })}
          </div>
        </Card>
      )}

      {/* Final answer */}
      {run.final_answer && (
        <Card className="overflow-hidden">
          <div className="px-5 py-3.5 border-b border-slate-700/50 flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-emerald-500/20 flex items-center justify-center">
              <svg viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5 text-emerald-400">
                <path fillRule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z" clipRule="evenodd"/>
              </svg>
            </div>
            <h2 className="text-sm font-semibold text-white">Final Answer</h2>
          </div>
          <div className="p-5 text-sm text-slate-200 leading-relaxed">
            <ReactMarkdown components={MD}>{run.final_answer}</ReactMarkdown>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── Tool output viewer — pretty-prints JSON if possible ───────────────────────
function OutputViewer({ output }: { output: string }) {
  let display = output;
  try {
    const parsed = JSON.parse(output);
    display = JSON.stringify(parsed, null, 2);
  } catch { /* not JSON, show as-is */ }

  return (
    <pre className="text-xs text-slate-400 bg-slate-950/60 border border-slate-800 rounded-xl p-3 overflow-x-auto whitespace-pre-wrap max-h-48">
      {display.slice(0, 800)}{display.length > 800 ? "\n…(truncated)" : ""}
    </pre>
  );
}
