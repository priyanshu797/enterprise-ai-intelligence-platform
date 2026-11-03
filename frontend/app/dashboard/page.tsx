"use client";

import { useEffect, useState } from "react";
import RequireAuth from "@/components/RequireAuth";
import { StatCard, Card, Badge, Empty, ErrorBanner } from "@/components/UI";
import { api, DashboardStats } from "@/lib/api";

export default function DashboardPage() {
  return <RequireAuth><DashboardView /></RequireAuth>;
}

function DashboardView() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    Promise.all([api.dashboard(), api.health()])
      .then(([s, h]) => { setStats(s); setHealth(h); })
      .catch(e => setErr(e.message));
  }, []);

  if (err) return <div className="p-6"><ErrorBanner msg={err} /></div>;

  return (
    <div className="p-6 max-w-6xl animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <div className="mt-2 flex items-center gap-3 flex-wrap">
          {health && (
            <>
              <span className="flex items-center gap-1.5 text-xs text-slate-400">
                <span className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
                {health.environment}
              </span>
              <span className="text-slate-700">·</span>
              <span className="text-xs text-slate-400">
                Primary: <span className="text-indigo-400 font-medium">{health.llm_primary}</span>
              </span>
              <span className="text-slate-700">·</span>
              <span className="text-xs text-slate-400">
                Fallback: <span className="text-purple-400 font-medium">{health.llm_fallback}</span>
              </span>
              <span className="text-slate-700">·</span>
              <span className="text-xs text-slate-400">
                Vector: <span className="text-cyan-400 font-medium">{health.vector_store}</span>
              </span>
            </>
          )}
        </div>
      </div>

      {/* Skeleton while loading */}
      {!stats ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="h-24 rounded-2xl shimmer" />
          ))}
        </div>
      ) : (
        <>
          {/* Primary stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <StatCard label="Documents" value={String(stats.total_documents)}
              sub={`${stats.total_chunks} chunks indexed`}
              gradient="bg-grad-indigo"
              icon={<svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd"/></svg>} />
            <StatCard label="Conversations" value={String(stats.total_conversations)}
              sub={`${stats.total_messages} messages`}
              gradient="bg-grad-purple"
              icon={<svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z"/></svg>} />
            <StatCard label="Agent runs" value={String(stats.total_agent_runs)}
              gradient="bg-grad-cyan"
              icon={<svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path d="M13 7H7v6h6V7z"/><path fillRule="evenodd" d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z" clipRule="evenodd"/></svg>} />
            <StatCard label="Avg Latency" value={`${stats.avg_latency_ms}ms`}
              gradient="bg-grad-green"
              icon={<svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd"/></svg>} />
          </div>

          {/* Token stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <StatCard label="Total Tokens" value={stats.total_tokens.toLocaleString()}
              sub={`↑${stats.prompt_tokens.toLocaleString()} in  ↓${stats.completion_tokens.toLocaleString()} out`} />
            <StatCard label="Est. Cost" value={`$${stats.total_cost_usd.toFixed(4)}`}
              sub="Approximate USD" />
            <StatCard label="Cache Hits" value={String(stats.cache_hits)}
              sub="Saved LLM calls" />
            <StatCard label="Providers"
              value={Object.keys(stats.tokens_by_provider).length ? Object.keys(stats.tokens_by_provider).join(", ") : "—"}
              sub={Object.keys(stats.tokens_by_provider).length ? Object.entries(stats.tokens_by_provider).map(([k,v]) => `${k}: ${(v as number).toLocaleString()}`).join(" · ") : undefined} />
          </div>

          {/* Bottom panels */}
          <div className="grid md:grid-cols-2 gap-4">
            <Card className="p-5">
              <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-indigo-400">
                  <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd"/>
                </svg>
                Documents by Status
              </h2>
              {Object.keys(stats.documents_by_status).length === 0 ? (
                <p className="text-sm text-slate-500">No documents yet.</p>
              ) : (
                <div className="space-y-2">
                  {Object.entries(stats.documents_by_status).map(([k, v]) => (
                    <div key={k} className="flex items-center justify-between">
                      <Badge status={k}>{k}</Badge>
                      <span className="text-sm font-semibold text-white">{v as number}</span>
                    </div>
                  ))}
                </div>
              )}
            </Card>

            <Card className="p-5">
              <h2 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-purple-400">
                  <path fillRule="evenodd" d="M18 3a1 1 0 00-1.447-.894L8.763 6H5a3 3 0 000 6h.28l1.771 5.316A1 1 0 008 18h1a1 1 0 001-1v-4.382l6.553 3.276A1 1 0 0018 15V3z" clipRule="evenodd"/>
                </svg>
                Recent Activity
              </h2>
              {stats.recent_activity.length === 0 ? (
                <p className="text-sm text-slate-500">No activity yet.</p>
              ) : (
                <ul className="space-y-2 max-h-52 overflow-y-auto">
                  {stats.recent_activity.map((a: any) => (
                    <li key={a.id} className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 min-w-0">
                        <Badge status={a.status}>{a.action}</Badge>
                        <span className="text-xs text-slate-500 truncate">{a.detail?.slice(0,40)}</span>
                      </div>
                      <span className="text-xs text-slate-600 flex-shrink-0">
                        {new Date(a.created_at).toLocaleTimeString()}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
