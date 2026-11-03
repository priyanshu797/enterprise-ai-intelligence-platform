"use client";

import { ReactNode } from "react";

// ─── Stat card ────────────────────────────────────────────────────────────────
export function StatCard({
  label, value, sub, icon, gradient,
}: {
  label: string; value: string; sub?: string;
  icon?: ReactNode; gradient?: string;
}) {
  return (
    <div className="relative overflow-hidden rounded-2xl bg-slate-800/50 border border-slate-700/50 p-5">
      {gradient && (
        <div className={`absolute inset-0 opacity-10 ${gradient}`} />
      )}
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">{label}</p>
          <p className="mt-2 text-2xl font-bold text-white">{value}</p>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        {icon && (
          <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-slate-400">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Section card ─────────────────────────────────────────────────────────────
export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-2xl bg-slate-800/50 border border-slate-700/50 ${className}`}>
      {children}
    </div>
  );
}

// ─── Status badge ─────────────────────────────────────────────────────────────
const STATUS: Record<string, string> = {
  answered:     "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
  insufficient: "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30",
  conflicting:  "bg-orange-500/15 text-orange-400 ring-1 ring-orange-500/30",
  unsupported:  "bg-red-500/15 text-red-400 ring-1 ring-red-500/30",
  pending:      "bg-slate-500/15 text-slate-400 ring-1 ring-slate-500/30",
  processing:   "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30",
  completed:    "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
  failed:       "bg-red-500/15 text-red-400 ring-1 ring-red-500/30",
  duplicate:    "bg-slate-500/15 text-slate-400 ring-1 ring-slate-500/30",
  running:      "bg-blue-500/15 text-blue-400 ring-1 ring-blue-500/30",
  success:      "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
};

export function Badge({ status, children }: { status: string; children?: ReactNode }) {
  return (
    <span className={`badge ${STATUS[status] || STATUS.pending}`}>
      {children ?? status}
    </span>
  );
}

// ─── Spinner ──────────────────────────────────────────────────────────────────
export function Spinner({ size = 4 }: { size?: number }) {
  return (
    <svg
      className={`animate-spin w-${size} h-${size} text-indigo-400`}
      fill="none" viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

// ─── Empty state ──────────────────────────────────────────────────────────────
export function Empty({ icon, title, desc }: { icon: string; title: string; desc?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-14 text-center">
      <div className="text-4xl mb-3">{icon}</div>
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {desc && <p className="mt-1 text-xs text-slate-500 max-w-xs">{desc}</p>}
    </div>
  );
}

// ─── Page header ──────────────────────────────────────────────────────────────
export function PageHeader({ title, sub, action }: { title: string; sub?: string; action?: ReactNode }) {
  return (
    <div className="flex items-start justify-between mb-6">
      <div>
        <h1 className="text-xl font-bold text-white">{title}</h1>
        {sub && <p className="mt-1 text-sm text-slate-400">{sub}</p>}
      </div>
      {action}
    </div>
  );
}

// ─── Error banner ─────────────────────────────────────────────────────────────
export function ErrorBanner({ msg }: { msg: string }) {
  return (
    <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3">
      <span>⚠</span> {msg}
    </div>
  );
}

export function SuccessBanner({ msg }: { msg: string }) {
  return (
    <div className="flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm rounded-xl px-4 py-3">
      <span>✓</span> {msg}
    </div>
  );
}
