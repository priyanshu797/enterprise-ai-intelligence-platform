"use client";

import { useEffect, useState, useRef } from "react";
import RequireAuth from "@/components/RequireAuth";
import { Badge, Card, Empty, ErrorBanner, SuccessBanner, Spinner, PageHeader } from "@/components/UI";
import { api, Doc } from "@/lib/api";

export default function DocumentsPage() {
  return <RequireAuth><DocumentsView /></RequireAuth>;
}

const FILE_ICONS: Record<string, string> = { pdf: "📄", txt: "📝", md: "📋", csv: "📊", markdown: "📋" };

function DocumentsView() {
  const [docs, setDocs]       = useState<Doc[]>([]);
  const [uploading, setUpload] = useState(false);
  const [error, setError]     = useState("");
  const [msg, setMsg]         = useState("");
  const [selected, setSelected] = useState<Doc | null>(null);
  const [chunks, setChunks]   = useState<any[] | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = () => api.listDocuments().then(setDocs).catch(e => setError(e.message));
  useEffect(() => { load(); }, []);

  // Poll while processing
  useEffect(() => {
    const pending = docs.some(d => d.status === "pending" || d.status === "processing");
    if (!pending) return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [docs]);

  const onUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setUpload(true); setError(""); setMsg("");
    try {
      const doc = await api.uploadDocument(file);
      setMsg(doc.status === "duplicate"
        ? `"${doc.filename}" already exists — skipped.`
        : `"${doc.filename}" uploaded and queued for processing.`);
      await load();
    } catch (err: any) { setError(err.message); }
    finally { setUpload(false); if (fileRef.current) fileRef.current.value = ""; }
  };

  const onDelete = async (id: string) => {
    if (!confirm("Delete this document and its vectors?")) return;
    try { await api.deleteDocument(id); await load(); }
    catch (err: any) { setError(err.message); }
  };

  return (
    <div className="p-6 max-w-6xl animate-fade-in">
      <PageHeader title="Documents" sub="Upload and manage your knowledge base files" />

      {/* Upload zone */}
      <Card className="p-5 mb-6">
        <form onSubmit={onUpload} className="flex items-center gap-4 flex-wrap">
          <label className="flex-1 min-w-0 flex items-center gap-3 bg-slate-900/60 border border-dashed border-slate-700 hover:border-indigo-500/60 rounded-xl px-4 py-3 cursor-pointer transition-colors group">
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-slate-500 group-hover:text-indigo-400 flex-shrink-0 transition-colors">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd"/>
            </svg>
            <span className="text-sm text-slate-400 group-hover:text-slate-300 truncate">
              {fileRef.current?.files?.[0]?.name ?? "Choose file — PDF, TXT, Markdown, CSV (max 25 MB)"}
            </span>
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.markdown,.csv" required className="hidden" />
          </label>
          <button type="submit" disabled={uploading} className="btn-primary flex-shrink-0">
            {uploading ? <><Spinner size={4} />Uploading…</> : <>
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd"/>
              </svg>
              Upload
            </>}
          </button>
        </form>
      </Card>

      {error && <div className="mb-4"><ErrorBanner msg={error} /></div>}
      {msg   && <div className="mb-4"><SuccessBanner msg={msg} /></div>}

      {/* Table */}
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700/50">
                <th className="text-left px-5 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">File</th>
                <th className="text-left px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">Type</th>
                <th className="text-left px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">Status</th>
                <th className="text-right px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">Chunks</th>
                <th className="text-right px-4 py-3.5 text-xs font-semibold text-slate-500 uppercase tracking-wider">Uploaded</th>
                <th className="px-4 py-3.5"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {docs.length === 0 && (
                <tr><td colSpan={6}><Empty icon="📂" title="No documents yet" desc="Upload a PDF, TXT, Markdown or CSV file to start building your knowledge base." /></td></tr>
              )}
              {docs.map(d => (
                <tr key={d.id} className="hover:bg-white/[.02] transition-colors">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <span className="text-xl">{FILE_ICONS[d.file_type] ?? "📄"}</span>
                      <div className="min-w-0">
                        <p className="font-medium text-slate-200 truncate max-w-xs">{d.filename}</p>
                        <p className="text-xs text-slate-500">{(d.file_size / 1024).toFixed(1)} KB</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-4">
                    <span className="text-xs font-semibold text-slate-400 uppercase bg-slate-800 rounded px-2 py-0.5">{d.file_type}</span>
                  </td>
                  <td className="px-4 py-4">
                    <div className="flex items-center gap-2">
                      {(d.status === "pending" || d.status === "processing") && (
                        <span className="w-2 h-2 rounded-full bg-blue-400 pulse-dot flex-shrink-0" />
                      )}
                      <Badge status={d.status}>{d.status}</Badge>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-right">
                    <span className="text-slate-300 font-semibold">{d.num_chunks}</span>
                  </td>
                  <td className="px-4 py-4 text-right text-xs text-slate-500">
                    {new Date(d.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-4 text-right">
                    <button onClick={() => { setSelected(d); setChunks(null); api.getChunks(d.id).then(setChunks); }}
                      className="text-xs text-indigo-400 hover:text-indigo-300 mr-3 font-medium">
                      Chunks
                    </button>
                    <button onClick={() => onDelete(d.id)} className="text-xs text-slate-500 hover:text-red-400 transition-colors">
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Chunk drawer */}
      {selected && (
        <div className="fixed inset-y-0 right-0 w-[420px] bg-slate-900 border-l border-slate-800 z-50 flex flex-col shadow-2xl animate-slide-up">
          <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
            <div>
              <p className="font-semibold text-white text-sm">{selected.filename}</p>
              <p className="text-xs text-slate-500 mt-0.5">{selected.num_chunks} chunks</p>
            </div>
            <button onClick={() => setSelected(null)} className="btn-ghost p-2">
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd"/>
              </svg>
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {chunks === null ? (
              <div className="flex justify-center py-8"><Spinner size={5} /></div>
            ) : chunks.length === 0 ? (
              <Empty icon="🧩" title="No chunks" desc="Document may still be processing." />
            ) : (
              chunks.map(c => (
                <div key={c.id} className="bg-slate-800/60 rounded-xl p-4 border border-slate-700/30">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-xs font-semibold text-indigo-400">#{c.chunk_index}</span>
                    <span className="text-xs text-slate-500">{c.token_count} tokens · {c.char_count} chars</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed line-clamp-5">{c.text}</p>
                </div>
              ))
            )}
          </div>
        </div>
      )}
      {selected && <div className="fixed inset-0 bg-black/40 z-40" onClick={() => setSelected(null)} />}
    </div>
  );
}
