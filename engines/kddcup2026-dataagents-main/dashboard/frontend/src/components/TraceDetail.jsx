import { ArrowLeft, Clock, Coins, Layers, Search } from "lucide-react";
import { buildSpanTree, spanDuration, fmt, fmtDuration, fmtTokens } from "../utils";
import StatusBadge from "./StatusBadge";
import Timeline from "./Timeline";
import Inspector from "./Inspector";

function navigate(hash) {
  window.location.hash = hash;
}

export default function TraceDetail({ trace, spans, selectedSpanId, onSelectSpan, loading }) {
  if (loading && !trace) {
    return (
      <div className="space-y-4 mt-4">
        <div className="h-16 rounded-xl shimmer" />
        <div className="grid grid-cols-2 gap-4">
          <div className="h-96 rounded-xl shimmer" />
          <div className="h-96 rounded-xl shimmer" />
        </div>
      </div>
    );
  }

  if (!trace) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="w-14 h-14 rounded-2xl bg-panel flex items-center justify-center mb-4">
          <Search className="w-6 h-6 text-fg-muted" />
        </div>
        <p className="text-fg-secondary font-medium">Trace not found</p>
      </div>
    );
  }

  const tree = buildSpanTree(spans);
  const maxDur = Math.max(...spans.map(spanDuration), 1);
  const selectedSpan = spans.find((s) => s.span_id === selectedSpanId);

  return (
    <div>
      {/* Header */}
      <div className="flex flex-wrap items-center gap-4 py-4 border-b border-border mb-4 animate-fade-up">
        <button
          onClick={() => navigate("")}
          className="p-2 rounded-lg text-fg-muted hover:text-fg hover:bg-elevated transition-colors"
          title="Back to list"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <h2 className="text-lg font-semibold tracking-tight">
          {trace.task_id || trace.trace_id}
        </h2>
        <StatusBadge status={trace.status} size="md" />

        <div className="flex items-center gap-5 ml-auto text-sm text-fg-secondary">
          <span className="flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-fg-muted" />
            <strong className="font-semibold text-fg tabular-nums">{fmtDuration(trace.duration_ms)}</strong>
          </span>
          <span className="flex items-center gap-1.5">
            <Coins className="w-3.5 h-3.5 text-fg-muted" />
            <strong className="font-semibold text-fg tabular-nums">{fmtTokens(trace.total_tokens)}</strong>
          </span>
          <span className="flex items-center gap-1.5">
            <Layers className="w-3.5 h-3.5 text-fg-muted" />
            <strong className="font-semibold text-fg tabular-nums">{fmt(spans.length)}</strong>
          </span>
        </div>
      </div>

      {/* Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-4 min-h-[calc(100vh-200px)]">
        {/* Timeline */}
        <div className="flex flex-col rounded-xl border border-border bg-surface overflow-hidden animate-fade-up"
             style={{ animationDelay: "50ms" }}>
          <div className="px-4 py-3 border-b border-border bg-panel/50">
            <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider">Span Timeline</h3>
          </div>
          <div className="flex-1 overflow-y-auto">
            <Timeline
              nodes={tree}
              maxDuration={maxDur}
              selectedSpanId={selectedSpanId}
              onSelectSpan={onSelectSpan}
            />
          </div>
        </div>

        {/* Inspector */}
        <div className="lg:sticky lg:top-4 lg:self-start flex flex-col rounded-xl border border-border bg-surface overflow-hidden animate-fade-up"
             style={{ animationDelay: "100ms" }}>
          <div className="px-4 py-3 border-b border-border bg-panel/50">
            <h3 className="text-[11px] font-semibold text-fg-muted uppercase tracking-wider">Inspector</h3>
          </div>
          <div className="flex-1 overflow-y-auto">
            <Inspector span={selectedSpan} />
          </div>
        </div>
      </div>
    </div>
  );
}
