import { ChevronLeft, ChevronRight, Search } from "lucide-react";
import clsx from "clsx";
import { fmt, fmtDuration, fmtTime, fmtTokens } from "../utils";
import StatusBadge from "./StatusBadge";

function navigate(hash) {
  window.location.hash = hash;
}

export default function TraceList({
  traces,
  total,
  hasMore,
  offset,
  limit,
  loading,
  statusFilter,
  sortBy,
  sortOrder,
  onStatusFilter,
  onSort,
  onPrev,
  onNext,
}) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const sortVal = `${sortBy}:${sortOrder}`;

  return (
    <div>
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-fg-muted uppercase tracking-wider font-medium">Status</label>
          <select
            value={statusFilter}
            onChange={(e) => onStatusFilter(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-fg outline-none
                       focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors"
          >
            <option value="">All</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="timeout">Timeout</option>
            <option value="error">Error</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-fg-muted uppercase tracking-wider font-medium">Sort</label>
          <select
            value={sortVal}
            onChange={(e) => onSort(e.target.value)}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-fg outline-none
                       focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors"
          >
            <option value="started_at:desc">Newest first</option>
            <option value="started_at:asc">Oldest first</option>
            <option value="duration_ms:desc">Longest first</option>
            <option value="duration_ms:asc">Shortest first</option>
            <option value="total_tokens:desc">Most tokens</option>
            <option value="task_id:asc">Task ID A–Z</option>
          </select>
        </div>
        <span className="ml-auto text-xs text-fg-muted tabular-nums">
          {fmt(total)} trace{total !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Loading state */}
      {loading && traces.length === 0 && (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-14 rounded-xl shimmer" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && traces.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="w-14 h-14 rounded-2xl bg-panel flex items-center justify-center mb-4">
            <Search className="w-6 h-6 text-fg-muted" />
          </div>
          <p className="text-fg-secondary font-medium">No traces found</p>
          <p className="text-sm text-fg-muted mt-1">
            Enable SQLite tracing and run the agent to see traces here.
          </p>
        </div>
      )}

      {/* Table */}
      {traces.length > 0 && (
        <div className="rounded-xl border border-border overflow-hidden bg-surface">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-border bg-panel/50">
                {["Task ID", "Status", "Duration", "Tokens", "Steps", "Time"].map((h) => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 text-[11px] font-semibold text-fg-muted uppercase tracking-wider whitespace-nowrap"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traces.map((t, i) => (
                <tr
                  key={t.trace_id}
                  onClick={() => navigate("trace/" + encodeURIComponent(t.trace_id))}
                  className="group border-b border-border/50 last:border-b-0 cursor-pointer
                             hover:bg-accent-dimmed/50 transition-colors animate-fade-up"
                  style={{ animationDelay: `${i * 30}ms` }}
                >
                  <td className="px-4 py-3 text-sm font-medium text-accent">
                    {t.task_id || t.trace_id}
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={t.status} />
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-fg-secondary tabular-nums">
                    {fmtDuration(t.duration_ms)}
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-fg-secondary tabular-nums">
                    {fmtTokens(t.total_tokens)}
                  </td>
                  <td className="px-4 py-3 text-sm font-mono text-fg-secondary tabular-nums">
                    {fmt(t.span_count)}
                  </td>
                  <td className="px-4 py-3 text-sm text-fg-muted whitespace-nowrap">
                    {fmtTime(t.started_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {total > 0 && (
        <div className="flex items-center justify-between pt-4">
          <span className="text-xs text-fg-muted tabular-nums">
            Page {page} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              disabled={offset <= 0}
              onClick={onPrev}
              className="flex items-center gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-fg
                         hover:bg-elevated transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              Previous
            </button>
            <button
              disabled={!hasMore}
              onClick={onNext}
              className="flex items-center gap-1 rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-fg
                         hover:bg-elevated transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              Next
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
