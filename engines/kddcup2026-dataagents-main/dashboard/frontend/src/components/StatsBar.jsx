import { BarChart3, CheckCircle2, Coins, Clock, Timer } from "lucide-react";
import { fmt, fmtDuration, fmtTokens } from "../utils";

function StatCard({ icon: Icon, label, value, color, subtext }) {
  return (
    <div className="group relative flex items-center gap-3 rounded-xl border border-border bg-surface px-4 py-3 min-w-[140px]
                    hover:border-border-hover transition-colors">
      <div className={`flex items-center justify-center w-9 h-9 rounded-lg shrink-0`}
           style={{ background: `var(--color-${color}-dimmed)` }}>
        <Icon className="w-4 h-4" style={{ color: `var(--color-${color})` }} />
      </div>
      <div className="flex flex-col">
        <span className="text-[11px] font-medium text-fg-muted uppercase tracking-wider">{label}</span>
        <div className="flex items-baseline gap-1.5">
          <span className="text-xl font-semibold tabular-nums animate-count">{value}</span>
          {subtext && <span className="text-xs text-fg-muted">{subtext}</span>}
        </div>
      </div>
    </div>
  );
}

export default function StatsBar({ runs, selectedRunId, rlStats }) {
  const run = selectedRunId ? runs.find((r) => r.run_id === selectedRunId) : null;
  const taskCount = run ? run.task_count : runs.reduce((s, r) => s + r.task_count, 0);
  const okCount = run ? run.ok_count : runs.reduce((s, r) => s + (r.ok_count || 0), 0);
  const tokens = run ? run.tokens : runs.reduce((s, r) => s + (r.tokens || 0), 0);
  const avgMs = run
    ? run.avg_ms
    : runs.length
      ? runs.reduce((s, r) => s + (r.avg_ms || 0), 0) / runs.length
      : null;
  const pct = taskCount > 0 ? Math.round((okCount / taskCount) * 100) : 0;

  return (
    <div className="flex flex-wrap items-center gap-3 mb-5">
      <StatCard icon={BarChart3} label="Tasks" value={fmt(taskCount)} color="accent" />
      <StatCard icon={CheckCircle2} label="OK" value={fmt(okCount)} color="success" subtext={`${pct}%`} />
      <StatCard icon={Coins} label="Tokens" value={fmtTokens(tokens)} color="purple" />
      <StatCard icon={Clock} label="Avg Duration" value={fmtDuration(avgMs)} color="warning" />
      {rlStats?.rl_generations > 0 && (
        <StatCard icon={Timer} label="RL Wait" value={fmtDuration(rlStats.total_acquire_wait_ms)}
                  color="warning" subtext={`${rlStats.total_retries ?? 0} retries`} />
      )}
    </div>
  );
}
