import { Activity, Sun, Moon, Radio } from "lucide-react";
import clsx from "clsx";

export default function TopBar({ runs, selectedRunId, onRunChange, sseConnected, theme, onToggleTheme }) {
  return (
    <header className="flex flex-wrap items-center gap-4 border-b border-border py-4 mb-5">
      {/* Brand */}
      <div className="flex items-center gap-2.5">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-accent-dimmed">
          <Activity className="w-4 h-4 text-accent" />
        </div>
        <h1 className="text-base font-semibold tracking-tight">DABench Traces</h1>
      </div>

      {/* Run selector */}
      <div className="flex items-center gap-2">
        <label className="text-xs text-fg-muted uppercase tracking-wider font-medium">Run</label>
        <select
          value={selectedRunId}
          onChange={(e) => onRunChange(e.target.value)}
          className="min-w-[220px] rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-fg outline-none
                     focus:border-accent focus:ring-1 focus:ring-accent/30 transition-colors"
        >
          <option value="">All runs</option>
          {runs.map((r) => (
            <option key={r.run_id} value={r.run_id}>
              {r.run_id} ({r.task_count} tasks)
            </option>
          ))}
        </select>
      </div>

      {/* Live indicator */}
      <div className="flex items-center gap-2 ml-auto">
        <div className={clsx(
          "w-2 h-2 rounded-full shrink-0",
          sseConnected ? "bg-success live-glow text-success" : "bg-fg-muted"
        )} />
        <span className="text-xs text-fg-muted">
          {sseConnected ? "Live" : "Disconnected"}
        </span>
      </div>

      {/* Theme toggle */}
      <button
        onClick={onToggleTheme}
        className="p-2 rounded-lg text-fg-muted hover:text-fg hover:bg-elevated transition-colors"
        title="Toggle theme"
      >
        {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
      </button>
    </header>
  );
}
