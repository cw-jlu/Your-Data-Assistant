import { Check, Clock, X, Loader2 } from "lucide-react";
import clsx from "clsx";

const config = {
  completed: { icon: Check, color: "success", label: "Completed" },
  error: { icon: X, color: "danger", label: "Error" },
  timeout: { icon: Clock, color: "warning", label: "Timeout" },
  running: { icon: Loader2, color: "warning", label: "Running" },
};

export default function StatusBadge({ status, size = "sm" }) {
  const { icon: Icon, color, label } = config[status] || config.running;
  const isRunning = status === "running";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 font-medium rounded-full",
        size === "sm" && "text-xs px-2.5 py-0.5",
        size === "md" && "text-sm px-3 py-1",
      )}
      style={{
        color: `var(--color-${color})`,
        background: `var(--color-${color}-dimmed)`,
      }}
    >
      <Icon className={clsx("w-3 h-3 shrink-0", isRunning && "animate-spin")} />
      {label}
    </span>
  );
}

export function StatusDot({ status }) {
  const { color } = config[status] || config.running;
  const symbol =
    status === "completed" ? "✓" : status === "error" ? "✕" : status === "timeout" ? "!" : "●";

  return (
    <span className="text-xs" style={{ color: `var(--color-${color})` }}>
      {symbol}
    </span>
  );
}
