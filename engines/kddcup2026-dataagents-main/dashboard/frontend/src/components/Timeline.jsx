import clsx from "clsx";
import { spanDuration, fmtDuration, kindClass } from "../utils";
import { StatusDot } from "./StatusBadge";

const KIND_COLORS = {
  accent: { bar: "bg-accent", badge: "text-accent bg-accent-dimmed" },
  success: { bar: "bg-success", badge: "text-success bg-success-dimmed" },
  danger: { bar: "bg-danger", badge: "text-danger bg-danger-dimmed" },
  warning: { bar: "bg-warning", badge: "text-warning bg-warning-dimmed" },
  purple: { bar: "bg-purple", badge: "text-purple bg-purple-dimmed" },
  muted: { bar: "bg-fg-muted", badge: "text-fg-muted bg-muted-bg" },
};

function TimelineNode({ node, depth, maxDuration, selectedSpanId, onSelectSpan, index }) {
  const dur = spanDuration(node);
  const pct = maxDuration > 0 ? Math.max(2, (dur / maxDuration) * 100) : 2;
  const kc = kindClass(node.kind);
  const colors = KIND_COLORS[kc] || KIND_COLORS.muted;
  const isActive = node.span_id === selectedSpanId;
  const indent = depth * 20;

  return (
    <>
      <div
        onClick={() => onSelectSpan(node.span_id)}
        className={clsx(
          "px-3 py-2 cursor-pointer transition-all border-b border-border/30",
          isActive
            ? "bg-accent-dimmed border-l-2 border-l-accent"
            : "hover:bg-elevated/50 border-l-2 border-l-transparent",
        )}
      >
        <div className="flex items-center gap-2 text-[13px]" style={{ paddingLeft: indent }}>
          {/* Kind badge */}
          <span className={clsx("text-[10px] font-semibold uppercase tracking-wide px-1.5 py-px rounded shrink-0", colors.badge)}>
            {node.kind}
          </span>
          {/* Name */}
          <span className={clsx("flex-1 truncate", isActive ? "text-fg" : "text-fg-secondary")}>
            {node.name}
          </span>
          {/* Status */}
          <StatusDot status={node.status} />
          {/* Duration */}
          <span className="text-xs text-fg-muted font-mono tabular-nums shrink-0">
            {fmtDuration(dur)}
          </span>
        </div>

        {/* Bar */}
        <div className="mt-1.5 h-1 rounded-full bg-border/50 overflow-hidden" style={{ marginLeft: indent }}>
          <div
            className={clsx("h-full rounded-full bar-animated", colors.bar)}
            style={{ width: `${pct}%`, animationDelay: `${index * 40}ms`, opacity: 0.8 }}
          />
        </div>
      </div>

      {node.children?.map((child, i) => (
        <TimelineNode
          key={child.span_id}
          node={child}
          depth={depth + 1}
          maxDuration={maxDuration}
          selectedSpanId={selectedSpanId}
          onSelectSpan={onSelectSpan}
          index={index + i + 1}
        />
      ))}
    </>
  );
}

export default function Timeline({ nodes, maxDuration, selectedSpanId, onSelectSpan }) {
  if (!nodes.length) {
    return (
      <div className="flex items-center justify-center h-full py-12 text-fg-muted text-sm">
        No spans recorded.
      </div>
    );
  }

  let idx = 0;
  return (
    <div>
      {nodes.map((node) => {
        const currentIdx = idx;
        idx += 1 + (node.children?.length || 0);
        return (
          <TimelineNode
            key={node.span_id}
            node={node}
            depth={0}
            maxDuration={maxDuration}
            selectedSpanId={selectedSpanId}
            onSelectSpan={onSelectSpan}
            index={currentIdx}
          />
        );
      })}
    </div>
  );
}
