export function fmt(n) {
  if (n == null) return "–";
  return Number(n).toLocaleString("en-US");
}

export function fmtDuration(ms) {
  if (ms == null) return "–";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const m = Math.floor(ms / 60_000);
  const s = ((ms % 60_000) / 1000).toFixed(0);
  return `${m}m ${s}s`;
}

export function fmtTime(iso) {
  if (!iso) return "–";
  try {
    return new Date(iso).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

export function fmtTokens(n) {
  if (n == null) return "–";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}


export function prettyJson(obj) {
  if (obj == null) return "";
  try {
    return typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

export function truncate(s, max = 4000) {
  if (!s) return "";
  s = String(s);
  return s.length > max ? s.slice(0, max) + "…" : s;
}

export function kindClass(kind) {
  const map = {
    generation: "accent",
    function: "success",
    error: "danger",
    turn: "muted",
    agent: "purple",
  };
  return map[kind] || "muted";
}

export function spanDuration(sp) {
  if (!sp.started_at || !sp.ended_at) return 0;
  try {
    return new Date(sp.ended_at) - new Date(sp.started_at);
  } catch {
    return 0;
  }
}

export function buildSpanTree(flatSpans) {
  const byId = new Map();
  const roots = [];
  for (const sp of flatSpans) {
    byId.set(sp.span_id, { ...sp, children: [] });
  }
  for (const sp of flatSpans) {
    const node = byId.get(sp.span_id);
    if (sp.parent_span_id && byId.has(sp.parent_span_id)) {
      byId.get(sp.parent_span_id).children.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}
