async function request(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchRuns() {
  return request("/api/runs");
}

export async function fetchTraces({
  runId,
  status,
  limit = 50,
  offset = 0,
  sortBy = "started_at",
  sortOrder = "desc",
} = {}) {
  const p = new URLSearchParams();
  if (runId) p.set("run_id", runId);
  if (status) p.set("status", status);
  p.set("limit", String(limit));
  p.set("offset", String(offset));
  p.set("sort_by", sortBy);
  p.set("sort_order", sortOrder);
  return request("/api/traces?" + p);
}

export async function fetchTraceDetail(traceId) {
  return request("/api/traces/" + encodeURIComponent(traceId));
}

export async function fetchRateLimitStats(runId) {
  const p = new URLSearchParams();
  if (runId) p.set("run_id", runId);
  return request("/api/stats/rate-limit?" + p);
}

export function createSSE(runId, { onSpan, onConnect, onDisconnect }) {
  const params = runId ? "?run_id=" + encodeURIComponent(runId) : "";
  const source = new EventSource("/api/stream" + params);

  source.onopen = () => onConnect?.();
  source.onerror = () => onDisconnect?.();

  const handler = (e) => onSpan?.(JSON.parse(e.data));
  source.addEventListener("span_start", handler);
  source.addEventListener("span_end", handler);

  return () => source.close();
}
