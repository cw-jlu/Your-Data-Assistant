import { useState, useEffect, useCallback, useRef } from "react";
import * as api from "./api";
import TopBar from "./components/TopBar";
import StatsBar from "./components/StatsBar";
import TraceList from "./components/TraceList";
import TraceDetail from "./components/TraceDetail";

function getRoute() {
  const h = window.location.hash.slice(1) || "";
  if (h.startsWith("trace/")) return { view: "detail", traceId: decodeURIComponent(h.slice(6)) };
  return { view: "list", traceId: null };
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem("dabench-theme");
    if (stored === "light" || stored === "dark") return stored;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("dabench-theme", theme);
  }, [theme]);

  const toggle = useCallback(() => setTheme((t) => (t === "dark" ? "light" : "dark")), []);
  return [theme, toggle];
}

export default function App() {
  const [theme, toggleTheme] = useTheme();

  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [traces, setTraces] = useState([]);
  const [total, setTotal] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [offset, setOffset] = useState(0);
  const limit = 50;
  const [statusFilter, setStatusFilter] = useState("");
  const [sortBy, setSortBy] = useState("started_at");
  const [sortOrder, setSortOrder] = useState("desc");
  const [loading, setLoading] = useState(false);

  const [route, setRoute] = useState(getRoute);
  const [currentTrace, setCurrentTrace] = useState(null);
  const [currentSpans, setCurrentSpans] = useState([]);
  const [selectedSpanId, setSelectedSpanId] = useState(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [rlStats, setRlStats] = useState(null);

  const sseCleanup = useRef(null);

  const loadRuns = useCallback(async () => {
    try {
      setRuns(await api.fetchRuns());
    } catch {
      setRuns([]);
    }
  }, []);

  const loadTraces = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.fetchTraces({
        runId: selectedRunId,
        status: statusFilter,
        limit,
        offset,
        sortBy,
        sortOrder,
      });
      setTraces(data.items || []);
      setTotal(data.total || 0);
      setHasMore(data.has_more || false);
    } catch {
      setTraces([]);
      setTotal(0);
    }
    setLoading(false);
  }, [selectedRunId, statusFilter, offset, sortBy, sortOrder]);

  const loadDetail = useCallback(async (traceId, { resetSelection = false } = {}) => {
    setLoading(true);
    try {
      const data = await api.fetchTraceDetail(traceId);
      setCurrentTrace(data.trace);
      setCurrentSpans(data.spans || []);
      if (resetSelection) setSelectedSpanId(null);
    } catch {
      setCurrentTrace(null);
      setCurrentSpans([]);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    const onHash = () => setRoute(getRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    api.fetchRateLimitStats(selectedRunId || undefined)
      .then(setRlStats)
      .catch(() => setRlStats(null));
  }, [selectedRunId]);

  useEffect(() => {
    if (route.view === "list") {
      loadTraces();
    } else if (route.traceId) {
      loadDetail(route.traceId, { resetSelection: true });
    }
  }, [route, loadTraces, loadDetail]);

  useEffect(() => {
    if (route.view === "list") {
      const id = setInterval(() => {
        loadTraces();
        loadRuns();
      }, 5000);
      return () => clearInterval(id);
    }
    if (route.view === "detail" && currentTrace?.status === "running") {
      const id = setInterval(() => loadDetail(route.traceId), 5000);
      return () => clearInterval(id);
    }
  }, [route, currentTrace?.status, loadTraces, loadRuns, loadDetail]);

  useEffect(() => {
    sseCleanup.current?.();
    const cleanup = api.createSSE(selectedRunId, {
      onConnect: () => setSseConnected(true),
      onDisconnect: () => setSseConnected(false),
      onSpan: (span) => {
        setCurrentTrace((t) => {
          if (!t || span.trace_id !== t.trace_id) return t;
          return t;
        });
        setCurrentSpans((prev) => {
          if (!prev.length) return prev;
          const idx = prev.findIndex((s) => s.span_id === span.span_id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = span;
            return next;
          }
          if (prev[0] && span.trace_id === prev[0].trace_id) {
            return [...prev, span];
          }
          return prev;
        });
      },
    });
    sseCleanup.current = cleanup;
    return cleanup;
  }, [selectedRunId]);

  const handleRunChange = (id) => {
    setSelectedRunId(id);
    setOffset(0);
  };

  const handleSort = (val) => {
    const [by, order] = val.split(":");
    setSortBy(by);
    setSortOrder(order);
    setOffset(0);
  };

  const handleStatusFilter = (val) => {
    setStatusFilter(val);
    setOffset(0);
  };

  return (
    <div className="mx-auto max-w-[1440px] px-5 pb-12">
      <TopBar
        runs={runs}
        selectedRunId={selectedRunId}
        onRunChange={handleRunChange}
        sseConnected={sseConnected}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {route.view === "list" && (
        <>
          <StatsBar runs={runs} selectedRunId={selectedRunId} rlStats={rlStats} />
          <TraceList
            traces={traces}
            total={total}
            hasMore={hasMore}
            offset={offset}
            limit={limit}
            loading={loading}
            statusFilter={statusFilter}
            sortBy={sortBy}
            sortOrder={sortOrder}
            onStatusFilter={handleStatusFilter}
            onSort={handleSort}
            onPrev={() => setOffset((o) => Math.max(0, o - limit))}
            onNext={() => setOffset((o) => o + limit)}
          />
        </>
      )}

      {route.view === "detail" && (
        <TraceDetail
          trace={currentTrace}
          spans={currentSpans}
          selectedSpanId={selectedSpanId}
          onSelectSpan={setSelectedSpanId}
          loading={loading}
        />
      )}

    </div>
  );
}
