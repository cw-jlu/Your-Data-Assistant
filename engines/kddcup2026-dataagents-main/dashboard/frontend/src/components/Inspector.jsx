import { Fragment, useState, useMemo } from "react";
import { ChevronRight, MousePointer2 } from "lucide-react";
import clsx from "clsx";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import sql from "highlight.js/lib/languages/sql";
import { Marked } from "marked";
import { spanDuration, fmtDuration, fmtTime, fmt, prettyJson, truncate, kindClass } from "../utils";
import StatusBadge from "./StatusBadge";

hljs.registerLanguage("python", python);
hljs.registerLanguage("sql", sql);

function escapeHtmlForMarked(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const marked = new Marked({
  renderer: {
    code({ text, lang }) {
      let html;
      try {
        html = lang && hljs.getLanguage(lang)
          ? hljs.highlight(text, { language: lang }).value
          : escapeHtmlForMarked(text);
      } catch { html = escapeHtmlForMarked(text); }
      return `<pre class="hljs font-mono text-xs leading-relaxed whitespace-pre-wrap break-words bg-canvas border border-border rounded-lg p-3 overflow-x-auto my-2"><code>${html}</code></pre>`;
    },
    html({ text }) {
      return escapeHtmlForMarked(text);
    },
  },
  gfm: true,
  breaks: true,
});

function Collapsible({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="mt-3 rounded-lg border border-border overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 w-full px-3 py-2 text-left text-sm font-medium text-fg-secondary
                   bg-panel/50 hover:bg-elevated/50 transition-colors"
      >
        <ChevronRight className={clsx("w-3.5 h-3.5 text-fg-muted transition-transform", open && "rotate-90")} />
        {title}
      </button>
      {open && (
        <div className="px-3 py-3 border-t border-border max-h-[400px] overflow-y-auto">
          {children}
        </div>
      )}
    </div>
  );
}

function Pre({ children, className }) {
  return (
    <pre className={clsx("font-mono text-xs text-fg-secondary leading-relaxed whitespace-pre-wrap break-words", className)}>
      {children}
    </pre>
  );
}

function UsageGrid({ usage }) {
  if (!usage) return null;
  const items = [
    ["Input Tokens", usage.input_tokens],
    ["Output Tokens", usage.output_tokens],
    ...(usage.total_tokens != null ? [["Total Tokens", usage.total_tokens]] : []),
  ];
  return (
    <div className="grid grid-cols-2 gap-2 mt-3">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-lg border border-border bg-canvas px-3 py-2">
          <div className="text-[10px] font-medium text-fg-muted uppercase tracking-wider">{label}</div>
          <div className="text-lg font-semibold tabular-nums">{fmt(value)}</div>
        </div>
      ))}
    </div>
  );
}

function MessageItem({ msg }) {
  const role = msg.role || "unknown";
  const roleColors = {
    system: "text-warning",
    user: "text-accent",
    assistant: "text-purple",
    tool: "text-success",
  };

  let content = "";
  if (typeof msg.content === "string") {
    content = msg.content;
  } else if (Array.isArray(msg.content)) {
    content = msg.content
      .map((c) => (typeof c === "string" ? c : c.type === "text" ? c.text || "" : JSON.stringify(c, null, 2)))
      .join("\n");
  } else if (msg.content != null) {
    content = JSON.stringify(msg.content, null, 2);
  }

  return (
    <div className="rounded-lg border border-border bg-canvas p-2.5 mb-2 last:mb-0">
      <div className={clsx("text-[10px] font-semibold uppercase tracking-wider mb-1", roleColors[role] || "text-fg-muted")}>
        {role}
      </div>
      <div className="font-mono text-xs text-fg-secondary leading-relaxed whitespace-pre-wrap break-words max-h-48 overflow-y-auto">
        {truncate(content)}
      </div>
    </div>
  );
}

function Markdown({ text }) {
  const html = useMemo(() => {
    try { return marked.parse(text); }
    catch { return null; }
  }, [text]);
  if (!html) return <Pre>{text}</Pre>;
  return (
    <div
      className="prose-sm text-sm text-fg-secondary leading-relaxed break-words
                 [&_p]:my-1.5 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5
                 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-fg [&_h1]:mt-3 [&_h1]:mb-1
                 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-fg [&_h2]:mt-2.5 [&_h2]:mb-1
                 [&_h3]:text-sm [&_h3]:font-medium [&_h3]:text-fg [&_h3]:mt-2 [&_h3]:mb-0.5
                 [&_strong]:text-fg [&_em]:text-fg-secondary
                 [&_code]:font-mono [&_code]:text-xs [&_code]:bg-panel [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded
                 [&_a]:text-accent [&_a]:underline"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function ResponseSection({ output }) {
  if (!output || !Array.isArray(output)) return <Pre>{prettyJson(output)}</Pre>;
  const msg = output[0];
  if (!msg) return null;
  const reasoning = msg.reasoning_content;
  const content = msg.content;
  const toolCalls = msg.tool_calls;
  return (
    <div className="space-y-2">
      {reasoning && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-purple mb-1">Thinking</div>
          <div className="rounded-lg border border-purple/20 bg-purple-dimmed/50 px-3 py-2">
            <Markdown text={reasoning} />
          </div>
        </div>
      )}
      {content && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-fg-muted mb-1">Content</div>
          <Pre>{content}</Pre>
        </div>
      )}
      {toolCalls && toolCalls.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-success mb-1">Tool Calls</div>
          <div className="rounded-lg border border-success/20 bg-success-dimmed/50 px-3 py-2">
            <Pre>{prettyJson(toolCalls)}</Pre>
          </div>
        </div>
      )}
    </div>
  );
}

function RateLimitGrid({ rl }) {
  if (!rl) return null;
  const items = [
    ["Acquire Wait", fmtDuration(rl.acquire_wait_ms)],
    ["Retries", fmt(rl.retry_count)],
    ["TPM Delta", rl.tpm_delta > 0 ? `+${fmt(rl.tpm_delta)}` : fmt(rl.tpm_delta)],
    ...(rl.exhausted ? [["Exhausted", "Yes"]] : []),
    ...(rl.error ? [["Error", rl.error]] : []),
  ];
  return (
    <div className="grid grid-cols-2 gap-2 mt-3">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-lg border border-warning/30 bg-warning-dimmed/30 px-3 py-2">
          <div className="text-[10px] font-medium text-warning uppercase tracking-wider">{label}</div>
          <div className="text-lg font-semibold tabular-nums">{value}</div>
        </div>
      ))}
    </div>
  );
}

function GenerationSection({ attrs }) {
  return (
    <>
      <UsageGrid usage={attrs.usage} />
      <RateLimitGrid rl={attrs.usage?.rate_limit} />
      {attrs.model && (
        <div className="mt-3 text-sm">
          <span className="text-fg-muted">Model: </span>
          <span className="font-mono text-accent">{attrs.model}</span>
        </div>
      )}
      {attrs.input && Array.isArray(attrs.input) && (
        <Collapsible title={`Prompt Messages (${attrs.input.length})`}>
          {attrs.input.map((msg, i) => (
            <MessageItem key={i} msg={msg} />
          ))}
        </Collapsible>
      )}
      {attrs.output && (
        <Collapsible title="Response" defaultOpen>
          <ResponseSection output={attrs.output} />
        </Collapsible>
      )}
      {attrs.model_config && (
        <Collapsible title="Model Config">
          <Pre>{prettyJson(attrs.model_config)}</Pre>
        </Collapsible>
      )}
    </>
  );
}

function parseInputCode(name, raw) {
  if (name !== "execute_python" || raw == null) return null;
  try {
    const obj = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (typeof obj?.code === "string") return obj.code;
  } catch { /* not JSON */ }
  return null;
}

function CodeBlock({ code, language = "python" }) {
  const html = useMemo(() => {
    try {
      return hljs.highlight(code, { language }).value;
    } catch {
      return null;
    }
  }, [code, language]);

  if (!html) {
    return (
      <pre className="font-mono text-xs leading-relaxed whitespace-pre-wrap break-words
                      bg-canvas border border-border rounded-lg p-3 overflow-x-auto text-fg-secondary">
        {code}
      </pre>
    );
  }

  return (
    <pre
      className="hljs font-mono text-xs leading-relaxed whitespace-pre-wrap break-words
                 bg-canvas border border-border rounded-lg p-3 overflow-x-auto"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function parsePythonOutput(raw) {
  if (raw == null) return null;
  try {
    const obj = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (typeof obj?.success !== "boolean") return null;
    return obj;
  } catch { return null; }
}

function PythonOutputSection({ parsed }) {
  return (
    <div className="space-y-2">
      {parsed.success ? (
        <>
          {parsed.output && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-success mb-1">stdout</div>
              <div className="rounded-lg border border-success/20 bg-success-dimmed/50 px-3 py-2">
                <Pre>{parsed.output}</Pre>
              </div>
            </div>
          )}
          {parsed.stderr && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-warning mb-1">stderr</div>
              <div className="rounded-lg border border-warning/20 bg-warning-dimmed/50 px-3 py-2">
                <Pre>{parsed.stderr}</Pre>
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {parsed.error && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-danger mb-1">Error</div>
              <div className="rounded-lg border border-danger/30 bg-danger-dimmed px-3 py-2">
                <Pre className="!text-danger">{parsed.error}</Pre>
              </div>
            </div>
          )}
          {parsed.traceback && (
            <Collapsible title="Traceback">
              <Pre className="!text-danger">{parsed.traceback}</Pre>
            </Collapsible>
          )}
        </>
      )}
    </div>
  );
}

function parseFileList(name, raw) {
  if (name !== "inspect_files" || raw == null) return null;
  try {
    const obj = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!Array.isArray(obj?.files)) return null;
    return obj.files;
  } catch { return null; }
}

function fmtBytes(n) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}B`;
}

const formatColors = {
  csv: "text-success",
  sqlite: "text-accent",
  json: "text-warning",
  markdown: "text-purple",
  pdf: "text-danger",
  mp4: "text-danger",
};

function ColumnsTable({ columns, dtypes, profile }) {
  return (
    <table className="w-full text-xs font-mono mt-1">
      <thead>
        <tr className="text-fg-muted text-left border-b border-border">
          <th className="py-1 pr-2 font-medium">Column</th>
          <th className="py-1 pr-2 font-medium">Type</th>
          <th className="py-1 pr-2 font-medium text-right">Nulls</th>
          <th className="py-1 font-medium">Stats</th>
        </tr>
      </thead>
      <tbody>
        {columns.map((col, i) => {
          const p = profile?.[i] || {};
          const dtype = dtypes?.[i] || "–";
          const nullPct = p.null_rate != null ? `${(p.null_rate * 100).toFixed(0)}%` : "–";
          let stats = "";
          if (p.min != null && p.max != null) stats = `${p.min} ~ ${p.max}`;
          else if (p.unique_count != null) stats = `${p.unique_count} unique`;
          if (p.sample_values?.length) stats += (stats ? " · " : "") + p.sample_values.slice(0, 3).join(", ");
          if (p.values?.length) stats += (stats ? " · " : "") + p.values.join(", ");
          return (
            <tr key={col} className="border-b border-border/30">
              <td className="py-0.5 pr-2 text-fg-secondary">{col}</td>
              <td className="py-0.5 pr-2 text-accent">{dtype}</td>
              <td className="py-0.5 pr-2 text-right tabular-nums text-fg-muted">{nullPct}</td>
              <td className="py-0.5 text-fg-muted truncate max-w-[200px]" title={stats}>{stats || "–"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function FileSchemaDetail({ schema, format }) {
  if (!schema) return null;

  if (schema.columns?.length) {
    return <ColumnsTable columns={schema.columns} dtypes={schema.dtypes} profile={schema.profile} />;
  }

  if (format === "sqlite" && schema.tables) {
    return (
      <div className="space-y-3 mt-1">
        {Object.entries(schema.tables).map(([table, info]) => (
          <div key={table}>
            <div className="text-xs font-semibold text-accent mb-1">{table}</div>
            {info.columns?.length > 0 && (
              <ColumnsTable columns={info.columns} dtypes={info.dtypes} profile={info.profile} />
            )}
          </div>
        ))}
      </div>
    );
  }

  if (format === "markdown") {
    return (
      <div className="text-xs font-mono mt-1 space-y-1">
        {schema.line_count != null && (
          <div><span className="text-fg-muted">Lines: </span><span className="tabular-nums">{schema.line_count}</span></div>
        )}
        {schema.head?.length > 0 && (
          <div className="rounded border border-border bg-canvas p-2 text-fg-secondary whitespace-pre-wrap">
            {schema.head.join("\n")}
          </div>
        )}
      </div>
    );
  }

  if (format === "json") {
    const items = [
      schema.kind && ["Kind", schema.kind],
      schema.key_count != null && ["Keys", `${schema.key_count} (${schema.keys?.join(", ")})`],
    ].filter(Boolean);
    if (!items.length) return null;
    return (
      <div className="text-xs font-mono mt-1 space-y-0.5">
        {items.map(([label, val]) => (
          <div key={label}><span className="text-fg-muted">{label}: </span><span className="text-fg-secondary">{val}</span></div>
        ))}
      </div>
    );
  }

  return null;
}

function hasFileDetail(f) {
  const s = f.schema;
  if (!s) return false;
  if (s.columns?.length) return true;
  if (f.format === "sqlite" && s.tables) return true;
  if (f.format === "markdown" && (s.line_count != null || s.head)) return true;
  if (f.format === "json" && (s.kind || s.keys)) return true;
  return false;
}

function FileListSection({ files }) {
  const [expanded, setExpanded] = useState(null);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="text-fg-muted text-left border-b border-border">
            <th className="py-1.5 pr-3 font-medium">Path</th>
            <th className="py-1.5 pr-3 font-medium text-right">Size</th>
            <th className="py-1.5 pr-3 font-medium">Format</th>
            <th className="py-1.5 font-medium text-right">Rows</th>
          </tr>
        </thead>
        <tbody>
          {files.map((f) => {
            const expandable = hasFileDetail(f);
            const isOpen = expanded === f.path;
            return (
              <Fragment key={f.path}>
                <tr
                  className={clsx(
                    "border-b border-border/50 transition-colors",
                    expandable ? "cursor-pointer hover:bg-panel/50" : "",
                    isOpen && "bg-panel/50"
                  )}
                  onClick={() => expandable && setExpanded(isOpen ? null : f.path)}
                >
                  <td className="py-1 pr-3 text-fg-secondary">
                    {expandable && (
                      <ChevronRight className={clsx("inline w-3 h-3 mr-1 text-fg-muted transition-transform", isOpen && "rotate-90")} />
                    )}
                    {f.path}
                  </td>
                  <td className="py-1 pr-3 text-right tabular-nums text-fg-muted">{fmtBytes(f.size)}</td>
                  <td className={clsx("py-1 pr-3", formatColors[f.format] || "text-fg-muted")}>{f.format}</td>
                  <td className="py-1 text-right tabular-nums text-fg-muted">{f.schema?.row_count ?? "–"}</td>
                </tr>
                {isOpen && (
                  <tr>
                    <td colSpan={4} className="px-4 py-2 bg-elevated/30 border-b border-border/50">
                      <FileSchemaDetail schema={f.schema} format={f.format} />
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function FunctionSection({ attrs }) {
  const code = parseInputCode(attrs.name, attrs.input);
  const pyOutput = attrs.name === "execute_python" ? parsePythonOutput(attrs.output) : null;
  const fileList = parseFileList(attrs.name, attrs.output);
  return (
    <>
      {attrs.name && (
        <div className="mt-2 text-sm">
          <span className="text-fg-muted">Function: </span>
          <span className="font-mono text-success">{attrs.name}</span>
        </div>
      )}
      {code != null ? (
        <Collapsible title="Code" defaultOpen>
          <CodeBlock code={code} />
        </Collapsible>
      ) : attrs.input != null ? (
        <Collapsible title="Input" defaultOpen>
          <Pre>{prettyJson(attrs.input)}</Pre>
        </Collapsible>
      ) : null}
      {fileList ? (
        <Collapsible title={`Files (${fileList.length})`} defaultOpen>
          <FileListSection files={fileList} />
        </Collapsible>
      ) : pyOutput ? (
        <Collapsible title={pyOutput.success ? "Output" : "Output (Error)"} defaultOpen>
          <PythonOutputSection parsed={pyOutput} />
        </Collapsible>
      ) : attrs.output != null ? (
        <Collapsible title="Output" defaultOpen>
          <Pre>{prettyJson(attrs.output)}</Pre>
        </Collapsible>
      ) : null}
    </>
  );
}

function ErrorSection({ attrs }) {
  return (
    <>
      {attrs.error_type && (
        <div className="mt-2 text-sm">
          <span className="text-fg-muted">Error Type: </span>
          <span className="text-danger">{attrs.error_type}</span>
        </div>
      )}
      {attrs.message && (
        <div className="mt-2 rounded-lg border border-danger/30 bg-danger-dimmed px-3 py-2.5">
          <Pre className="!text-danger">{attrs.message}</Pre>
        </div>
      )}
      {attrs.error && (
        <Collapsible title="Error Details" defaultOpen>
          <Pre className="!text-danger">{prettyJson(attrs.error)}</Pre>
        </Collapsible>
      )}
      {attrs.data != null && (
        <Collapsible title="Error Data">
          <Pre>{prettyJson(attrs.data)}</Pre>
        </Collapsible>
      )}
    </>
  );
}

function AgentSection({ attrs }) {
  return (
    <>
      {attrs.protocol && (
        <div className="mt-2 text-sm">
          <span className="text-fg-muted">Protocol: </span>
          <span className="font-mono">{attrs.protocol}</span>
        </div>
      )}
      {attrs.max_steps != null && (
        <div className="mt-1 text-sm">
          <span className="text-fg-muted">Max Steps: </span>
          <span className="tabular-nums">{attrs.max_steps}</span>
        </div>
      )}
      {attrs.tools?.length > 0 && (
        <div className="mt-1 text-sm">
          <span className="text-fg-muted">Tools: </span>
          <span className="font-mono text-xs">{attrs.tools.join(", ")}</span>
        </div>
      )}
      {attrs.question && (
        <Collapsible title="Question" defaultOpen>
          <Pre>{attrs.question}</Pre>
        </Collapsible>
      )}
    </>
  );
}

function TurnSection({ attrs }) {
  return (
    <>
      {attrs.turn != null && (
        <div className="mt-2 text-sm">
          <span className="text-fg-muted">Turn: </span>
          <span className="tabular-nums">{attrs.turn}</span>
        </div>
      )}
      {attrs.agent_name && (
        <div className="mt-1 text-sm">
          <span className="text-fg-muted">Agent: </span>
          <span className="font-mono">{attrs.agent_name}</span>
        </div>
      )}
      <UsageGrid usage={attrs.usage} />
    </>
  );
}

const SECTION_MAP = {
  generation: GenerationSection,
  function: FunctionSection,
  error: ErrorSection,
  agent: AgentSection,
  turn: TurnSection,
};

export default function Inspector({ span }) {
  if (!span) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-16 text-center px-6">
        <div className="w-12 h-12 rounded-xl bg-panel flex items-center justify-center mb-3">
          <MousePointer2 className="w-5 h-5 text-fg-muted" />
        </div>
        <p className="text-fg-muted text-sm">Select a span from the timeline to inspect it.</p>
      </div>
    );
  }

  const attrs = span.attributes || {};
  const kc = kindClass(span.kind);
  const dur = spanDuration(span);
  const KindSection = SECTION_MAP[span.kind];

  const kindBadgeColors = {
    accent: "text-accent bg-accent-dimmed",
    success: "text-success bg-success-dimmed",
    danger: "text-danger bg-danger-dimmed",
    warning: "text-warning bg-warning-dimmed",
    purple: "text-purple bg-purple-dimmed",
    muted: "text-fg-muted bg-muted-bg",
  };

  return (
    <div className="p-4 animate-fade-up">
      {/* Header */}
      <div className="mb-4">
        <h3 className="flex items-center gap-2 text-base font-semibold mb-3">
          <span className={clsx("text-[10px] font-semibold uppercase tracking-wide px-1.5 py-px rounded", kindBadgeColors[kc])}>
            {span.kind}
          </span>
          {span.name}
        </h3>
        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-sm">
          <span className="text-fg-muted">Status</span>
          <span><StatusBadge status={span.status} /></span>
          <span className="text-fg-muted">Duration</span>
          <span className="tabular-nums">{fmtDuration(dur)}</span>
          <span className="text-fg-muted">Started</span>
          <span>{fmtTime(span.started_at)}</span>
          <span className="text-fg-muted">Ended</span>
          <span>{fmtTime(span.ended_at)}</span>
          <span className="text-fg-muted">Span ID</span>
          <span className="font-mono text-xs text-fg-secondary">{span.span_id}</span>
          {span.parent_span_id && (
            <>
              <span className="text-fg-muted">Parent</span>
              <span className="font-mono text-xs text-fg-secondary">{span.parent_span_id}</span>
            </>
          )}
        </div>
      </div>

      {/* Kind-specific section */}
      {KindSection && <KindSection attrs={attrs} />}

      {/* Raw attributes */}
      <Collapsible title="Raw Attributes">
        <Pre>{prettyJson(attrs)}</Pre>
      </Collapsible>
    </div>
  );
}
