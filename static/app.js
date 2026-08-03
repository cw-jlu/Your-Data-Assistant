const state = {
  engines: [],
  selected: new Set(),
  runs: [],
  workspace: null,
  uploadCapabilities: null,
  uploading: new Map(),
  turns: [],
  resultCache: new Map(),
  activeRun: null,
  activeTab: "result",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const LLM_SETTINGS_KEY = "data-agent.llm-settings.v1";

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (typeof options.body === "string") headers["Content-Type"] = "application/json";
  const response = await fetch(path, { ...options, headers });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[char]);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

function toast(message, detail = "", error = false) {
  const el = document.createElement("div");
  el.className = `toast${error ? " error" : ""}`;
  el.innerHTML = `<strong>${escapeHtml(message)}</strong>${detail ? `<small>${escapeHtml(detail)}</small>` : ""}`;
  $("#toast-stack").appendChild(el);
  setTimeout(() => el.remove(), 4300);
}

function updateLlmSummary() {
  const apiBase = $("#api-base").value.trim();
  const model = $("#model-name").value.trim();
  const configured = Boolean(apiBase && model);
  $("#model-summary").textContent = model || "LLM 未配置";
  $("#llm-config-state").textContent = configured ? "已配置" : "未配置";
  $("#global-settings-trigger").classList.toggle("configured", configured);
}

function loadLlmSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(LLM_SETTINGS_KEY) || "{}");
  } catch (_error) {
    localStorage.removeItem(LLM_SETTINGS_KEY);
  }
  const fields = {
    "api-base": saved.apiBase,
    "model-name": saved.model,
    "max-workers": saved.maxWorkers,
    "max-steps": saved.maxSteps,
    timeout: saved.timeout,
    experiment: saved.experiment,
  };
  Object.entries(fields).forEach(([id, value]) => {
    if (value !== undefined && value !== null) $(`#${id}`).value = value;
  });
  // API Key is deliberately never restored from localStorage or SQLite.
  $("#api-key").value = "";
  updateLlmSummary();
}

function validateLlmSettings() {
  const apiBase = $("#api-base").value.trim();
  const model = $("#model-name").value.trim();
  if (!apiBase) return "请填写 LLM API URL";
  if (!model) return "请填写模型名称";
  try {
    const url = new URL(apiBase);
    if (!["http:", "https:"].includes(url.protocol)) return "LLM API URL 必须使用 HTTP 或 HTTPS";
  } catch (_error) {
    return "LLM API URL 格式不正确";
  }
  return "";
}

function openSettings() {
  const upload = $("#upload-popover");
  const popover = $("#settings-popover");
  const backdrop = $("#settings-backdrop");
  const apiBase = $("#api-base");
  if (!popover || !backdrop || !apiBase) {
    toast("无法打开 LLM 设置", "设置界面加载不完整，请重启客户端。", true);
    return;
  }
  upload?.classList.remove("open");
  popover.classList.add("open");
  backdrop.classList.add("open");
  requestAnimationFrame(() => apiBase.focus({ preventScroll: true }));
}

function closeSettings() {
  $("#settings-popover").classList.remove("open");
  $("#settings-backdrop").classList.remove("open");
}

function saveLlmSettings() {
  const error = validateLlmSettings();
  if (error) {
    toast("LLM 设置未完成", error, true);
    return;
  }
  localStorage.setItem(LLM_SETTINGS_KEY, JSON.stringify({
    apiBase: $("#api-base").value.trim(),
    model: $("#model-name").value.trim(),
    maxWorkers: Number($("#max-workers").value),
    maxSteps: Number($("#max-steps").value),
    timeout: Number($("#timeout").value),
    experiment: $("#experiment").value.trim(),
  }));
  updateLlmSummary();
  closeSettings();
  toast("LLM 设置已保存", "API Key 仅在当前应用进程中保留");
}

function clearLlmSettings() {
  localStorage.removeItem(LLM_SETTINGS_KEY);
  $("#api-base").value = "";
  $("#model-name").value = "";
  $("#api-key").value = "";
  $("#max-workers").value = "2";
  $("#max-steps").value = "16";
  $("#timeout").value = "900";
  $("#experiment").value = "exp_154_v1_audio_asr";
  updateLlmSummary();
  toast("LLM 设置已清除", "没有保存任何 API Key");
}

function switchView(name) {
  $$(".view").forEach((el) => el.classList.toggle("active", el.id === `view-${name}`));
  $$(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.view === name));
  $("#page-title").textContent = {
    workspace: "分析工作区",
    runs: "运行记录",
    architecture: "融合架构",
  }[name];
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadEngines() {
  const { engines } = await api("/api/engines");
  state.engines = engines;
  $("#ready-engine-count").textContent = engines.filter((engine) => engine.ready).length;
  $("#engine-grid").innerHTML = engines.map((engine, index) => `
    <button class="engine-card ${engine.ready ? "" : "unready"}"
      type="button" aria-pressed="false" ${engine.ready ? "" : "disabled"}
      style="--accent:${engine.accent}" data-engine="${engine.id}">
      <span class="engine-index">0${index + 1}</span>
      <small>${escapeHtml(engine.subtitle)}</small>
      <h4>${escapeHtml(engine.name)}</h4>
      <p>${escapeHtml(engine.description)}</p>
      <div class="engine-tags">
        ${engine.strengths.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
      </div>
    </button>
  `).join("");
  $$(".engine-card").forEach((card) => {
    card.addEventListener("click", () => toggleEngine(card.dataset.engine));
  });
  const preferred = engines.find((engine) => engine.id === "mamba" && engine.ready)
    || engines.find((engine) => engine.ready);
  if (preferred) state.selected.add(preferred.id);
  renderSelection();
}

function toggleEngine(engineId) {
  const engine = state.engines.find((item) => item.id === engineId);
  if (!engine?.ready) {
    toast("引擎未就绪", "请确认源码与 uv 环境可用。", true);
    return;
  }
  state.selected.has(engineId) ? state.selected.delete(engineId) : state.selected.add(engineId);
  renderSelection();
}

function renderSelection() {
  $$(".engine-card").forEach((card) => {
    const selected = state.selected.has(card.dataset.engine);
    card.classList.toggle("selected", selected);
    card.setAttribute("aria-pressed", String(selected));
  });
  const selected = state.engines.filter((engine) => state.selected.has(engine.id));
  $("#engine-selection-mini").innerHTML = selected.length
    ? selected.map((engine) => `<span style="--engine:${engine.accent}">${escapeHtml(engine.name)}</span>`).join("")
    : "未选择引擎";
  $("#experiment-field").style.display = state.selected.has("kobushi") ? "grid" : "none";
  renderUploadCapabilities();
  renderFiles();
}

async function loadUploadCapabilities() {
  state.uploadCapabilities = await api("/api/upload-capabilities");
  renderUploadCapabilities();
}

function currentSupportedExtensions() {
  if (!state.uploadCapabilities || !state.selected.size) return new Set();
  const selectedSets = [...state.selected].map(
    (engineId) => new Set(state.uploadCapabilities.engine_support[engineId]?.extensions || []),
  );
  const intersection = new Set(selectedSets[0]);
  for (const supported of selectedSets.slice(1)) {
    [...intersection].forEach((extension) => {
      if (!supported.has(extension)) intersection.delete(extension);
    });
  }
  return intersection;
}

function fileExtension(name) {
  const index = name.lastIndexOf(".");
  return index >= 0 ? name.slice(index).toLowerCase() : "";
}

function renderUploadCapabilities() {
  if (!state.uploadCapabilities || !state.engines.length) return;
  const supported = currentSupportedExtensions();
  const selectedNames = state.engines
    .filter((engine) => state.selected.has(engine.id))
    .map((engine) => engine.name);
  $("#file-input").accept = [...supported].sort().join(",");
  $("#compatibility-summary").innerHTML = selectedNames.length
    ? `<strong>${escapeHtml(selectedNames.join(" + "))}</strong><span>共同原生支持 ${supported.size} 种扩展名</span>`
    : `<strong>尚未选择引擎</strong><span>选择引擎后显示可用格式</span>`;
  $("#file-capabilities").innerHTML = state.uploadCapabilities.groups.map((group) => {
    const extensions = group.extensions.filter((extension) => supported.has(extension));
    if (!extensions.length) return "";
    return `<div class="capability-group">
      <strong>${escapeHtml(group.name)}</strong>
      <span>${extensions.map((extension) => escapeHtml(extension)).join(" · ")}</span>
    </div>`;
  }).join("") || `<div class="trace-empty">当前引擎组合没有共同支持的文件类型。</div>`;
  $("#engine-file-matrix").innerHTML = state.engines.map((engine) => {
    const support = state.uploadCapabilities.engine_support[engine.id];
    return `<div class="matrix-engine">
      <strong style="--matrix-accent:${engine.accent}">${escapeHtml(engine.name)}</strong>
      <span>${support.extensions.map((extension) => escapeHtml(extension)).join(" · ")}</span>
      ${support.notes.map((note) => `<small>${escapeHtml(note)}</small>`).join("")}
    </div>`;
  }).join("");
}

async function ensureWorkspace() {
  if (!state.workspace) {
    state.workspace = await api("/api/workspaces", { method: "POST", body: "{}" });
  }
  return state.workspace;
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  const supported = currentSupportedExtensions();
  const incompatible = files.filter((file) => !supported.has(fileExtension(file.name)));
  if (incompatible.length) {
    toast(
      "所选引擎不能共同处理这些文件",
      incompatible.map((file) => file.name).join("、"),
      true,
    );
    $("#file-input").value = "";
    return;
  }
  $("#upload-popover").classList.remove("open");
  try {
    const workspace = await ensureWorkspace();
    for (const file of files) {
      const uploadKey = `${file.name}-${file.size}-${Date.now()}`;
      state.uploading.set(uploadKey, { name: file.name, size: file.size });
      renderFiles();
      try {
        const response = await fetch(
          `/api/workspaces/${encodeURIComponent(workspace.id)}/files?name=${encodeURIComponent(file.name)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/octet-stream" },
            body: file,
          },
        );
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      } finally {
        state.uploading.delete(uploadKey);
      }
      state.workspace = await api(`/api/workspaces/${encodeURIComponent(workspace.id)}`);
      renderFiles();
    }
    toast("上传完成", `${files.length} 个文件已加入本地工作区`);
  } catch (error) {
    toast("上传失败", error.message, true);
  } finally {
    $("#file-input").value = "";
    renderFiles();
  }
}

async function removeFile(name) {
  if (!state.workspace) return;
  try {
    state.workspace = await api(
      `/api/workspaces/${encodeURIComponent(state.workspace.id)}/files?name=${encodeURIComponent(name)}`,
      { method: "DELETE" },
    );
    renderFiles();
  } catch (error) {
    toast("删除失败", error.message, true);
  }
}

function renderFiles() {
  const uploaded = state.workspace?.files || [];
  const uploading = [...state.uploading.values()];
  const supported = currentSupportedExtensions();
  $("#uploaded-files").innerHTML = [
    ...uploaded.map((file) => `
      <div class="file-chip ${supported.has(file.extension) ? "" : "incompatible"}">
        <span class="file-type">${escapeHtml(file.extension.replace(".", "").slice(0, 4) || "FILE")}</span>
        <span><strong>${escapeHtml(file.name)}</strong><small>${supported.has(file.extension) ? formatBytes(file.size) : "与当前引擎不兼容"}</small></span>
        <button data-remove-file="${escapeHtml(file.name)}" title="移除">×</button>
      </div>
    `),
    ...uploading.map((file) => `
      <div class="file-chip uploading">
        <span class="file-type">···</span>
        <span><strong>${escapeHtml(file.name)}</strong><small>上传中 · ${formatBytes(file.size)}</small></span>
      </div>
    `),
  ].join("");
  $$("[data-remove-file]").forEach((button) => {
    button.addEventListener("click", () => removeFile(button.dataset.removeFile));
  });
}

function runPayload(query) {
  return {
    engine_ids: [...state.selected],
    workspace_id: state.workspace?.id,
    query,
    api_base: $("#api-base").value.trim(),
    model: $("#model-name").value.trim(),
    api_key: $("#api-key").value,
    max_workers: Number($("#max-workers").value),
    max_steps: Number($("#max-steps").value),
    timeout: Number($("#timeout").value),
    experiment: $("#experiment").value.trim(),
  };
}

async function launch() {
  const query = $("#query-input").value.trim();
  const settingsError = validateLlmSettings();
  if (settingsError) {
    openSettings();
    return toast("请先配置 LLM", settingsError, true);
  }
  if (!state.selected.size) return toast("请选择至少一个引擎", "", true);
  if (!state.workspace?.files?.length) return toast("请先上传资料", "点击输入框下方的＋查看支持格式。", true);
  if (!query) return toast("请输入 Query", "", true);
  if (state.uploading.size) return toast("文件仍在上传", "请等待上传完成。", true);
  const supported = currentSupportedExtensions();
  const incompatible = state.workspace.files.filter((file) => !supported.has(file.extension));
  if (incompatible.length) {
    return toast(
      "当前引擎组合无法处理已上传文件",
      `请移除 ${incompatible.map((file) => file.name).join("、")}，或调整引擎选择。`,
      true,
    );
  }

  const button = $("#launch-button");
  button.disabled = true;
  try {
    const fileSnapshot = state.workspace.files.map((file) => ({ ...file }));
    const workspaceId = state.workspace.id;
    const response = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(runPayload(query)),
    });
    state.turns.push({
      query,
      files: fileSnapshot,
      workspaceId,
      runIds: response.runs.map((run) => run.id),
      createdAt: new Date().toISOString(),
    });
    $("#query-input").value = "";
    state.workspace = null;
    renderFiles();
    $("#welcome-block").classList.add("conversation-started");
    await refreshRuns();
    renderConversation();
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  } catch (error) {
    toast("启动失败", error.message, true);
  } finally {
    button.disabled = false;
  }
}

const statusLabel = {
  queued: "排队中",
  running: "分析中",
  succeeded: "已完成",
  failed: "失败",
  stopped: "已停止",
  interrupted: "已中断",
};
const engineAccent = (id) => state.engines.find((engine) => engine.id === id)?.accent || "#5eead4";

function renderConversation() {
  const feed = $("#conversation-feed");
  feed.innerHTML = state.turns.map((turn) => {
    const runs = turn.runIds.map((id) => state.runs.find((run) => run.id === id)).filter(Boolean);
    return `
      <section class="conversation-turn">
        <div class="user-message">
          <div class="message-label">YOU</div>
          <p>${escapeHtml(turn.query).replace(/\n/g, "<br>")}</p>
          <div class="message-files">
            ${turn.files.map((file) => `<span>${escapeHtml(file.name)} · ${formatBytes(file.size)}</span>`).join("")}
          </div>
        </div>
        <div class="agent-responses">
          ${runs.map((run) => renderAgentResponse(run)).join("")}
        </div>
      </section>
    `;
  }).join("");
  $$("[data-open-run]").forEach((button) => {
    button.addEventListener("click", () => openRun(button.dataset.openRun, button.dataset.openTab || "result"));
  });
  hydrateCompletedResults();
}

function renderAgentResponse(run) {
  const cached = state.resultCache.get(run.id);
  let content = "";
  if (run.status === "succeeded" && cached) {
    content = renderInlineResult(cached);
  } else if (["failed", "stopped", "interrupted"].includes(run.status)) {
    const tail = (run.log_tail || []).slice(-5).join("\n");
    content = `<pre class="inline-error">${escapeHtml(run.error || tail || "引擎未生成结果")}</pre>`;
  } else {
    const tail = (run.log_tail || []).slice(-4).join("\n");
    content = `
      <div class="thinking-row"><i></i><span>${run.status === "queued" ? "等待启动" : "正在分析上传的资料"}</span></div>
      ${tail ? `<pre class="inline-log">${escapeHtml(tail)}</pre>` : ""}
    `;
  }
  return `
    <article class="agent-message" style="--accent:${engineAccent(run.engine_id)}">
      <header>
        <div><span class="agent-orb"></span><strong>${escapeHtml(run.engine_name)}</strong></div>
        <span class="status-pill ${run.status}">${statusLabel[run.status] || run.status}</span>
      </header>
      <div class="agent-content">${content}</div>
      <footer>
        <button data-open-run="${run.id}" data-open-tab="result">查看输出</button>
        <button data-open-run="${run.id}" data-open-tab="trace">完整 Trace</button>
      </footer>
    </article>
  `;
}

function renderInlineResult(payload) {
  if (!payload?.previews?.length) {
    return `<p class="no-result">运行成功，但没有找到 prediction.csv。</p>`;
  }
  return payload.previews.slice(0, 1).map((preview) => {
    if (preview.error) return `<p class="no-result">${escapeHtml(preview.error)}</p>`;
    const head = `<tr>${preview.columns.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr>`;
    const rows = preview.rows.slice(0, 8).map(
      (row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`,
    ).join("");
    return `<div class="answer-label">ANSWER · ${escapeHtml(preview.task_id)}</div>
      <div class="table-scroll inline-table"><table>${head}${rows}</table></div>`;
  }).join("");
}

async function hydrateCompletedResults() {
  const missing = state.runs.filter(
    (run) => run.status === "succeeded"
      && state.turns.some((turn) => turn.runIds.includes(run.id))
      && !state.resultCache.has(run.id),
  );
  await Promise.all(missing.map(async (run) => {
    try {
      state.resultCache.set(run.id, await api(`/api/runs/${encodeURIComponent(run.id)}/preview`));
    } catch {
      state.resultCache.set(run.id, { previews: [] });
    }
  }));
  if (missing.length) {
    renderConversationWithoutHydration();
  }
}

function renderConversationWithoutHydration() {
  const feed = $("#conversation-feed");
  feed.innerHTML = state.turns.map((turn) => {
    const runs = turn.runIds.map((id) => state.runs.find((run) => run.id === id)).filter(Boolean);
    return `
      <section class="conversation-turn">
        <div class="user-message">
          <div class="message-label">YOU</div>
          <p>${escapeHtml(turn.query).replace(/\n/g, "<br>")}</p>
          <div class="message-files">${turn.files.map((file) => `<span>${escapeHtml(file.name)} · ${formatBytes(file.size)}</span>`).join("")}</div>
        </div>
        <div class="agent-responses">${runs.map((run) => renderAgentResponse(run)).join("")}</div>
      </section>`;
  }).join("");
  $$("[data-open-run]").forEach((button) => {
    button.addEventListener("click", () => openRun(button.dataset.openRun, button.dataset.openTab || "result"));
  });
}

function runCard(run) {
  const date = run.started_at || run.created_at;
  const time = date ? new Date(date).toLocaleString("zh-CN", { hour12: false }) : "—";
  return `
    <article class="run-card" data-run="${run.id}" style="--accent:${engineAccent(run.engine_id)}">
      <span class="run-accent"></span>
      <div class="run-main"><strong>${escapeHtml(run.engine_name)}</strong><small>${escapeHtml(run.id)}</small></div>
      <div class="run-cell"><small>任务</small><b>上传式 Query</b></div>
      <div class="run-cell"><small>启动时间</small><b>${escapeHtml(time)}</b></div>
      <span class="status-pill ${run.status}">${statusLabel[run.status] || run.status}</span>
    </article>
  `;
}

function renderRuns() {
  $("#run-count").textContent = state.runs.length;
  const empty = `<div class="empty-state"><span>◎</span><p>还没有运行</p><small>上传文件并提出问题后，记录会出现在这里</small></div>`;
  $("#all-runs").innerHTML = state.runs.length ? state.runs.map(runCard).join("") : empty;
  $$("#all-runs .run-card").forEach((card) => {
    card.addEventListener("click", () => openRun(card.dataset.run, "result"));
  });
}

async function refreshRuns() {
  try {
    const { runs } = await api("/api/runs");
    state.runs = runs;
    renderRuns();
    if (state.turns.length) renderConversation();
    if (state.activeRun) await refreshDrawer();
  } catch {
    $("#health-label").textContent = "离线";
  }
}

async function openRun(id, tab = "result") {
  state.activeRun = id;
  state.activeTab = tab;
  $("#run-drawer").classList.add("open");
  $("#drawer-backdrop").classList.add("open");
  setDrawerTab(tab);
  await refreshDrawer();
}

function closeDrawer() {
  state.activeRun = null;
  $("#run-drawer").classList.remove("open");
  $("#drawer-backdrop").classList.remove("open");
}

function setDrawerTab(tab) {
  state.activeTab = tab;
  $$(".drawer-tabs button").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $$(".drawer-pane").forEach((pane) => pane.classList.toggle("active", pane.id === `drawer-${tab}`));
}

async function refreshDrawer() {
  if (!state.activeRun) return;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(state.activeRun)}`);
    $("#drawer-title").textContent = run.engine_name;
    $("#drawer-meta").innerHTML = [
      statusLabel[run.status] || run.status,
      `退出码 ${run.exit_code ?? "—"}`,
      run.output_dir,
    ].map((item) => `<span>${escapeHtml(item)}</span>`).join("");
    $("#stop-run").classList.toggle("visible", run.status === "running");
    if (state.activeTab === "result") await loadPreview();
    if (state.activeTab === "trace") await loadTrace();
  } catch {
    closeDrawer();
  }
}

async function loadPreview() {
  const target = $("#result-output");
  target.innerHTML = `<div class="empty-state"><p>读取结果中…</p></div>`;
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/preview`);
    if (!payload.previews.length) {
      target.innerHTML = `<div class="empty-state"><span>◇</span><p>还没有 prediction.csv</p><small>结果生成后会自动显示</small></div>`;
      return;
    }
    target.innerHTML = payload.previews.map((preview) => {
      if (preview.error) return `<div class="result-block"><h4>${escapeHtml(preview.task_id)}</h4><p>${escapeHtml(preview.error)}</p></div>`;
      const head = `<tr>${preview.columns.map((cell) => `<th>${escapeHtml(cell)}</th>`).join("")}</tr>`;
      const body = preview.rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("");
      return `<div class="result-block"><h4>${escapeHtml(preview.task_id)}</h4>
        <div class="result-path">${escapeHtml(preview.path)}</div>
        <div class="table-scroll"><table>${head}${body}</table></div></div>`;
    }).join("");
  } catch (error) {
    target.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`;
  }
}

async function loadTrace() {
  const target = $("#trace-output");
  target.innerHTML = `<div class="empty-state"><p>读取 Trace 中…</p></div>`;
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(state.activeRun)}/trace`);
    const processLog = (payload.process_log || []).join("\n") || "等待进程日志…";
    const artifacts = (payload.artifacts || []).map((artifact, index) => `
      <details class="trace-artifact" ${index === 0 ? "open" : ""}>
        <summary>
          <span><b>${escapeHtml(artifact.name)}</b><small>${escapeHtml(artifact.kind)} · ${escapeHtml(artifact.path)}</small></span>
          <i>⌄</i>
        </summary>
        <pre>${escapeHtml(typeof artifact.content === "string" ? artifact.content : JSON.stringify(artifact.content, null, 2))}</pre>
      </details>
    `).join("");
    target.innerHTML = `
      <section class="trace-section">
        <div class="trace-heading"><strong>Runtime log</strong><span>${payload.process_log.length} lines</span></div>
        <pre class="trace-log">${escapeHtml(processLog)}</pre>
      </section>
      <section class="trace-section">
        <div class="trace-heading"><strong>Trace artifacts</strong><span>${payload.artifacts.length} files / databases</span></div>
        ${artifacts || `<div class="trace-empty">运行结束后，这里会显示 trace.json、summary.json 或 tracing.db。</div>`}
      </section>`;
    const pane = $("#drawer-trace");
    pane.scrollTop = pane.scrollHeight;
  } catch (error) {
    target.innerHTML = `<div class="empty-state"><p>${escapeHtml(error.message)}</p></div>`;
  }
}

async function stopRun() {
  if (!state.activeRun) return;
  try {
    await api(`/api/runs/${encodeURIComponent(state.activeRun)}/stop`, { method: "POST", body: "{}" });
    toast("已发送停止请求");
    await refreshRuns();
  } catch (error) {
    toast("停止失败", error.message, true);
  }
}

function togglePopover(targetId) {
  if (targetId === "#settings-popover") {
    openSettings();
    return;
  }
  const target = $(targetId);
  const willOpen = !target.classList.contains("open");
  $(".upload-popover").classList.remove("open");
  $(".settings-popover").classList.remove("open");
  target.classList.toggle("open", willOpen);
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.view)));
  $$("[data-go]").forEach((item) => item.addEventListener("click", () => switchView(item.dataset.go)));
  $("#refresh-button").addEventListener("click", refreshRuns);
  $("#select-all").addEventListener("click", () => {
    state.engines.filter((engine) => engine.ready).forEach((engine) => state.selected.add(engine.id));
    renderSelection();
  });
  $("#upload-trigger").addEventListener("click", (event) => {
    event.stopPropagation();
    togglePopover("#upload-popover");
  });
  ["#rail-upload-trigger", "#drop-target", "#format-guide-trigger"].forEach((selector) => {
    $(selector).addEventListener("click", (event) => {
      event.stopPropagation();
      togglePopover("#upload-popover");
    });
  });
  $("#settings-trigger").addEventListener("click", (event) => {
    event.stopPropagation();
    togglePopover("#settings-popover");
  });
  $("#model-settings-trigger").addEventListener("click", (event) => {
    event.stopPropagation();
    togglePopover("#settings-popover");
  });
  $("#global-settings-trigger").addEventListener("click", (event) => {
    event.stopPropagation();
    openSettings();
  });
  $("#settings-close").addEventListener("click", closeSettings);
  $("#settings-backdrop").addEventListener("click", closeSettings);
  $("#settings-save").addEventListener("click", saveLlmSettings);
  $("#settings-clear").addEventListener("click", clearLlmSettings);
  $("#toggle-api-key").addEventListener("click", () => {
    const input = $("#api-key");
    const visible = input.type === "text";
    input.type = visible ? "password" : "text";
    $("#toggle-api-key").textContent = visible ? "显示" : "隐藏";
  });
  ["#api-base", "#model-name"].forEach((selector) => {
    $(selector).addEventListener("input", updateLlmSummary);
  });
  $("#choose-files").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", (event) => uploadFiles(event.target.files));
  const dropTarget = $("#drop-target");
  ["dragenter", "dragover"].forEach((eventName) => {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropTarget.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropTarget.classList.remove("dragging");
    });
  });
  dropTarget.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));
  $("#launch-button").addEventListener("click", launch);
  $("#query-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      launch();
    }
  });
  $("#close-drawer").addEventListener("click", closeDrawer);
  $("#drawer-backdrop").addEventListener("click", closeDrawer);
  $("#stop-run").addEventListener("click", stopRun);
  $$(".drawer-tabs button").forEach((button) => button.addEventListener("click", async () => {
    setDrawerTab(button.dataset.tab);
    await refreshDrawer();
  }));
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".upload-popover") && !event.target.closest("#upload-trigger")) {
      $("#upload-popover").classList.remove("open");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    $("#upload-popover").classList.remove("open");
    closeSettings();
    if ($("#run-drawer").classList.contains("open")) closeDrawer();
  });
}

async function init() {
  loadLlmSettings();
  bindEvents();
  setInterval(() => {
    $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  }, 1000);
  try {
    await api("/api/health");
    $("#health-label").textContent = "正常";
    await Promise.all([loadEngines(), loadUploadCapabilities()]);
    await refreshRuns();
  } catch (error) {
    $("#health-label").textContent = "离线";
    toast("无法连接本地服务", error.message, true);
  }
  setInterval(refreshRuns, 1800);
}

init();
