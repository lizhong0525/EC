import { FormEvent, useEffect, useMemo, useState } from "react";
import { Download, RefreshCw } from "lucide-react";

import { BenchmarkLauncher } from "./BenchmarkLauncher";
import {
  claimProactiveMessage,
  createSession,
  deleteMemory,
  getBenchmarkReport,
  getBenchmarkRuns,
  generateBenchmarkReport,
  getMemories,
  getMessages,
  getModelSnapshot,
  getSession,
  getTrace,
  sendTurn,
  setProactivity,
} from "./api";
import type {
  BenchmarkReport,
  BenchmarkRun,
  ChatMessage,
  DecisionView,
  MemoryRecord,
  SessionView,
  ModelSnapshot,
  TraceRecord,
} from "./types";

const INITIAL_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text:
    "你好，我是EcoCompanion研究原型。我可以倾听并帮助整理感受，但不能诊断、治疗或替代现实中的专业支持。",
};

const SESSION_STORAGE_KEY = "ecocompanion.research_session_id";
type ActiveView = "companion" | "benchmark";

function percentage(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function displayMetric(value: unknown): string {
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (typeof value === "boolean") {
    return value ? "是" : "否";
  }
  return typeof value === "string" ? value : "—";
}

function reportColumns(
  report: BenchmarkReport | null,
): string[] {
  const firstRow = report?.report.rows[0];
  if (!firstRow) return [];
  return Object.entries(firstRow)
    .filter(([, value]) =>
      ["string", "number", "boolean"].includes(typeof value),
    )
    .map(([key]) => key);
}

function App() {
  const [activeView, setActiveView] = useState<ActiveView>("companion");
  const [session, setSession] = useState<SessionView | null>(null);
  const [modelSnapshot, setModelSnapshot] = useState<ModelSnapshot | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [decision, setDecision] = useState<DecisionView | null>(null);
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [traces, setTraces] = useState<TraceRecord[]>([]);
  const [busy, setBusy] = useState(false);
  const [proactivityBusy, setProactivityBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showResearch, setShowResearch] = useState(false);
  const [benchmarkRuns, setBenchmarkRuns] = useState<BenchmarkRun[]>([]);
  const [benchmarkReport, setBenchmarkReport] =
    useState<BenchmarkReport | null>(null);
  const [benchmarkBusy, setBenchmarkBusy] = useState(false);
  const modelCalls = decision?.model_calls ?? [];
  const auditCodes = decision?.audit_codes ?? [];

  useEffect(() => {
    void getModelSnapshot().then(setModelSnapshot).catch(() => undefined);
    async function restoreOrCreateSession(): Promise<void> {
      const storedSessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
      if (storedSessionId) {
        try {
          const [restoredSession, storedMessages] = await Promise.all([
            getSession(storedSessionId),
            getMessages(storedSessionId),
          ]);
          setSession(restoredSession);
          setMessages([
            INITIAL_MESSAGE,
            ...storedMessages.map((message) => ({
              id: message.message_id,
              role: message.role === "COMPANION" ? "assistant" as const : "user" as const,
              text: message.content,
            })),
          ]);
          return;
        } catch {
          window.localStorage.removeItem(SESSION_STORAGE_KEY);
        }
      }

      const created = await createSession();
      window.localStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      setSession(created);
    }

    void restoreOrCreateSession().catch((reason: unknown) => {
      setError(reason instanceof Error ? reason.message : "无法创建研究会话。");
    });
  }, []);

  const providerLabel = modelSnapshot?.active_model
    ? `${modelSnapshot.provider_location === "local" ? "本地模型" : "外部模型"} · ${modelSnapshot.active_model}`
    : "本地安全模板";

  const statusTone = useMemo(() => {
    const level = decision?.escalation_level ?? session?.state.risk_level ?? "L0";
    if (level === "L4") return "critical";
    if (level === "L3" || level === "L2") return "elevated";
    if (level === "B" || level === "L1") return "watch";
    return "calm";
  }, [decision, session]);

  async function refreshResearchPanels(sessionId: string): Promise<void> {
    const [nextMemories, nextTraces] = await Promise.all([
      getMemories(sessionId),
      getTrace(sessionId),
    ]);
    setMemories(nextMemories);
    setTraces(nextTraces);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || !session || busy) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text,
    };
    const assistantId = crypto.randomUUID();
    setMessages((current) => [
      ...current,
      userMessage,
      { id: assistantId, role: "assistant", text: "", pending: true },
    ]);
    setInput("");
    setBusy(true);
    setError(null);

    try {
      const result = await sendTurn(session.session_id, text, (token) => {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? { ...message, text: message.text + token }
              : message,
          ),
        );
      });
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, text: result.response, pending: false }
            : message,
        ),
      );
      setSession(result.session);
      setDecision(result.decision);
      if (showResearch) {
        await refreshResearchPanels(result.session.session_id);
      }
    } catch (reason) {
      setMessages((current) =>
        current.filter((message) => message.id !== assistantId),
      );
      setError(reason instanceof Error ? reason.message : "对话请求失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleResearchToggle() {
    const nextValue = !showResearch;
    setShowResearch(nextValue);
    if (nextValue && session) {
      try {
        await refreshResearchPanels(session.session_id);
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : "研究数据读取失败。");
      }
    }
  }

  async function handleDeleteMemory(memoryId: string) {
    if (!session) return;
    await deleteMemory(session.session_id, memoryId);
    setMemories((current) =>
      current.filter((memory) => memory.memory_id !== memoryId),
    );
  }

  async function handleNewSession() {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createSession();
      window.localStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      setSession(created);
      setMessages([INITIAL_MESSAGE]);
      setDecision(null);
      setMemories([]);
      setTraces([]);
      setInput("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法新建研究会话。");
    } finally {
      setBusy(false);
    }
  }

  async function handleProactivityToggle() {
    if (!session || proactivityBusy) return;
    setProactivityBusy(true);
    setError(null);
    try {
      const updated = await setProactivity(
        session.session_id,
        !session.state.proactive_enabled,
      );
      setSession(updated);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "主动关怀设置失败。");
    } finally {
      setProactivityBusy(false);
    }
  }

  async function handleClaimProactiveMessage() {
    if (!session || proactivityBusy || !session.state.proactive_enabled) return;
    setProactivityBusy(true);
    setError(null);
    try {
      const prompt = await claimProactiveMessage(session.session_id);
      setMessages((current) => [
        ...current,
        {
          id: prompt.prompt_id,
          role: "assistant",
          text: prompt.text,
        },
      ]);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "暂时无法领取主动问候。",
      );
    } finally {
      setProactivityBusy(false);
    }
  }

  async function openBenchmarkView() {
    setActiveView("benchmark");
    setBenchmarkBusy(true);
    setError(null);
    try {
      const catalog = await getBenchmarkRuns();
      setBenchmarkRuns(catalog.runs);
      const firstReady = catalog.runs.find((run) => run.report_available);
      if (firstReady) {
        setBenchmarkReport(await getBenchmarkReport(firstReady.run_id));
      } else {
        setBenchmarkReport(null);
      }
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Benchmark 结果读取失败。",
      );
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function handleSelectRun(run: BenchmarkRun) {
    if (benchmarkBusy) return;
    setBenchmarkBusy(true);
    setError(null);
    try {
      const report = run.report_available
        ? await getBenchmarkReport(run.run_id)
        : await generateBenchmarkReport(run.run_id);
      setBenchmarkReport(report);
      setBenchmarkRuns((await getBenchmarkRuns()).runs);
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Benchmark 报告读取失败。",
      );
    } finally {
      setBenchmarkBusy(false);
    }
  }

  async function handleBenchmarkReady(runId: string) {
    const [report, catalog] = await Promise.all([
      getBenchmarkReport(runId), getBenchmarkRuns(),
    ]);
    setBenchmarkReport(report);
    setBenchmarkRuns(catalog.runs);
  }

  const columns = reportColumns(benchmarkReport);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Synthetic safety research console</p>
          <h1>EcoCompanion</h1>
        </div>
        <div className="topbar-actions">
          <nav className="view-switcher" aria-label="研究视图">
            <button
              className={activeView === "companion" ? "active" : ""}
              type="button"
              onClick={() => setActiveView("companion")}
            >
              陪伴演示
            </button>
            <button
              className={activeView === "benchmark" ? "active" : ""}
              type="button"
              onClick={() => void openBenchmarkView()}
            >
              Benchmark
            </button>
          </nav>
          {activeView === "companion" && (
            <>
              <button
                className="new-session-button"
                type="button"
                disabled={!session || busy}
                onClick={() => void handleNewSession()}
              >
                新建会话
              </button>
              <div className={`status-pill ${statusTone}`}>
                <span className="status-dot" />
                {session?.state.phase ?? "CONNECTING"}
                <strong>{session?.state.risk_level ?? "—"}</strong>
              </div>
            </>
          )}
        </div>
      </header>

      <section className="safety-notice">
        <strong>研究原型：</strong>
        不提供诊断、治疗或实时紧急服务。若你或他人正处于即时危险，请立即联系当地紧急服务或现场可信任人员。
      </section>

      {error && activeView === "benchmark" && (
        <div className="error-banner benchmark-error">{error}</div>
      )}

      {activeView === "benchmark" ? (
        <section className="benchmark-workspace">
          <aside className="run-list-card">
            <div className="side-heading">
              <div>
                <p className="eyebrow">Reproducible runs</p>
                <h2>实验运行</h2>
              </div>
              <button
                className="text-button benchmark-icon-button"
                type="button"
                title="刷新运行列表"
                aria-label="刷新运行列表"
                disabled={benchmarkBusy}
                onClick={() => void openBenchmarkView()}
              >
                <RefreshCw size={16} />
              </button>
            </div>
            <BenchmarkLauncher onReportReady={handleBenchmarkReady} />
            {benchmarkBusy && benchmarkRuns.length === 0 && (
              <p className="muted">正在读取本地实验工件…</p>
            )}
            {!benchmarkBusy && benchmarkRuns.length === 0 && (
              <div className="empty-state">
                <strong>尚无可展示的运行</strong>
              </div>
            )}
            <div className="run-list">
              {benchmarkRuns.map((run) => (
                <button
                  className={
                    benchmarkReport?.run.run_id === run.run_id
                      ? "run-row selected"
                      : "run-row"
                  }
                  disabled={benchmarkBusy}
                  key={run.run_id}
                  type="button"
                  onClick={() => void handleSelectRun(run)}
                >
                  <span className={`track-badge track-${run.track}`}>
                    Track {run.track}
                  </span>
                  <strong>{run.run_id}</strong>
                  <small>
                    {run.completed_unit_count}/{run.planned_unit_count} 单元 ·{" "}
                    {run.status === "complete" ? "已完成" : "进行中"}
                  </small>
                  <small>{run.methods.join(" · ")}</small>
                  {!run.report_available && <em>生成报告</em>}
                </button>
              ))}
            </div>
          </aside>

          <article className="report-card">
            {benchmarkReport ? (
              <>
                <div className="report-heading">
                  <div>
                    <p className="eyebrow">Versioned result artifact</p>
                    <h2>{benchmarkReport.run.run_id}</h2>
                  </div>
                  <span className={`track-badge track-${benchmarkReport.run.track}`}>
                    Track {benchmarkReport.run.track}
                  </span>
                </div>
                <div className="report-downloads">
                  {(["md", "json"] as const).map((format) => (
                    <a key={format} href={`/api/v1/research/benchmark/runs/${encodeURIComponent(benchmarkReport.run.run_id)}/download?format=${format}`}
                      download>
                      <Download size={15} />
                      {format === "md" ? "Markdown 报告" : "JSON 数据"}
                    </a>
                  ))}
                </div>
                <dl className="report-summary">
                  <div>
                    <dt>状态</dt>
                    <dd>{benchmarkReport.report.complete ? "完整" : "未完成"}</dd>
                  </div>
                  <div>
                    <dt>已完成单元</dt>
                    <dd>{benchmarkReport.report.completed_unit_count}</dd>
                  </div>
                  <div>
                    <dt>计划单元</dt>
                    <dd>
                      {benchmarkReport.report.planned_unit_count ?? "未记录"}
                    </dd>
                  </div>
                  <div>
                    <dt>Schema</dt>
                    <dd>{benchmarkReport.report.schema_version}</dd>
                  </div>
                </dl>
                <div className="report-table-wrap">
                  <table className="report-table">
                    <thead>
                      <tr>
                        {columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {benchmarkReport.report.rows.map((row, index) => (
                        <tr key={`${benchmarkReport.run.run_id}-${index}`}>
                          {columns.map((column) => (
                            <td key={column}>{displayMetric(row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="scope-note">
                  这些结果只描述合成 Benchmark 中的行为，不构成临床诊断、
                  治疗效果或真实未成年人部署安全性的证据。
                </p>
              </>
            ) : (
              <div className="empty-state report-empty">
                <strong>{benchmarkBusy ? "正在读取或生成报告…" : "暂无选中的报告"}</strong>
              </div>
            )}
          </article>
        </section>
      ) : (
      <div className={`workspace ${showResearch ? "with-research" : ""}`}>
        <section className="chat-card">
          <div className="conversation" aria-live="polite">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <span>{message.role === "user" ? "测试者" : "Eco"}</span>
                <p>
                  {message.text ||
                    (message.pending ? <i className="typing">正在组织回应…</i> : "")}
                </p>
              </article>
            ))}
          </div>

          {error && <div className="error-banner">{error}</div>}

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="输入合成测试对话或成年研究测试内容…"
              rows={3}
              disabled={!session || busy}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <div className="composer-actions">
              <small>
                配置：{session?.model ?? providerLabel}
                {decision && (
                  <>
                    {" · "}本轮{modelCalls.length} 次调用
                    {" · "}
                    <strong className={decision.used_fallback ? "degraded-text" : "model-text"}>
                      {decision.used_fallback ? "模板降级" : "模型生成"}
                    </strong>
                  </>
                )}
              </small>
              <button type="submit" disabled={!session || busy || !input.trim()}>
                {busy ? "处理中" : "发送"}
              </button>
            </div>
          </form>
        </section>

        <aside className="side-card">
          <div className="side-heading">
            <div>
              <p className="eyebrow">Decision snapshot</p>
              <h2>安全决策</h2>
            </div>
            <button className="text-button" onClick={handleResearchToggle}>
              {showResearch ? "隐藏研究面板" : "显示研究面板"}
            </button>
          </div>

          {decision ? (
            <>
              <section className={`runtime-status ${decision.used_fallback ? "degraded" : "generated"}`}>
                <strong>
                  {decision.used_fallback
                    ? "本轮使用安全模板，未直接发布模型候选回复"
                    : `本轮已审核模型回复 · ${modelCalls.length} 次实际调用`}
                </strong>
                <span>
                  {modelCalls.length
                    ? `${modelCalls.map((call) => call.model).join("、")} · 输出 ${modelCalls.reduce((total, call) => total + call.output_tokens, 0)} tokens`
                    : "本轮没有成功完成的模型调用"}
                </span>
                {auditCodes.length > 0 && (
                  <small>状态代码：{auditCodes.join("、")}</small>
                )}
              </section>
              <dl className="decision-grid">
                <div>
                  <dt>主要动作</dt>
                  <dd>{decision.primary_action}</dd>
                </div>
                <div>
                  <dt>目标</dt>
                  <dd>{decision.escalation_target}</dd>
                </div>
                <div>
                  <dt>恶化风险</dt>
                  <dd>{percentage(decision.risk_estimate.depression_worsening)}</dd>
                </div>
                <div>
                  <dt>即时风险</dt>
                  <dd>{percentage(decision.risk_estimate.immediate_safety_risk)}</dd>
                </div>
                <div>
                  <dt>依赖风险</dt>
                  <dd>{percentage(decision.risk_estimate.dependency_risk)}</dd>
                </div>
                <div>
                  <dt>规则不确定性*</dt>
                  <dd>{percentage(decision.uncertainty)}</dd>
                </div>
              </dl>
              <div className="evidence-list">
                <span>证据代码</span>
                {decision.evidence_codes.length ? (
                  decision.evidence_codes.map((code) => (
                    <code key={code}>{code}</code>
                  ))
                ) : (
                  <em>暂无明确风险证据</em>
                )}
              </div>
              <small className="score-note">启发式规则分数，不是诊断概率或统计置信度。</small>
            </>
          ) : (
            <p className="muted">完成一轮对话后显示结构化风险和动作。</p>
          )}

          <section className="proactivity-panel">
            <div>
              <strong>主动关怀</strong>
              <p>
                默认关闭，仅供成年研究测试者显式开启。消息只在本页面领取，
                每个模拟日最多一次。
              </p>
            </div>
            <div className="proactivity-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={!session || proactivityBusy}
                onClick={() => void handleProactivityToggle()}
              >
                {session?.state.proactive_enabled ? "关闭主动关怀" : "开启主动关怀"}
              </button>
              {session?.state.proactive_enabled && (
                <button
                  type="button"
                  disabled={proactivityBusy}
                  onClick={() => void handleClaimProactiveMessage()}
                >
                  领取今日问候
                </button>
              )}
            </div>
          </section>

          {showResearch && (
            <div className="research-panels">
              <section>
                <h3>可删除记忆</h3>
                {memories.length === 0 && <p className="muted">暂无普通记忆。</p>}
                {memories.map((memory) => (
                  <div className="memory-row" key={memory.memory_id}>
                    <div>
                      <small>{memory.kind}</small>
                      <p>{memory.content}</p>
                    </div>
                    <button onClick={() => void handleDeleteMemory(memory.memory_id)}>
                      删除
                    </button>
                  </div>
                ))}
              </section>
              <section>
                <h3>最近Trace</h3>
                {traces.slice(-5).map((trace) => (
                  <div className="trace-row" key={trace.trace_id}>
                    <strong>#{trace.step_id}</strong>
                    <span>{trace.selected_action}</span>
                    <small>
                      {trace.safety_audit.passed ? "AUDIT PASS" : "AUDIT BLOCK"}
                    </small>
                  </div>
                ))}
              </section>
            </div>
          )}
        </aside>
      </div>
      )}
    </main>
  );
}

export default App;
