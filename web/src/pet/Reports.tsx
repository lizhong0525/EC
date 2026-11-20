import { useEffect, useRef, useState } from "react";
import { Download, FileText, LoaderCircle, Plus, Printer, Save, Trash2, X } from "lucide-react";
import { request, type StoredMessage } from "./api";

interface CheckIn { local_date: string; mood: number; energy: number; sleep: number; stress: number; note: string; }
interface ReportDocument {
  report_id: string; created_at: string; summary: string; summary_source: string;
  check_in: CheckIn | null; evidence: { message_id: string; content: string; created_at: string }[];
  notice: string; source_note?: string;
  assessments?: { qwen: { status: string; wellbeing_level?: string; mood?: string; stress?: string; rationale?: string; support_focus?: string; evidence_quotes?: string[] }; decision?: { jev?: { choice?: string }; action?: string; level?: string } }[];
}
interface ReportSummary { report_id: string; created_at: string; check_in: CheckIn | null; has_summary: boolean; }
const emptyCheckIn = (): CheckIn => ({ local_date: new Date().toLocaleDateString("en-CA"), mood: 3, energy: 3, sleep: 3, stress: 3, note: "" });

export default function Reports({ sessionId, messages: initialMessages, onClose }: {
  sessionId: string; messages: StoredMessage[]; onClose: () => void;
}) {
  const [activeSession, setActiveSession] = useState(sessionId);
  const [messages, setMessages] = useState<StoredMessage[]>(initialMessages);
  const [source, setSource] = useState("local");
  const [sources, setSources] = useState<{ session_id: string; source: string; title: string }[]>([]);
  const [checkIn, setCheckIn] = useState(emptyCheckIn);
  const [selected, setSelected] = useState<string[]>([]);
  const [summary, setSummary] = useState("");
  const [preview, setPreview] = useState<(ReportDocument & { preview_id?: string }) | null>(null);
  const [saved, setSaved] = useState<ReportSummary[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const dialog = useRef<HTMLDivElement>(null);
  const users = messages.filter(item => item.role === "USER");
  const changed = () => { setPreview(null); setNotice(""); };
  const editSummary = (value: string) => {
    setSummary(value); setNotice("");
    setPreview(current => current?.preview_id ? { ...current, summary: value } : current);
  };

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLButtonElement>("button")?.focus();
    void request<{ items: ReportSummary[] }>("/reports", undefined, "GET").then(data => setSaved(data.items)).catch(() => {});
    void request<{ items: typeof sources }>("/reports/sources", undefined, "GET").then(data => setSources(data.items)).catch(() => {});
    return () => previous?.focus();
  }, []);
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key !== "Tab") return;
      const nodes = Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not(:disabled),input:not(:disabled),textarea:not(:disabled),select:not(:disabled),a[href]') ?? []);
      if (!nodes.length) return;
      if (event.shiftKey && document.activeElement === nodes[0]) { event.preventDefault(); nodes.at(-1)?.focus(); }
      if (!event.shiftKey && document.activeElement === nodes.at(-1)) { event.preventDefault(); nodes[0]?.focus(); }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [onClose]);

  function payload() {
    return {
      session_id: activeSession, message_ids: selected,
      source, check_in: checkIn, summary, preview_id: null,
    };
  }
  async function createPreview() {
    setBusy("preview"); setError(""); setNotice("");
    try {
      const result = await request<ReportDocument & { preview_id: string }>("/reports/preview", payload());
      setPreview(result);
    } catch (err) { setError(err instanceof Error ? err.message : "预览失败"); }
    finally { setBusy(""); }
  }
  async function save() {
    if (!preview) return;
    setBusy("save"); setError("");
    try {
      await request("/reports", { ...payload(), preview_id: preview.preview_id });
      const data = await request<{ items: ReportSummary[] }>("/reports", undefined, "GET");
      setSaved(data.items); setPreview(null); setSelected([]); setSummary("");
      setNotice("报告已保存到本机");
    } catch (err) { setError(err instanceof Error ? err.message : "保存失败"); }
    finally { setBusy(""); }
  }
  async function open(reportId: string) {
    setBusy("open"); setError("");
    try { setPreview(await request<ReportDocument>(`/reports/${reportId}`, undefined, "GET")); }
    catch (err) { setError(err instanceof Error ? err.message : "打开失败"); }
    finally { setBusy(""); }
  }
  async function remove(reportId: string) {
    if (!window.confirm("确定删除这份报告吗？此操作不可撤销。")) return;
    await request(`/reports/${reportId}`, undefined, "DELETE");
    setSaved(items => items.filter(item => item.report_id !== reportId));
    if (preview?.report_id === reportId) setPreview(null);
  }
  const updateRating = (key: "mood" | "energy" | "sleep" | "stress", value: number) => {
    setCheckIn(previous => ({ ...previous, [key]: value })); changed();
  };
  async function selectSource(value: string) {
    const [nextSource, nextSession] = value.split(":");
    setBusy("source"); setError(""); changed(); setSelected([]); setSummary("");
    try {
      const result = await request<StoredMessage[]>(`/reports/sources/${nextSource}/${nextSession}/messages`, undefined, "GET");
      setSource(nextSource); setActiveSession(nextSession); setMessages(result);
    } catch (err) { setError(err instanceof Error ? err.message : "会话加载失败"); }
    finally { setBusy(""); }
  }

  return <div className="pet-settings-shade pet-report-shade">
    <div ref={dialog} className="pet-reports" role="dialog" aria-modal="true" aria-label="自我记录与报告">
      <header className="pet-settings-header"><div><FileText size={19} /><strong>自我记录</strong><span> / 本地报告</span></div>
        <button className="pet-icon" title="关闭报告" aria-label="关闭报告" onClick={onClose}><X size={19} /></button></header>
      <div className="pet-report-body">
        <aside className="pet-report-list" aria-label="已保存报告"><h2>已保存</h2>
          {saved.length === 0 && <p>还没有报告</p>}
          {saved.map(item => <div key={item.report_id} className="pet-report-item">
            <button onClick={() => void open(item.report_id)}><FileText size={15} /><span>{new Date(item.created_at).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" })}</span></button>
            <button className="pet-icon" aria-label={`删除报告 ${item.report_id}`} title="删除报告" onClick={() => void remove(item.report_id)}><Trash2 size={14} /></button>
          </div>)}
        </aside>
        <section className="pet-report-editor">
          <div className="pet-section-heading"><span className="pet-eyebrow">MY WELLBEING</span><h2>留一份此刻的记录</h2></div>
          <p className="pet-report-notice">自评仅用于个人回顾，不是量表或诊断。报告不会自动发送给任何人。</p>
          <label className="pet-field">会话来源<select aria-label="报告会话来源" value={`${source}:${activeSession}`} disabled={!!busy} onChange={event => void selectSource(event.target.value)}>
            <option value={`local:${sessionId}`}>本地 · 当前对话</option>
            {sources.filter(item => !(item.source === "local" && item.session_id === sessionId)).map(item => <option key={`${item.source}:${item.session_id}`} value={`${item.source}:${item.session_id}`}>
              {item.source === "qq" ? "QQ" : item.source === "personal_wechat" ? "个人微信" : "本地"} · {item.title}
            </option>)}
          </select></label>
          <fieldset className="pet-ratings"><legend>今日自评 · {checkIn.local_date}</legend>
            {([["mood", "心情"], ["energy", "精力"], ["sleep", "睡眠"], ["stress", "压力"]] as const).map(([key, label]) =>
              <label key={key}>{label}<select value={checkIn[key]} onChange={e => updateRating(key, Number(e.target.value))}>
                {[1, 2, 3, 4, 5].map(value => <option key={value} value={value}>{value} / 5</option>)}
              </select></label>)}
          </fieldset>
          <label className="pet-field">今日备注<textarea rows={2} maxLength={2000} value={checkIn.note} onChange={e => { setCheckIn(previous => ({ ...previous, note: e.target.value })); changed(); }} /></label>
          <h3 className="pet-report-subtitle">选择要纳入的本人原话</h3>
          {users.length === 0 && <p className="pet-report-notice">当前会话还没有可选择的本人消息，也可以只记录今日自评。</p>}
          <div className="pet-evidence-list">{users.slice(-40).map(message => <label key={message.message_id}>
            <input type="checkbox" checked={selected.includes(message.message_id)} onChange={event => {
              setSelected(ids => event.target.checked ? [...ids, message.message_id] : ids.filter(id => id !== message.message_id)); changed();
            }} /><span>{message.content}</span></label>)}</div>
          <p className="pet-report-notice">评估来自所选消息当时保存的 Qwen 分层与 Jev 决策，不会在生成报告时重新推断。缺失评估会明确标记。</p>
          <label className="pet-field">个人回顾（可选）<textarea rows={4} maxLength={4000} value={summary}
            placeholder="写下你希望保留的回顾；不填写则引用所选原话" onChange={e => editSummary(e.target.value)} /></label>
          <div className="pet-report-actions"><button className="pet-secondary" disabled={!!busy} onClick={() => void createPreview()}>
            {busy === "preview" ? <LoaderCircle size={16} className="pet-spin" /> : <Plus size={16} />}生成预览</button>
            <button className="pet-primary" disabled={!preview?.preview_id || !!busy} onClick={() => void save()}><Save size={16} />保存快照</button></div>
          {preview && <article className="pet-report-preview"><div><h3>{preview.preview_id ? "报告预览" : "已保存报告"}</h3>
            {preview.report_id && <button className="pet-icon" title="关闭预览" aria-label="关闭预览" onClick={() => setPreview(null)}><X size={16} /></button>}</div>
            {preview.source_note && <p>{preview.source_note}</p>}
            {preview.summary && <p>{preview.summary}</p>}
            {preview.evidence.map(item => <blockquote key={item.message_id}>{item.content}</blockquote>)}
            {preview.assessments?.map((item, index) => <section className="pet-report-assessment" key={index}><h4>评估依据 {index + 1}</h4>
              {item.qwen.status === "ok" ? <><p>支持需要：{{ steady: "日常稳定", strained: "轻度困扰", distressed: "明显困扰", uncertain: "信息不足" }[item.qwen.wellbeing_level || "uncertain"]}</p>
                <p>{item.qwen.rationale || item.qwen.support_focus}</p>
                {item.qwen.evidence_quotes?.map(quote => <blockquote key={quote}>{quote}</blockquote>)}
                <p>Jev：{item.decision?.jev?.choice || "未提供"} · 最终动作：{item.decision?.action || "未记录"} · 安全等级：{item.decision?.level || "未记录"}</p></>
                : <p>本轮 Qwen 评估未完成，不据此推断心理状态。</p>}
            </section>)}
            {preview.report_id && <a href={`/api/pet/reports/${preview.report_id}/print`} target="_blank" rel="noreferrer" onClick={event => {
              if (window.petDesktop) {
                event.preventDefault();
                window.petDesktop.openReport(preview.report_id);
              }
            }}><Printer size={15} />打开打印 / PDF</a>}
          </article>}
          <div className="pet-report-status" role="status">{error || notice}</div>
        </section>
      </div>
      <footer className="pet-settings-footer"><span className="pet-policy-note"><Download size={14} />保存后可在打印窗口导出 PDF，文件仅保存在你选择的位置。</span>
        <button className="pet-secondary" onClick={onClose}>返回对话</button></footer>
    </div>
  </div>;
}
