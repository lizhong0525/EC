import { useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import {
  CheckCheck, ChevronDown, Circle, Coffee, FileText, Heart, LoaderCircle,
  Gamepad2, Maximize2, MessageCircle, Minus, PanelLeft, Plus, Power, Send, Settings2, ShieldCheck, X,
} from "lucide-react";
import Conversations from "./Conversations";
import Settings from "./Settings";
import Reports from "./Reports";
import {
  bootstrap, request, type PetTurn, type SessionActivation, type SettingsView,
  type Stage, type StoredMessage,
} from "./api";
import "./pet.css";

interface Message { id: string; role: "user" | "assistant"; text: string; stages?: Stage[]; mode?: string; }
const SESSION_KEY = "ecocompanion.pet.session";
const NAMES: Record<string, string> = { local: "本地检查", qwen: "Qwen", jev: "Jev", policy: "程序决策", deepseek: "DeepSeek", audit: "输出审查", recovered: "本地恢复" };
const ACTIONS: Record<string, string> = { idle: "在这里", wave: "你好呀", look: "放空片刻", think: "想一想", write: "认真听你说", sleep: "休息中", stretch: "伸个懒腰" };
const mode = new URLSearchParams(location.search).get("surface");
const nativePet = mode === "pet";
const nativePanel = mode === "panel";

export default function PetApp() {
  const [view, setView] = useState<SettingsView | null>(null);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatOpen, setChatOpen] = useState(!nativePet);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [reportsOpen, setReportsOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(() => window.innerWidth >= 700);
  const [historyRevision, setHistoryRevision] = useState(0);
  const [switching, setSwitching] = useState(false);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [action, setAction] = useState("idle");
  const [attention, setAttention] = useState(false);
  const [mediaError, setMediaError] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const scroll = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const panel = useRef<HTMLElement>(null);
  const busyRef = useRef(false);
  const switchingRef = useRef(false);
  const sessionRef = useRef("");
  const draftRef = useRef("");
  const drag = useRef<{ x: number; y: number; sx: number; sy: number; moved: boolean; active: boolean } | null>(null);
  const cfg = view?.settings;
  const animate = useCallback((next: string) => {
    setAction(next);
    window.petDesktop?.activity(next);
  }, []);

  function restoreSession(data: SessionActivation) {
    localStorage.setItem(SESSION_KEY, data.session_id);
    sessionRef.current = data.session_id;
    setSessionId(data.session_id);
    setMessages(data.messages.map(message => ({
      id: message.message_id, role: message.role === "USER" ? "user" : "assistant", text: message.content,
    })));
    draftRef.current = data.draft ?? "";
    setDraft(data.draft ?? "");
    setAttention(data.attention);
    animate(data.attention ? "attentive" : "idle");
    setError("");
  }
  async function newSession() {
    if (busyRef.current || switchingRef.current) return;
    switchingRef.current = true; setSwitching(true);
    try {
      if (sessionRef.current) await request(`/sessions/${sessionRef.current}/draft`, { text: draftRef.current }, "PUT");
      const session = await request<{ session_id: string }>("/sessions", {});
      restoreSession({ session_id: session.session_id, messages: [], attention: false });
      setHistoryRevision(value => value + 1);
      if (innerWidth < 700) setHistoryOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "新建对话失败");
    } finally { switchingRef.current = false; setSwitching(false); }
  }
  async function selectSession(id: string) {
    if (busyRef.current || switchingRef.current) return;
    if (id === sessionId) { if (innerWidth < 700) setHistoryOpen(false); return; }
    switchingRef.current = true; setSwitching(true);
    try {
      if (sessionRef.current) await request(`/sessions/${sessionRef.current}/draft`, { text: draftRef.current }, "PUT");
      const data = await request<SessionActivation>(`/sessions/${id}/activate`, {});
      restoreSession(data);
      if (innerWidth < 700) setHistoryOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "切换对话失败，当前对话已保留");
    } finally { switchingRef.current = false; setSwitching(false); }
  }
  useEffect(() => {
    document.title = "心伴 · 桌面心理支持伙伴";
    let cancelled = false;
    const init = async () => {
      try {
        const data = await bootstrap();
        if (cancelled) return;
        setView(data);
        window.petDesktop?.configure(data.settings);
        if (nativePet) return;
        const previous = data.current_session_id || localStorage.getItem(SESSION_KEY);
        if (previous) {
          try {
            const session = await request<SessionActivation>(`/sessions/${previous}/activate`, {});
            if (cancelled) return;
            restoreSession(session);
            return;
          } catch { /* A deleted or unavailable session starts fresh. */ }
        }
        if (!cancelled) await newSession();
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "无法连接本地服务");
      }
    };
    void init();
    return () => { cancelled = true; };
  }, []);
  useEffect(() => { sessionRef.current = sessionId; }, [sessionId]);
  useEffect(() => { draftRef.current = draft; }, [draft]);
  useEffect(() => {
    if (!sessionId) return;
    const id = sessionId, text = draft;
    const timer = window.setTimeout(() => {
      void request(`/sessions/${id}/draft`, { text }, "PUT").catch(() => {
        // Keep the text in the composer; retry on the next edit or navigation.
      });
    }, 400);
    return () => clearTimeout(timer);
  }, [sessionId, draft]);
  useEffect(() => () => {
    if (sessionRef.current) {
      void request(`/sessions/${sessionRef.current}/draft`, { text: draftRef.current }, "PUT");
    }
  }, []);
  useEffect(() => {
    const handle = () => { if (innerWidth < 700) setHistoryOpen(false); };
    window.addEventListener("resize", handle);
    return () => window.removeEventListener("resize", handle);
  }, []);
  useEffect(() => {
    const handle = (event: StorageEvent) => {
      if (event.key === "ecocompanion.pet.settings-revision") {
        void bootstrap().then(data => { setView(data); window.petDesktop?.configure(data.settings); });
      }
    };
    window.addEventListener("storage", handle);
    const offActivity = window.petDesktop?.onActivity(next => {
      setAction(next);
      setAttention(next === "attentive");
    });
    const offSettings = window.petDesktop?.onSettings(() => setSettingsOpen(true));
    return () => { window.removeEventListener("storage", handle); offActivity?.(); offSettings?.(); };
  }, []);
  useEffect(() => {
    if (chatOpen) { scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" }); }
  }, [messages, busy, chatOpen]);
  useEffect(() => {
    if (!cfg?.animation_enabled) { video.current?.pause(); return; }
    void video.current?.play().catch(() => { /* Browser may defer playback until visible. */ });
  }, [action, cfg?.animation_enabled]);
  useEffect(() => {
    if (action !== "idle" || busy || attention || !cfg?.animation_enabled) return;
    const timer = window.setTimeout(() => animate("look"), 32000);
    return () => window.clearTimeout(timer);
  }, [action, busy, attention, cfg?.animation_enabled, animate]);
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !settingsOpen) {
        if (nativePanel) window.petDesktop?.hideChat();
        else setChatOpen(false);
      }
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [settingsOpen]);

  async function send() {
    const text = draft.trim();
    if (!text || busyRef.current || switchingRef.current || !sessionId) return;
    busyRef.current = true; setBusy(true); setError("");
    const id = sessionId;
    setDraft("");
    const requestId = crypto.randomUUID();
    setMessages(previous => [...previous, { id: `${id}:user:${requestId}`, role: "user", text }]);
    animate(attention ? "attentive" : "think");
    try {
      const result = await request<PetTurn>(`/sessions/${id}/turn`, { text, request_id: requestId });
      if (sessionRef.current !== id) return;
      setMessages(previous => [...previous, {
        id: `${id}:assistant:${requestId}`, role: "assistant", text: result.result.response,
        stages: result.stages, mode: result.mode,
      }]);
      setAttention(result.attention);
      animate(result.attention ? "attentive" : "idle");
    } catch (err) {
      setError(err instanceof Error ? err.message : "发送失败");
      setDraft(text);
      // Read persisted messages before allowing a retry after a potentially committed turn.
      try {
        const data = await request<SessionActivation>(`/sessions/${id}/activate`, {});
        if (sessionRef.current === id) {
          setMessages(data.messages.map(item => ({ id: item.message_id, role: item.role === "USER" ? "user" : "assistant", text: item.content })));
          const committed = data.messages.some(item => item.message_id === `${id}:assistant:${requestId}`);
          setDraft(committed ? "" : text);
          if (committed) setError("上一条消息已完成，已恢复本地记录。");
        }
      } catch { /* The unsent draft remains editable and will be saved on the next change. */ }
      animate(attention ? "attentive" : "idle");
    } finally {
      setHistoryRevision(value => value + 1);
      busyRef.current = false; setBusy(false); input.current?.focus();
    }
  }
  function openChat() {
    if (nativePet) window.petDesktop?.toggleChat();
    else setChatOpen(previous => !previous);
    if (!attention && !busyRef.current && action !== "think") animate("wave");
  }
  function startDrag(event: PointerEvent<HTMLButtonElement>) {
    if (event.button !== 0) return;
    drag.current = { x: event.screenX, y: event.screenY, sx: position.x, sy: position.y, moved: false, active: false };
    if (nativePet) window.petDesktop?.drag(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function moveDrag(event: PointerEvent<HTMLButtonElement>) {
    if (!drag.current) return;
    const dx = event.screenX - drag.current.x;
    const dy = event.screenY - drag.current.y;
    if (Math.abs(dx) + Math.abs(dy) > 5) {
      drag.current.moved = true;
      if (nativePet && !drag.current.active) {
        drag.current.active = true;
        // Keep pointer hit-testing alive after the cursor leaves the sprite.
        window.petDesktop?.drag(true, drag.current.x, drag.current.y);
        // Apply the threshold-crossing frame as well as subsequent frames.
        window.petDesktop?.movePet(dx, dy, event.screenX, event.screenY);
        setIsDragging(true);
        return;
      }
    }
    if (!drag.current.moved) return;
    setIsDragging(true);
    if (nativePet) {
      window.petDesktop?.movePet(dx, dy, event.screenX, event.screenY);
      drag.current.x = event.screenX; drag.current.y = event.screenY;
    } else {
      const maxX = Math.min(150, Math.max(0, innerWidth / 2 - 120));
      const maxY = Math.max(0, innerHeight / 2 - 180);
      setPosition({
        x: Math.min(maxX, Math.max(-maxX, drag.current.sx + dx)),
        y: Math.min(maxY, Math.max(-maxY, drag.current.sy + dy)),
      });
    }
  }
  function endDrag() {
    const moved = drag.current?.moved ?? false;
    if (nativePet) window.petDesktop?.drag(false);
    drag.current = null; setIsDragging(false);
    if (!moved) openChat();
  }
  function cancelDrag() {
    if (nativePet) window.petDesktop?.drag(false);
    drag.current = null; setIsDragging(false);
  }
  function saveSettings(data: SettingsView) {
    setView(data);
    localStorage.setItem("ecocompanion.pet.settings-revision", String(Date.now()));
    window.petDesktop?.configure(data.settings);
  }
  function settings() {
    if (nativePet) window.petDesktop?.openSettings();
    else setSettingsOpen(true);
  }
  const active = cfg?.cloud_consent && view?.deepseek_key_configured;
  const name = cfg?.pet_name || "小溟";

  return <main className={`pet-app ${nativePet ? "pet-native-figure" : ""} ${nativePanel ? "pet-native-panel" : ""}`}>
    {!nativePet && !nativePanel && <>
      <header className="pet-app-bar"><div className="pet-brand"><Heart size={21} /><strong>心伴</strong><span>DESKTOP COMPANION</span></div>
        <div className="pet-app-actions"><span className="pet-version">PREVIEW 0.1</span><button className="pet-icon" title="进入小游戏" aria-label="进入小游戏" onClick={() => { location.href = "/games"; }}><Gamepad2 size={19} /></button><button className="pet-icon" title="应用设置" aria-label="应用设置" onClick={settings}><Settings2 size={19} /></button></div></header>
      <div className="pet-desk-date"><span>{new Date().toLocaleDateString("zh-CN", { month: "long", day: "numeric", weekday: "long" })}</span><span className="pet-desk-label"><Circle size={7} fill="currentColor" />留一点时间给自己</span></div>
    </>}
    <div className="pet-workspace">
      {(chatOpen || nativePanel) && !nativePet &&
        <section className={`pet-chat-shell ${historyOpen ? "pet-history-open" : ""}`} ref={panel} aria-label="与桌宠对话">
          {historyOpen && sessionId && <Conversations selected={sessionId} revision={historyRevision}
            disabled={busy || switching || !sessionId} onSelect={id => void selectSession(id)}
            onNew={() => void newSession()} onClose={() => setHistoryOpen(false)} />}
          {historyOpen && <button className="pet-history-scrim" aria-label="关闭对话列表" onClick={() => setHistoryOpen(false)} />}
          <div className="pet-chat">
          <header className="pet-chat-header">
            <button className="pet-icon" title="对话列表" aria-label="对话列表" aria-expanded={historyOpen}
              aria-controls="pet-conversations" onClick={() => setHistoryOpen(value => !value)}><PanelLeft size={18} /></button>
            <div className="pet-chat-avatar"><img src="/pet/portrait.png" alt="" /></div>
            <div className="pet-chat-title"><h1 title={name}>{name}</h1><span><i className={active ? "connected" : ""} />{busy ? "DeepSeek 正在回复" : active ? "DeepSeek 对话" : "未配置 DeepSeek"}</span></div>
            <button className="pet-icon" title="新建对话" aria-label="新建对话"
              disabled={busy || switching || !sessionId} onClick={() => void newSession()}><Plus size={19} /></button>
            <button className="pet-icon" title="自我记录与报告" aria-label="自我记录与报告"
              disabled={!sessionId} onClick={() => setReportsOpen(true)}><FileText size={18} /></button>
            <button className="pet-icon" title="应用设置" aria-label="应用设置" onClick={settings}><Settings2 size={18} /></button>
            <button className="pet-icon" title="收起对话" aria-label="收起对话" onClick={() => nativePanel ? window.petDesktop?.hideChat() : setChatOpen(false)}><Minus size={19} /></button>
          </header>
          <div className="pet-chat-policy"><ShieldCheck size={13} />成年测试预览 · 非诊断与紧急救助服务</div>
          <div className="pet-messages" ref={scroll} aria-live="polite" aria-busy={busy || switching}>
            <div className="pet-conversation-date">今天</div>
            <article className="pet-message pet-assistant"><span className="pet-message-author">{name}</span>
              <div className="pet-bubble">我在。今天有什么想聊的吗？<br />一点小事，也可以慢慢说。</div>
              <span className="pet-message-time">此刻</span>
            </article>
            {messages.map(message => <article key={message.id} className={`pet-message pet-${message.role}`}>
              <span className="pet-message-author">{message.role === "user" ? "我" : name}</span>
              <div className="pet-bubble">{message.text}</div>
              {message.role === "assistant" && message.stages &&
            <details className="pet-turn-details"><summary><CheckCheck size={12} />{message.mode === "deepseek" ? "DeepSeek · 已审查" : message.mode === "recovered" ? "本地记录 · 已恢复" : "DeepSeek 不可用 · 安全回退"}<ChevronDown size={12} /></summary>
                  <ol>{message.stages.map((stage, index) => <li key={index}><span className={`pet-stage-dot pet-stage-${stage.state}`} /><strong>{NAMES[stage.name] || stage.name}</strong><span>{stage.detail}</span></li>)}</ol>
                </details>}
            </article>)}
            {busy && <article className="pet-message pet-assistant"><span className="pet-message-author">{name}</span><div className="pet-bubble pet-typing"><i /><i /><i /><span>正在整理回复</span></div></article>}
            {switching && <div className="pet-history-loading" role="status"><LoaderCircle size={16} className="pet-spin" />正在打开对话</div>}
          </div>
          {error && <div className="pet-chat-error" role="alert">{error}<button className="pet-icon" title="关闭提示" aria-label="关闭提示" onClick={() => setError("")}><X size={14} /></button></div>}
          <form className="pet-composer" onSubmit={event => { event.preventDefault(); void send(); }}>
            <textarea ref={input} aria-label="消息" placeholder="想说的话，慢慢来…" rows={2}
              maxLength={8000} value={draft} disabled={switching} onChange={event => setDraft(event.target.value)}
              onKeyDown={event => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault(); void send();
                }
              }} />
            <div className="pet-composer-bottom"><span><ShieldCheck size={12} />{active ? "回复由 DeepSeek 生成" : "请先在设置中配置 DeepSeek"}</span>
              <button type="submit" className="pet-send" title="发送消息" aria-label="发送消息" disabled={!draft.trim() || busy || switching || !sessionId}>
                {busy ? <LoaderCircle size={18} className="pet-spin" /> : <Send size={18} />}
              </button></div>
          </form>
          </div>
        </section>}
      {!nativePanel && <div className={`pet-companion ${isDragging ? "pet-dragging" : ""}`}
        style={{ "--pet-scale": (cfg?.pet_scale ?? 100) / 100, transform: `translate(${position.x}px, ${position.y}px)` } as CSSProperties}>
        {!chatOpen && !nativePet && <button className="pet-invitation" onClick={openChat}>今天过得怎么样？<MessageCircle size={14} /></button>}
        <div className="pet-figure-wrap">
          <button className="pet-figure" aria-label="点击桌宠展开对话" title="与桌宠对话"
            onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={endDrag}
            onPointerCancel={cancelDrag} onLostPointerCapture={cancelDrag}
            onClick={event => { if (event.detail === 0) openChat(); }}>
            <video ref={video} key={action} src={`/pet/${action === "attentive" ? "idle" : action}.webm`} muted playsInline autoPlay={cfg?.animation_enabled ?? true}
              loop={action === "idle" || action === "attentive" || action === "think" || action === "sleep"}
              onLoadedData={() => { setMediaError(false); if (cfg && !cfg.animation_enabled) video.current?.pause(); }}
              onError={() => setMediaError(true)} onEnded={() => animate("idle")} />
            {mediaError && <span className="pet-media-error">动画暂不可用</span>}
          </button>
        </div>
        <div className="pet-caption"><span className="pet-presence" />{name}<span className="pet-caption-divider">/</span><span>{attention ? "认真听你说" : ACTIONS[action] || "在这里"}</span></div>
        <div className="pet-tools">
          <button className="pet-icon" title="与桌宠对话" aria-label="与桌宠对话" onClick={openChat}><MessageCircle size={18} /></button>
          <button className={`pet-icon ${action === "sleep" ? "pet-selected" : ""}`} title={action === "sleep" ? "唤醒桌宠" : "休息一下"} aria-label={action === "sleep" ? "唤醒桌宠" : "休息一下"}
            disabled={busy || attention || action === "think"} onClick={() => animate(action === "sleep" ? "stretch" : "sleep")}><Coffee size={18} /></button>
          <button className="pet-icon" title="应用设置" aria-label="桌宠设置" onClick={settings}><Settings2 size={18} /></button>
          <button className="pet-icon" title="小游戏" aria-label="小游戏" onClick={() => window.petDesktop ? window.petDesktop.openGames() : window.open("/games", "_blank")}><Gamepad2 size={18} /></button>
          {nativePet && <button className="pet-icon" title="退出桌宠" aria-label="退出桌宠" onClick={() => window.petDesktop?.quit()}><Power size={17} /></button>}
        </div>
      </div>}
    </div>
    {!nativePet && !nativePanel && <footer className="pet-app-footer">
      <span><span className="pet-presence" />{active ? cfg?.deepseek_model : "DeepSeek 未配置"}</span>
      <span>Qwen <b>{cfg?.qwen_enabled ? "已启用 · 待调用" : "已关闭"}</b></span>
      <span>Jev <b>{cfg?.jev_enabled && view?.jev_key_configured && cfg.cloud_consent ? "已配置 · 待调用" : "未启用"}</b></span>
      <button onClick={settings}>隐私与设置<Maximize2 size={12} /></button>
    </footer>}
    {!view && error && <div className="pet-startup-error" role="alert">{error}<button className="pet-secondary" onClick={() => location.reload()}>重新连接</button></div>}
    {settingsOpen && view && <Settings initial={view} onSaved={saveSettings} onClose={() => setSettingsOpen(false)}
      onPreview={next => { if (!busy && !attention && action !== "think") animate(next); }} />}
    {reportsOpen && <Reports sessionId={sessionId} messages={messages.map(message => ({ message_id: message.id, role: message.role === "user" ? "USER" : "COMPANION", content: message.text }))} onClose={() => setReportsOpen(false)} />}
  </main>;
}
