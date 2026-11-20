import { useEffect, useRef, useState } from "react";
import {
  Check, LoaderCircle, MessageSquareText, Play, Plug, Save, ShieldCheck, Square,
} from "lucide-react";
import { request, type MessagingSettings, type MessagingSettingsView } from "./api";
import { SecretField, Toggle } from "./Settings";
import WeChatIdentityPicker from "./WeChatIdentityPicker";

const empty: MessagingSettings = {
  qq: { enabled: false, app_id: "", client_secret: "", intents: 1 << 25 },
  wechat: { enabled: false, token: "", app_id: "", app_secret: "", allow_ip_cidr: "" },
  personal_wechat: { enabled: false, account_wxid: "", allowed_chats: [], allow_groups: false, test_account_confirmed: false },
  callback_host: "127.0.0.1", callback_port: 8765, per_user_per_minute: 6, max_inflight: 8,
};
type Runtime = { running: boolean; ready?: boolean; channels: string[]; detail: string };
type Check = { channel: string; ok: boolean; detail: string };
const channelNames: Record<string, string> = {
  qq: "QQ", personal_wechat: "个人微信桥接", deepseek: "DeepSeek",
};

export default function MessagingSettingsPanel() {
  const [initial, setInitial] = useState<MessagingSettingsView | null>(null);
  const [draft, setDraft] = useState<MessagingSettings>(empty);
  const [secrets, setSecrets] = useState({ qq_client_secret: "", personal_wechat_bridge_token: "" });
  const [clear, setClear] = useState({ qq: false, personal: false });
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(false);
  const [checks, setChecks] = useState<Check[]>([]);
  const [runtime, setRuntime] = useState<Runtime>({ running: false, channels: [], detail: "消息网关未启动" });
  const [bridge, setBridge] = useState<Runtime>({ running: false, channels: [], detail: "独立微信测试桥接未启动" });
  const feedback = useRef<HTMLDivElement>(null);

  function showFeedback() {
    window.setTimeout(() => feedback.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" }), 0);
  }

  useEffect(() => {
    let cancelled = false;
    void Promise.all([
      request<MessagingSettingsView>("/messaging/settings", undefined, "GET"),
      request<Runtime>("/messaging/runtime", undefined, "GET"),
      request<Runtime>("/messaging/wechat-bridge", undefined, "GET"),
    ]).then(([view, status, bridgeStatus]) => {
      if (!cancelled) { setInitial(view); setDraft(view.settings); setRuntime(status); setBridge(bridgeStatus); }
    }).catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : "渠道配置读取失败"); });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    if (!bridge.running) return;
    const timer = window.setInterval(() => {
      void request<Runtime>("/messaging/wechat-bridge", undefined, "GET")
        .then(setBridge).catch(() => {});
    }, 1500);
    return () => window.clearInterval(timer);
  }, [bridge.running]);
  function setQQ(value: Partial<MessagingSettings["qq"]>) {
    setDraft(previous => ({ ...previous, qq: { ...previous.qq, ...value } })); setNotice("");
  }
  function setPersonal(value: Partial<MessagingSettings["personal_wechat"]>) {
    setDraft(previous => ({ ...previous, personal_wechat: { ...previous.personal_wechat, ...value } })); setNotice("");
  }
  async function save() {
    setBusy(true); setError(""); setNotice(""); setChecks([]);
    try {
      const view = await request<MessagingSettingsView>("/messaging/settings", {
        settings: { ...draft, wechat: { ...draft.wechat, enabled: false } }, ...secrets,
        clear_qq_client_secret: clear.qq, clear_personal_wechat_bridge_token: clear.personal,
      }, "PUT");
      if (runtime.running) {
        const [status, bridgeStatus] = await Promise.all([
          request<Runtime>("/messaging/runtime", undefined, "GET"),
          request<Runtime>("/messaging/wechat-bridge", undefined, "GET"),
        ]);
        setRuntime(status); setBridge(bridgeStatus);
      }
      setInitial(view); setDraft(view.settings);
      setSecrets({ qq_client_secret: "", personal_wechat_bridge_token: "" });
      setClear({ qq: false, personal: false });
      setNotice(runtime.running ? "渠道配置已保存，网关与监听已按新配置重载" : "渠道配置已保存；启动网关后生效");
    } catch (err) { setError(err instanceof Error ? err.message : "保存失败"); }
    finally { setBusy(false); showFeedback(); }
  }
  async function preflight() {
    setChecking(true); setError(""); setChecks([]);
    try {
      const result = await request<{ checks: Check[]; detail: string }>("/messaging/preflight", {});
      setChecks(result.checks);
      if (!result.checks.length) setError(result.detail);
    } catch (err) { setError(err instanceof Error ? err.message : "连接预检失败"); }
    finally { setChecking(false); showFeedback(); }
  }
  async function setRunning(running: boolean) {
    setBusy(true); setError(""); setNotice("");
    try {
      const status = await request<Runtime>("/messaging/runtime", { running });
      setRuntime(status); setNotice(status.detail);
    } catch (err) { setError(err instanceof Error ? err.message : "网关操作失败"); }
    finally { setBusy(false); showFeedback(); }
  }
  async function setBridgeRunning(running: boolean) {
    setBusy(true); setError(""); setNotice("");
    try {
      const status = await request<Runtime>("/messaging/wechat-bridge", { running });
      setBridge(status); setNotice(status.detail);
    } catch (err) { setError(err instanceof Error ? err.message : "桥接操作失败"); }
    finally { setBusy(false); showFeedback(); }
  }
  async function checkBridge() {
    setChecking(true); setError(""); setNotice("");
    try {
      const result = await request<{ ok: boolean; detail: string }>("/messaging/wechat-bridge/check", {});
      if (result.ok) setNotice(result.detail); else setError(result.detail);
    } catch (err) { setError(err instanceof Error ? err.message : "桥接预检失败"); }
    finally { setChecking(false); showFeedback(); }
  }
  if (!initial && !error) return <div className="pet-history-loading"><LoaderCircle className="pet-spin" size={17} />读取渠道配置</div>;
  return <div className="pet-channel-settings">
    {initial && <>
      <div className="pet-runtime">
        <div><span className={runtime.running ? "pet-runtime-dot is-running" : "pet-runtime-dot"} />
          <strong>{runtime.running ? "消息网关运行中" : "消息网关未启动"}</strong>
          <span>{runtime.channels.map(item => channelNames[item]).join(" · ")}</span></div>
        <button type="button" className="pet-secondary" disabled={busy} onClick={() => void setRunning(!runtime.running)}>
          {runtime.running ? <Square size={14} /> : <Play size={14} />}{runtime.running ? "停止" : "启动已保存配置"}</button>
      </div>
      <p className="pet-policy-note"><ShieldCheck size={15} />渠道自动复用桌宠的 DeepSeek、人格与评估设置：{initial.deepseek_key_configured && initial.cloud_consent ? "DeepSeek 回复已启用，Qwen 仅做评估" : "DeepSeek 未配置，普通聊天不会生成回复"}。渠道会话独立保存。</p>
      <div ref={feedback} className="pet-channel-feedback" aria-live="polite">
        {(error || notice) && <span className={error ? "pet-error" : "pet-positive"} role="status">
          {error || <><Check size={14} />{notice}</>}</span>}
        {checks.map(check => <p key={check.channel} className={check.ok ? "pet-positive" : "pet-error"} role="status">
          {channelNames[check.channel] || check.channel}：{check.detail}</p>)}
      </div>
      <div className="pet-provider-title"><MessageSquareText size={27} /><div><h3>QQ 官方机器人 · 私聊</h3><span>AppID + AppSecret，无需 NapCat</span></div></div>
      <Toggle label="启用 QQ 渠道" checked={draft.qq.enabled} onChange={enabled => setQQ({ enabled })} />
      <label className="pet-field">QQ AppID<input required={draft.qq.enabled} value={draft.qq.app_id} onChange={event => setQQ({ app_id: event.target.value })} /></label>
      <SecretField label="QQ AppSecret" configured={initial.qq_secret_configured} value={secrets.qq_client_secret}
        onChange={value => setSecrets(previous => ({ ...previous, qq_client_secret: value }))}
        clear={clear.qq} onClear={() => setClear(previous => ({ ...previous, qq: !previous.qq }))} />
      <div className="pet-setting-divider" />
      <div className="pet-provider-title"><MessageSquareText size={27} /><div><h3>个人微信 · 隔离测试桥接</h3><span>非官方 API；本地数据库读取 + 窗口自动化</span></div><span className="pet-small-tag">默认关闭</span></div>
      <p className="pet-policy-note"><ShieldCheck size={15} />仅限专用测试账号。存在客户端版本、账号条款、窗口焦点与误发风险；本版仅私聊白名单，不接管当前微信。</p>
      <p className="pet-policy-note">微信发送回复的几秒内，请停止移动鼠标和键盘操作。需要手动使用微信时，先停止微信桥接。</p>
      <WeChatIdentityPicker value={draft.personal_wechat} disabled={busy} onChange={setPersonal} />
      <SecretField label="独立桥接 Token" configured={initial.personal_wechat_token_configured} value={secrets.personal_wechat_bridge_token}
        onChange={value => setSecrets(previous => ({ ...previous, personal_wechat_bridge_token: value }))}
        clear={clear.personal} onClear={() => setClear(previous => ({ ...previous, personal: !previous.personal }))} />
      <p className="pet-policy-note">首次启用且留空时会自动生成独立 Token，无需复制。首次执行 <code>.\Install-WeChatTestBridge.ps1</code> 安装可选依赖；保存配置并启动网关后，再单独启动此桥接。不会随应用启动。</p>
      <div className="pet-channel-actions">
        <button type="button" className="pet-secondary" disabled={busy || checking} onClick={() => void checkBridge()}><Plug size={14} />桥接预检</button>
        <button type="button" className="pet-secondary" disabled={busy || !draft.personal_wechat.enabled} onClick={() => void setBridgeRunning(!bridge.running)}>
          {bridge.running ? <Square size={14} /> : <Play size={14} />}{bridge.running ? "停止微信桥接" : "启动微信桥接"}</button>
        <span className={bridge.ready ? "pet-positive" : bridge.running ? "pet-muted" : "pet-error"} role="status">{bridge.detail}</span>
      </div>
      <div className="pet-setting-divider" />
      <div className="pet-two-columns">
        <label className="pet-field">本机回调端口<input type="number" min="1024" max="65535" value={draft.callback_port} onChange={event => setDraft(previous => ({ ...previous, callback_port: Number(event.target.value) }))} /></label>
        <label className="pet-field">每用户每分钟消息数<input type="number" min="1" max="30" value={draft.per_user_per_minute} onChange={event => setDraft(previous => ({ ...previous, per_user_per_minute: Number(event.target.value) }))} /></label>
      </div>
      <button type="button" className="pet-secondary" disabled={checking || busy} onClick={() => void preflight()}>
        {checking ? <LoaderCircle size={15} className="pet-spin" /> : <Plug size={15} />}检查已保存的连接</button>
      <div className="pet-channel-footer"><span className="pet-muted">启动和预检只使用已保存的配置</span>
        <button type="button" className="pet-primary" disabled={busy} onClick={() => void save()}><Save size={14} />{busy ? "处理中" : "保存渠道配置"}</button></div>
    </>}
    {!initial && error && <p className="pet-error" role="alert">{error}</p>}
  </div>;
}
