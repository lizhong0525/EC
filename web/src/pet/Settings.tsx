import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ArrowLeft, Check, ChevronRight, Cpu, Eye, EyeOff, Info, KeyRound, LoaderCircle,
  MessageSquareText, Palette, Plug, RotateCcw, Save, Settings2, ShieldCheck, Sparkles, Trash2, X,
} from "lucide-react";
import { request, type PetSettings, type SettingsView } from "./api";
import MessagingSettingsPanel from "./MessagingSettings";

type Tab = "connection" | "personality" | "assessment" | "channels" | "appearance" | "about";
const TABS = [
  { id: "connection", label: "对话服务", icon: Plug },
  { id: "personality", label: "人格与表达", icon: Sparkles },
  { id: "assessment", label: "评估与决策", icon: Cpu },
  { id: "channels", label: "消息渠道", icon: MessageSquareText },
  { id: "appearance", label: "桌宠与偏好", icon: Palette },
  { id: "about", label: "隐私与来源", icon: ShieldCheck },
] as const;

export function Toggle({ label, checked, onChange }: {
  label: string; checked: boolean; onChange: (value: boolean) => void;
}) {
  return <label className="pet-toggle-row"><span>{label}</span>
    <input type="checkbox" role="switch" checked={checked}
      onChange={event => onChange(event.target.checked)} />
  </label>;
}

export function SecretField({ label, configured, value, onChange, clear, onClear }: {
  label: string; configured: boolean; value: string; onChange: (value: string) => void;
  clear: boolean; onClear: () => void;
}) {
  const [visible, setVisible] = useState(false);
  return <div className="pet-field">
    <label>{label}<span className={configured && !clear ? "pet-positive" : "pet-muted"}>
      {clear ? "保存后移除" : configured ? "已保存" : "未配置"}</span></label>
    <div className="pet-secret">
      <KeyRound size={16} />
      <input aria-label={label} type={visible ? "text" : "password"} autoComplete="off"
        value={value} onChange={event => onChange(event.target.value)}
        placeholder={configured && !clear ? "已保存，留空不更改" : "输入 API Key"} />
      <button type="button" title={visible ? "隐藏密钥" : "显示密钥"}
        aria-label={visible ? "隐藏密钥" : "显示密钥"} onClick={() => setVisible(!visible)}>
        {visible ? <EyeOff size={16} /> : <Eye size={16} />}
      </button>
      {configured && <button type="button" title={clear ? "撤销移除" : "移除密钥"}
        aria-label={clear ? "撤销移除" : "移除密钥"} onClick={onClear}>
        {clear ? <RotateCcw size={16} /> : <Trash2 size={16} />}
      </button>}
    </div>
  </div>;
}

export default function Settings({ initial, onSaved, onClose, onPreview }: {
  initial: SettingsView; onSaved: (data: SettingsView) => void; onClose: () => void;
  onPreview: (action: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("connection");
  const [draft, setDraft] = useState<PetSettings>({ ...initial.settings });
  const [deepseekKey, setDeepseekKey] = useState("");
  const [jevKey, setJevKey] = useState("");
  const [clearDeepseek, setClearDeepseek] = useState(false);
  const [clearJev, setClearJev] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const dialog = useRef<HTMLFormElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialog.current?.querySelector<HTMLElement>("button")?.focus();
    return () => previous?.focus();
  }, []);
  const dirty = JSON.stringify(draft) !== JSON.stringify(initial.settings)
    || !!deepseekKey || !!jevKey || clearDeepseek || clearJev;
  const set = <K extends keyof PetSettings>(key: K, value: PetSettings[K]) => {
    setDraft(previous => ({ ...previous, [key]: value }));
    setNotice("");
  };
  const close = () => {
    if (!dirty || window.confirm("还有未保存的设置，确定放弃更改吗？")) onClose();
  };
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });
  async function save(event: FormEvent) {
    event.preventDefault();
    setSaving(true); setError(""); setNotice("");
    try {
      const view = await request<SettingsView>("/settings", {
        settings: draft,
        deepseek_api_key: deepseekKey || null,
        jev_api_key: jevKey || null,
        clear_deepseek_key: clearDeepseek,
        clear_jev_key: clearJev,
      }, "PUT");
      setDeepseekKey(""); setJevKey(""); setClearDeepseek(false); setClearJev(false);
      onSaved(view);
      setNotice("设置已保存");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    } finally { setSaving(false); }
  }
  async function test(provider: "deepseek" | "qwen" | "jev") {
    if (dirty) { setError("请先保存更改，再检查连接。"); return; }
    setTesting(provider); setError(""); setNotice("");
    try {
      const result = await request<{ ok: boolean; detail: string }>(`/test/${provider}`, {});
      if (result.ok) setNotice(result.detail); else setError(result.detail);
    } catch (err) { setError(err instanceof Error ? err.message : "连接失败"); }
    finally { setTesting(""); }
  }
  const heading = TABS.find(item => item.id === tab)!.label;
  return <div className="pet-settings-shade">
    <form ref={dialog} className="pet-settings" role="dialog" aria-modal="true" aria-label="应用设置" onSubmit={save}
      onKeyDown={event => {
        if (event.key !== "Tab") return;
        const items = Array.from(dialog.current!.querySelectorAll<HTMLElement>(
          'button:not(:disabled), input:not(:disabled), select, textarea, a[href]',
        ));
        const first = items[0], last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}>
      <header className="pet-settings-header">
        <div><Settings2 size={19} /><strong>设置</strong><span> / {heading}</span></div>
        <button className="pet-icon" type="button" aria-label="关闭设置" title="关闭设置" onClick={close}><X size={19} /></button>
      </header>
      <div className="pet-settings-body">
        <nav className="pet-settings-nav" aria-label="设置分类">
          {TABS.map(({ id, label, icon: Icon }) => <button key={id} type="button"
            aria-current={id === tab ? "page" : undefined} onClick={() => { setTab(id); setError(""); setNotice(""); }}>
            <Icon size={17} /><span>{label}</span><ChevronRight size={14} />
          </button>)}
        </nav>
        <section className="pet-settings-content" key={tab}>
          <div className="pet-section-heading"><span className="pet-eyebrow">PREFERENCES</span><h2>{heading}</h2></div>
          {tab === "connection" && <>
            <div className="pet-provider-title"><span className="pet-provider-mark">D</span><div><h3>DeepSeek</h3><span>对话回复</span></div>
              <span className="pet-small-tag">{initial.deepseek_key_configured ? "已配置" : "待配置"}</span></div>
            <SecretField label="DeepSeek API Key" value={deepseekKey} onChange={setDeepseekKey}
              configured={initial.deepseek_key_configured} clear={clearDeepseek} onClear={() => setClearDeepseek(!clearDeepseek)} />
            <label className="pet-field">API 地址
              <select value={draft.deepseek_base_url} onChange={e => set("deepseek_base_url", e.target.value as PetSettings["deepseek_base_url"])}>
                <option value="https://api.deepseek.com">https://api.deepseek.com</option>
                <option value="https://api.deepseek.com/v1">https://api.deepseek.com/v1</option>
              </select>
            </label>
            <label className="pet-field">对话模型
              <select value={draft.deepseek_model} onChange={e => set("deepseek_model", e.target.value as PetSettings["deepseek_model"])}>
                <option value="deepseek-flash">deepseek-flash</option>
                <option value="deepseek-v4-pro">deepseek-v4-pro</option>
                {draft.deepseek_model === "deepseek-chat" && <option value="deepseek-chat">deepseek-chat → Flash · 非思考</option>}
                {draft.deepseek_model === "deepseek-reasoner" && <option value="deepseek-reasoner">deepseek-reasoner → Flash · 思考</option>}
              </select>
            </label>
            {!["deepseek-chat", "deepseek-reasoner"].includes(draft.deepseek_model) &&
              <Toggle label="思考模式" checked={draft.deepseek_thinking} onChange={v => set("deepseek_thinking", v)} />}
            <div className="pet-setting-divider" />
            <Toggle label="允许对话内容发送至已启用的云端模型" checked={draft.cloud_consent} onChange={v => set("cloud_consent", v)} />
            <p className="pet-policy-note"><ShieldCheck size={15} />DeepSeek 接收当前消息与最近对话；启用 Jev 后，也会向 TypeSafe 发送这些内容及评估摘要。关闭后仅使用本地能力。</p>
            <button className="pet-secondary" type="button" disabled={!!testing} onClick={() => test("deepseek")}>
              {testing === "deepseek" ? <LoaderCircle className="pet-spin" size={16} /> : <Plug size={16} />}验证文本生成
            </button>
            <p className="pet-policy-note"><Info size={15} />验证仅发送固定问候，不发送聊天记录或人格提示词；会产生少量 API 费用。思考模式的生成上限包含推理消耗。</p>
            <div className="pet-two-columns">
              <label className="pet-field">生成上限 · tokens<input type="number" min="128" max="8192" step="128" value={draft.max_tokens} onChange={e => set("max_tokens", Number(e.target.value))} /></label>
              <label className="pet-field">请求超时 · 秒<input type="number" min="5" max="120" value={draft.timeout_seconds} onChange={e => set("timeout_seconds", Number(e.target.value))} /></label>
            </div>
          </>}
          {tab === "personality" && <>
            <label className="pet-field">桌宠名字<input required maxLength={20} value={draft.pet_name} onChange={e => set("pet_name", e.target.value)} /></label>
            <label className="pet-field">人格提示词<textarea className="pet-prompt" maxLength={6000} rows={9} value={draft.personality_prompt} onChange={e => set("personality_prompt", e.target.value)} /></label>
            <span className="pet-counter">{draft.personality_prompt.length} / 6000</span>
            <div className="pet-range-label"><label htmlFor="pet-temperature">表达自由度</label><output>{draft.temperature.toFixed(1)}</output></div>
            <input id="pet-temperature" type="range" min="0" max="1.5" step="0.1" value={draft.temperature} onChange={e => set("temperature", Number(e.target.value))} />
            <div className="pet-range-ends"><span>克制</span><span>活泼</span></div>
            <label className="pet-field">上下文 · 最近对话轮数<input type="number" min="1" max="10" value={draft.context_turns} onChange={e => set("context_turns", Number(e.target.value))} /></label>
            <p className="pet-policy-note"><ShieldCheck size={15} />人格偏好不覆盖安全规则、AI 身份与非诊断边界。</p>
          </>}
          {tab === "assessment" && <>
            <div className="pet-provider-title"><Cpu size={27} /><div><h3>Qwen · 本地评估</h3><span>Ollama</span></div></div>
            <Toggle label="启用本地评估" checked={draft.qwen_enabled} onChange={v => set("qwen_enabled", v)} />
            <label className="pet-field">Ollama 地址<input required value={draft.qwen_base_url} onChange={e => set("qwen_base_url", e.target.value)} /></label>
            <label className="pet-field">本地模型<input required value={draft.qwen_model} onChange={e => set("qwen_model", e.target.value)} /></label>
            <div className="pet-two-columns"><label className="pet-field">评估超时 · 秒<input type="number" min="3" max="90" value={draft.assessment_timeout_seconds} onChange={e => set("assessment_timeout_seconds", Number(e.target.value))} /></label>
              <button className="pet-secondary pet-align-end" type="button" disabled={!!testing} onClick={() => test("qwen")}>{testing === "qwen" ? <LoaderCircle className="pet-spin" size={16} /> : <Plug size={16} />}检查本地模型</button></div>
            <div className="pet-setting-divider" />
            <div className="pet-provider-title"><span className="pet-provider-mark pet-jev">J</span><div><h3>Jev · 决策建议</h3><span>TypeSafe AI</span></div></div>
            <Toggle label="启用 Jev" checked={draft.jev_enabled} onChange={v => set("jev_enabled", v)} />
            <SecretField label="TypeSafe API Key" value={jevKey} onChange={setJevKey} configured={initial.jev_key_configured} clear={clearJev} onClear={() => setClearJev(!clearJev)} />
            <label className="pet-field">固定模型版本<input required pattern={"jev-[0-9]+\\.[0-9]+\\.[0-9]+"} value={draft.jev_model} onChange={e => set("jev_model", e.target.value)} /></label>
            <div className="pet-range-label"><label htmlFor="jev-confidence">普通动作建议的最低置信度</label><output>{draft.jev_confidence.toFixed(2)}</output></div>
            <input id="jev-confidence" type="range" min="0.5" max="0.99" step="0.01" value={draft.jev_confidence} onChange={e => set("jev_confidence", Number(e.target.value))} />
            <button className="pet-secondary" type="button" disabled={!!testing} onClick={() => test("jev")}>{testing === "jev" ? <LoaderCircle className="pet-spin" size={16} /> : <Plug size={16} />}检查 Jev 服务</button>
            <p className="pet-policy-note"><Info size={15} />置信度不是临床准确率。模型建议不能降低规则安全等级；超时或无效结果不会被当作安全。</p>
          </>}
          {tab === "appearance" && <>
            <div className="pet-range-label"><label htmlFor="pet-scale">桌宠大小</label><output>{draft.pet_scale}%</output></div>
            <input id="pet-scale" type="range" min="70" max="140" step="5" value={draft.pet_scale} onChange={e => set("pet_scale", Number(e.target.value))} />
            <div className="pet-setting-divider" />
            <Toggle label="播放角色动画" checked={draft.animation_enabled} onChange={v => set("animation_enabled", v)} />
            <Toggle label="桌面窗口置顶" checked={draft.always_on_top} onChange={v => set("always_on_top", v)} />
            <h3 className="pet-subheading">动作</h3>
            <div className="pet-action-list">
              {[["wave", "打个招呼"], ["look", "东张西望"], ["think", "想一想"], ["write", "轻快记录"], ["stretch", "伸个懒腰"], ["sleep", "休息一下"]].map(([id, label]) =>
                <button key={id} type="button" className="pet-secondary" onClick={() => onPreview(id)}><ChevronRight size={15} />{label}</button>)}
            </div>
          </>}
          {tab === "channels" && <MessagingSettingsPanel />}
          {tab === "about" && <>
            <h3>本地研究预览 · v0.1</h3>
            <dl className="pet-about-list">
              <div><dt>用途</dt><dd>面向成年测试者的心理支持与初步筛查原型，不是临床诊断、治疗或紧急救助服务。</dd></div>
              <div><dt>本地数据</dt><dd>设置与对话保存在本机用户目录的 EcoCompanion/pet 中。对话与调试轨迹未加密；本版不适合存储真实敏感个案。</dd></div>
              <div><dt>密钥</dt><dd>{initial.credential_storage === "windows-dpapi" ? "使用 Windows DPAPI 保护，绑定当前 Windows 用户。" : "当前平台密钥仅保留在后端进程内，重启后需要重新配置。"}</dd></div>
              <div><dt>云端</dt><dd>仅在明确允许后调用已启用的云端模型。调用可能产生费用，服务商的数据处理条款仍然适用。</dd></div>
              <div><dt>评估</dt><dd>Qwen 当前使用通用本地权重，尚未完成心理领域微调；Jev 中文决策能力尚待专项验证。</dd></div>
              <div><dt>角色与动画</dt><dd>角色 OC「溟月」出自画师上善无形。动画参考 dsh-pet-indesktop，由 PC2005-cloud/dsh-pet 社区整理制作。本版精选 7 段原始动画。</dd></div>
            </dl>
            <a className="pet-source-link" href="https://github.com/PC2005-cloud/dsh-pet" target="_blank" rel="noreferrer">素材来源 <ChevronRight size={14} /></a>
            <p className="pet-policy-note"><Info size={15} />素材沿用原非商业使用限制。公开比赛、分发或商业使用前，应另外确认适用授权。</p>
          </>}
        </section>
      </div>
      <footer className="pet-settings-footer">
        <div role="status" className={error ? "pet-error" : "pet-positive"}>
          {error || (notice && <><Check size={15} />{notice}</>) || (dirty ? "有未保存的更改" : "")}
        </div>
        <button type="button" className="pet-secondary" onClick={close}><ArrowLeft size={15} />返回</button>
        <button type="submit" className="pet-primary" disabled={saving || !dirty}>{saving ? <LoaderCircle size={16} className="pet-spin" /> : <Save size={16} />}保存设置</button>
      </footer>
    </form>
  </div>;
}
