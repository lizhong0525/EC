import { useEffect, useState } from "react";
import { Check, Download, LoaderCircle, Search, UserRoundCheck, UsersRound, X } from "lucide-react";
import { request, type MessagingSettings } from "./api";
import { Toggle } from "./Settings";

type PersonalSettings = MessagingSettings["personal_wechat"];
type Identity = { wxid: string; name: string };
type Discovery = { account: Identity; contacts: Identity[] };

export default function WeChatIdentityPicker({ value, disabled, onChange }: {
  value: PersonalSettings;
  disabled: boolean;
  onChange: (value: Partial<PersonalSettings>) => void;
}) {
  const [account, setAccount] = useState<Identity | null>(null);
  const [contacts, setContacts] = useState<Identity[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState("");
  const [installNeeded, setInstallNeeded] = useState(false);
  const [installProgress, setInstallProgress] = useState("");
  const [notice, setNotice] = useState("");
  const selected = new Set(value.allowed_chats);

  useEffect(() => {
    if (!installing) return;
    const timer = window.setInterval(() => {
      void request<{ running: boolean; detail: string }>(
        "/messaging/wechat-bridge/install", undefined, "GET",
      ).then(status => setInstallProgress(status.detail)).catch(() => {});
    }, 1000);
    return () => window.clearInterval(timer);
  }, [installing]);

  async function discover(search: string) {
    setLoading(true); setError(""); setNotice("");
    try {
      const result = await request<Discovery>("/messaging/wechat-contacts", {
        confirmed: value.test_account_confirmed,
        account_wxid: value.account_wxid || account?.wxid || "",
        query: search,
      });
      setAccount(result.account);
      setContacts(result.contacts);
      setInstallNeeded(false);
      const accountChanged = !!value.account_wxid && value.account_wxid !== result.account.wxid;
      onChange({
        account_wxid: result.account.wxid,
        ...(accountChanged ? { allowed_chats: [] } : {}),
      });
      setNotice(accountChanged
        ? "测试账号已更换，请重新选择好友"
        : result.contacts.length
          ? `${search ? "找到" : "已读取"} ${result.contacts.length} 个私聊联系人，请选择测试好友`
          : search ? "没有匹配的私聊好友，可清除搜索后查看全部好友" : "此账号暂未读取到私聊好友");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "账号与好友读取失败";
      setError(detail);
      setInstallNeeded(detail.includes("安装桥接组件") || detail.includes("尚未安装可选依赖"));
    } finally { setLoading(false); }
  }

  async function install() {
    if (!window.confirm("将安装固定版本的微信桥接组件，用于本地账号与好友读取及测试消息收发。是否继续？")) return;
    setInstalling(true); setInstallProgress("正在启动桥接组件安装"); setError(""); setNotice("");
    try {
      const result = await request<{ ok: boolean; detail: string }>("/messaging/wechat-bridge/install", {});
      setInstallNeeded(false); setNotice(result.detail);
    } catch (err) {
      setError(err instanceof Error ? err.message : "桥接组件安装失败");
    } finally { setInstalling(false); setInstallProgress(""); }
  }

  function toggleContact(contact: Identity) {
    if (selected.has(contact.wxid)) {
      onChange({ allowed_chats: value.allowed_chats.filter(wxid => wxid !== contact.wxid) });
      return;
    }
    if (selected.size >= 20) { setError("最多选择 20 个测试好友"); return; }
    setError("");
    onChange({ allowed_chats: [...value.allowed_chats, contact.wxid] });
  }

  return <div className="pet-wechat-setup">
    <Toggle label="我确认使用隔离测试账号与测试聊天" checked={value.test_account_confirmed}
      onChange={test_account_confirmed => {
        setAccount(null); setContacts([]); setError(""); setNotice("");
        onChange({ test_account_confirmed });
      }} />
    <Toggle label="启用个人微信测试桥接" checked={value.enabled}
      onChange={enabled => onChange({ enabled })} />
    <div className="pet-wechat-read-row">
      <button type="button" className="pet-secondary" disabled={disabled || loading || !value.test_account_confirmed}
        onClick={() => { setQuery(""); void discover(""); }}>
        {loading ? <LoaderCircle className="pet-spin" size={15} /> : <UserRoundCheck size={15} />}
        {account ? "刷新全部好友" : "读取账号与全部好友"}
      </button>
      <span className="pet-muted">只读，不启动回复</span>
    </div>
    {account && <div className="pet-wechat-account">
      <Check size={16} /><div><strong>{account.name || "已读取测试账号"}</strong>
        <span>当前登录账号 · 已自动识别内部 ID</span></div>
    </div>}
    {account && <>
      <div className="pet-wechat-search" role="search">
        <Search size={15} />
        <input aria-label="搜索测试好友" maxLength={80} value={query}
          placeholder="搜索好友备注或昵称" onChange={event => setQuery(event.target.value)}
          onKeyDown={event => {
            if (event.key !== "Enter" || event.nativeEvent.isComposing) return;
            event.preventDefault();
            event.stopPropagation();
            if (!loading) void discover(query);
          }} />
        <button type="button" className="pet-icon" title="搜索好友" aria-label="搜索好友"
          disabled={loading} onClick={() => void discover(query)}>
          <Search size={15} />
        </button>
      </div>
      {contacts.length === 0 && <div className="pet-wechat-empty" role="status">
        <UsersRound size={20} /><span>{query ? "没有匹配的好友" : "暂未读取到私聊好友"}</span>
        {query && <button type="button" className="pet-secondary" disabled={loading}
          onClick={() => { setQuery(""); void discover(""); }}>查看全部好友</button>}
      </div>}
      {contacts.length > 0 && <div className="pet-wechat-contacts" aria-label="选择测试好友">
        {contacts.map(contact => <label key={contact.wxid}>
          <input type="checkbox" checked={selected.has(contact.wxid)} disabled={disabled}
            onChange={() => toggleContact(contact)} />
          <span>{contact.name || "未命名联系人"}</span>
        </label>)}
      </div>}
      {selected.size > 0 && <div className="pet-wechat-selected" aria-label="已选择的测试好友">
        <span>已允许 {selected.size} 位测试好友</span>
        <div>{value.allowed_chats.map(wxid => {
          const contact = contacts.find(item => item.wxid === wxid);
          return <span key={wxid}>{contact?.name || "已保存的测试好友"}
            <button type="button" title={`移除${contact?.name || "测试好友"}`}
              aria-label={`移除${contact?.name || "测试好友"}`}
              onClick={() => onChange({ allowed_chats: value.allowed_chats.filter(item => item !== wxid) })}>
              <X size={12} />
            </button>
          </span>;
        })}</div>
      </div>}
      {value.allowed_chats.length === 0 && value.enabled &&
        <p className="pet-error" role="status">请至少选择一位测试好友，才能保存并启用桥接</p>}
    </>}
    {notice && <p className="pet-positive" role="status">{notice}</p>}
    {error && <p className="pet-error" role="alert">{error}</p>}
    {installNeeded && <button type="button" className="pet-secondary" disabled={installing}
      onClick={() => void install()}>
      {installing ? <LoaderCircle className="pet-spin" size={15} /> : <Download size={15} />}
      {installing ? "安装中，请保持应用开启" : "安装桥接组件"}
    </button>}
    {installing && <p className="pet-muted" role="status">{installProgress}</p>}
  </div>;
}
