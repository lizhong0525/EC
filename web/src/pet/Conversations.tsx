import { useEffect, useRef, useState } from "react";
import {
  Check, ChevronDown, LoaderCircle, MessageCircle, MoreHorizontal, Pencil,
  Plus, RotateCw, Search, Trash2, X,
} from "lucide-react";
import { request, type SessionPage } from "./api";

const PAGE_SIZE = 40;

export default function Conversations({ selected, revision, disabled, onSelect, onNew, onClose }: {
  selected: string; revision: number; disabled: boolean;
  onSelect: (id: string) => void; onNew: () => void; onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const [page, setPage] = useState<SessionPage>({ items: [], total: 0, has_more: false });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [menu, setMenu] = useState("");
  const [editing, setEditing] = useState("");
  const [title, setTitle] = useState("");
  const [managing, setManaging] = useState(false);
  const generation = useRef(0);

  function reload() { setRetry(value => value + 1); }
  useEffect(() => {
    const id = ++generation.current;
    setLoading(true); setError("");
    const timer = window.setTimeout(() => {
      void request<SessionPage>("/sessions/search", { query, limit: PAGE_SIZE, offset: 0 })
        .then(result => { if (id === generation.current) setPage(result); })
        .catch(() => { if (id === generation.current) setError("对话列表加载失败"); })
        .finally(() => { if (id === generation.current) setLoading(false); });
    }, query ? 250 : 0);
    return () => { clearTimeout(timer); generation.current++; };
  }, [query, revision, retry]);
  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu("");
    window.addEventListener("click", close);
    return () => window.removeEventListener("click", close);
  }, [menu]);

  async function more() {
    if (loading) return;
    const id = generation.current;
    setLoading(true); setError("");
    try {
      const result = await request<SessionPage>("/sessions/search", {
        query, limit: PAGE_SIZE, offset: page.items.length,
      });
      if (id === generation.current) setPage(previous => ({
        ...result,
        items: [...previous.items, ...result.items.filter(
          item => !previous.items.some(old => old.session_id === item.session_id),
        )],
      }));
    } catch {
      if (id === generation.current) setError("对话列表加载失败");
    } finally { if (id === generation.current) setLoading(false); }
  }
  async function rename(id: string) {
    const value = title.trim();
    if (!value || managing) return;
    setManaging(true); setError("");
    try {
      await request(`/sessions/${id}/title`, { title: value }, "PUT");
      setEditing(""); setMenu(""); reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : "重命名失败，请重试");
    } finally { setManaging(false); }
  }
  async function remove(id: string) {
    const item = page.items.find(entry => entry.session_id === id);
    if (!item || !window.confirm(`删除对话“${item.title}”及本机上的全部消息？此操作无法撤销。`)) return;
    setManaging(true); setError("");
    try {
      await request(`/sessions/${id}`, undefined, "DELETE");
      setMenu("");
      reload();
      if (selected === id) onNew();
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除失败，请重试");
      setManaging(false);
    }
  }

  return <aside className="pet-history" id="pet-conversations" aria-label="历史对话">
    <header><strong>对话</strong><span>{loading ? "" : page.total}</span>
      <button className="pet-icon pet-history-close" title="关闭对话列表" aria-label="关闭对话列表" onClick={onClose}><X size={16} /></button>
    </header>
    <button className="pet-history-new" disabled={disabled || managing} onClick={onNew}><Plus size={16} />新对话</button>
    <label className="pet-history-search"><Search size={14} />
      <input aria-label="搜索对话" placeholder="搜索标题或首条消息" maxLength={200}
        value={query} onChange={event => setQuery(event.target.value)} />
    </label>
    <nav aria-label="对话列表" className="pet-history-items" aria-busy={loading}>
      {page.items.map(item => <div className={`pet-history-item ${selected === item.session_id ? "pet-history-current" : ""}`}
        key={item.session_id} data-session-id={item.session_id}>
        <button className="pet-history-select" data-session-id={item.session_id}
          aria-label={`打开对话：${item.title}`} aria-current={selected === item.session_id ? "page" : undefined}
          disabled={disabled || loading || managing} title={item.title} onClick={() => onSelect(item.session_id)}>
          <MessageCircle size={14} />
          <span><strong>{item.title}</strong><small>
            <time dateTime={item.updated_at}>{new Date(item.updated_at).toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" })}</time>
            <span>{Math.floor(item.message_count / 2)} 轮</span>
          </small></span>
        </button>
        <button className="pet-icon pet-history-menu-button" aria-label={`管理对话：${item.title}`}
          title="管理对话" aria-expanded={menu === item.session_id}
          disabled={disabled || managing}
          onClick={event => { event.stopPropagation(); setMenu(menu === item.session_id ? "" : item.session_id); }}>
          <MoreHorizontal size={16} />
        </button>
        {menu === item.session_id && <div className="pet-history-menu" role="menu" onClick={event => event.stopPropagation()}>
          <button type="button" role="menuitem" onClick={() => { setEditing(item.session_id); setTitle(item.title); setMenu(""); }}><Pencil size={13} />重命名</button>
          <button type="button" role="menuitem" className="pet-history-delete" onClick={() => void remove(item.session_id)}><Trash2 size={13} />删除本机记录</button>
        </div>}
      </div>)}
      {!loading && !error && !page.items.length && <p className="pet-history-empty">{query ? "没有匹配的对话" : "暂无历史对话"}</p>}
      {loading && <div className="pet-history-loading" role="status"><LoaderCircle size={16} className="pet-spin" />加载中</div>}
      {error && <div className="pet-history-empty" role="alert">{error}
        <button className="pet-icon" title="重新加载对话" aria-label="重新加载对话" onClick={reload}><RotateCw size={16} /></button>
      </div>}
      {page.has_more && !error && <button className="pet-history-more" disabled={loading} onClick={() => void more()}><ChevronDown size={14} />加载更多</button>}
    </nav>
    {editing && <div className="pet-rename-shade" role="presentation">
      <form className="pet-rename" role="dialog" aria-modal="true" aria-label="重命名对话"
        onSubmit={event => { event.preventDefault(); void rename(editing); }}>
        <h2>重命名对话</h2>
        <input autoFocus aria-label="对话标题" maxLength={48} required value={title}
          onChange={event => setTitle(event.target.value)} onKeyDown={event => { if (event.key === "Escape") setEditing(""); }} />
        <div><button type="button" className="pet-secondary" onClick={() => setEditing("")}><X size={14} />取消</button>
          <button className="pet-primary" disabled={managing || !title.trim()}><Check size={14} />保存标题</button></div>
      </form>
    </div>}
  </aside>;
}
