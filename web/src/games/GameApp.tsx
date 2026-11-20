import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Check, CircleUserRound, Download, Eye, EyeOff, Flag, Gamepad2, LoaderCircle, Play, RotateCcw, Settings2, Swords, Users, Vote, X } from "lucide-react";
import Settings from "../pet/Settings";
import { bootstrap, request, type SettingsView } from "../pet/api";
import "./games.css";

type Game = { id: string; kind: "chess" | "undercover"; revision: number; result: string | null; next_actor: number | null; comment?: string; };
type ChessGame = Game & { human: "white" | "black"; board: Record<string, string>; legal: string[]; turn: "white" | "black"; check: boolean; can_claim: boolean; reason: string; history: { uci: string; san: string }[]; pgn: string; };
type WordGame = Game & { word: string; round: number; phase: "describe" | "discuss" | "vote" | "defend"; alive: number[]; players: { id: number; name: string }[]; candidates: number[]; votes_cast: number; human_alive: boolean; events: { round: number; kind: string; player: number | null; text: string; votes?: Record<string, number> }[]; reveal?: { name: string; word: string; role: string }[]; };
const PIECES: Record<string, string> = { K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙", k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const FILES = "abcdefgh";
const PHASES = { describe: "词语描述", discuss: "自由讨论", defend: "平票加赛", vote: "匿名投票" };
const NAMES = ["你", "林舟", "阿柚", "知夏"];
const FISH = ["/games/fish-03.png", "/games/fish-02.png", "/games/fish-05.png", "/games/fish-04.png"];
const KEY = "ecocompanion.games.current";

async function api<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  return request<T>(path.startsWith("/games") ? path : `/games${path}`, body, method);
}
const message = (error: unknown) => error instanceof Error ? error.message : "请求未完成，请重试";
function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/x-chess-pgn;charset=utf-8" }));
  const a = document.createElement("a"); a.href = url; a.download = name; a.click(); URL.revokeObjectURL(url);
}
function outcome(game: Game) {
  if (game.kind === "undercover") return game.result === "draw" ? "本局平局" : game.result === "civilians" ? "平民阵营获胜" : "卧底获胜";
  return game.result === "1/2-1/2" ? "本局和棋" : ((game.result === "1-0") === ((game as ChessGame).human === "white") ? "你赢了" : "DeepSeek 获胜");
}

function Chess({ game, onUpdate, busy }: { game: ChessGame; onUpdate: (game: Game) => void; busy: boolean }) {
  const [selected, setSelected] = useState("");
  const [promotion, setPromotion] = useState<string[]>([]);
  const [error, setError] = useState("");
  const moves = useRef<HTMLDivElement>(null);
  const humanTurn = game.next_actor === 0;
  useEffect(() => { setSelected(""); setPromotion([]); }, [game.revision]);
  useEffect(() => { moves.current?.scrollTo({ top: moves.current.scrollHeight, behavior: "smooth" }); }, [game.history.length]);
  async function command(action: string, uci = "") {
    try { onUpdate(await api(`/games/${game.id}/actions`, { revision: game.revision, action, text: uci })); }
    catch (err) { setError(message(err)); }
  }
  function square(name: string) {
    if (!humanTurn || busy || promotion.length) return;
    setError("");
    if (selected) {
      const options = game.legal.filter(move => move.startsWith(`${selected}${name}`));
      if (options.some(move => move.length === 5)) { setPromotion(options); return; }
      if (options.length) { void command("move", options[0]); return; }
    }
    setSelected(game.legal.some(move => move.startsWith(name)) ? name : "");
  }
  const orientation = game.human === "white" ? [0, 1, 2, 3, 4, 5, 6, 7] : [7, 6, 5, 4, 3, 2, 1, 0];
  return <div className="chess-layout">
    <div className="chess-stage">
      <div className="opponent"><div className="player-badge fish-avatar"><img src={FISH[0]} alt="DeepSeek 的大肥鱼头像" /></div><div><strong>DeepSeek</strong><span>{humanTurn ? "等待你的落子" : "正在思考"}</span></div><span className="turn-light" data-on={!humanTurn} /></div>
      <div className="board-frame"><div className="chess-board" role="grid" aria-label="国际象棋棋盘">
        {orientation.map(row => orientation.map(col => {
          const name = `${FILES[col]}${game.human === "white" ? 8 - row : row + 1}`;
          const piece = game.board[name];
          const target = selected && game.legal.some(move => move.startsWith(`${selected}${name}`));
          return <button key={name} role="gridcell" aria-label={`${name} ${piece || "空格"}`}
            className={`square ${(col + row) % 2 === 0 ? "dark" : "light"} ${selected === name ? "selected" : ""} ${game.history.at(-1)?.uci.endsWith(name) ? "last" : ""} ${target ? "target" : ""}`}
            disabled={!humanTurn || busy} onClick={() => square(name)}>
            {target && <i className="move-dot" />}
            {piece && <span className={`piece ${piece === piece.toUpperCase() ? "white" : "black"}`}>{PIECES[piece]}</span>}
            {col === 0 && <small className="rank-label">{name[1]}</small>}
            {row === 7 && <small className="file-label">{name[0]}</small>}
          </button>;
        }))}
      </div></div>
      <div className="opponent"><div className="player-badge human"><CircleUserRound size={21} /></div><div><strong>你 · {game.human === "white" ? "白方" : "黑方"}</strong><span>{game.result ? outcome(game) : game.check ? "将军！请保护你的王" : humanTurn ? "轮到你落子" : "对手回合"}</span></div><span className="turn-light" data-on={humanTurn} /></div>
      {promotion.length > 0 && <div className="promotion" role="dialog" aria-label="选择升变棋子">{["q", "r", "b", "n"].map(p => <button key={p} disabled={busy} onClick={() => { const uci = `${selected}${promotion[0].slice(2, 4)}${p}`; setPromotion([]); void command("move", uci); }}>{PIECES[game.human === "white" ? p.toUpperCase() : p]}</button>)}</div>}
      {error && <div className="game-error" role="alert">{error}</div>}
    </div>
    <aside className="chess-sidebar">
      <div className="game-status"><span>对局记录</span><span>{Math.ceil(game.history.length / 2)} 回合</span></div>
      <div className="move-history" ref={moves}>{game.history.length === 0 ? <div className="empty-history"><Swords size={29} /><p>棋局已就绪</p><span>你执{game.human === "white" ? "白先行" : "黑，DeepSeek 先行"}</span></div> : Array.from({ length: Math.ceil(game.history.length / 2) }, (_, i) => <div className="move-row" key={i}><span>{i + 1}.</span><span>{game.history[i * 2]?.san}</span><span>{game.history[i * 2 + 1]?.san}</span></div>)}</div>
      {game.comment && <div className="chess-comment"><span>DeepSeek 棋评</span><p>{game.comment}</p></div>}
      <div className="chess-actions">
        {!game.result && game.can_claim && <button onClick={() => void command("claim")} disabled={busy || !humanTurn}><Check size={16} />提和</button>}
        {!game.result && <button onClick={() => { if (confirm("确定认输？本局将立即结束。")) void command("resign"); }} disabled={busy}><Flag size={16} />认输</button>}
        {game.result && <span className="finished"><Check size={16} />{outcome(game)}</span>}
        <button title="下载 PGN 棋谱" aria-label="下载 PGN 棋谱" onClick={() => download(`${game.id}.pgn`, game.pgn)}><Download size={17} /></button>
      </div>
      <div className="notation-note">标准规则 · 本地校验 · 自动保存</div>
    </aside>
  </div>;
}

function Undercover({ game, onUpdate, busy }: { game: WordGame; onUpdate: (game: Game) => void; busy: boolean }) {
  const [draft, setDraft] = useState("");
  const [vote, setVote] = useState<number | null>(null);
  const [hidden, setHidden] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState(0);
  const log = useRef<HTMLDivElement>(null);
  useEffect(() => { setVote(null); setDraft(""); }, [game.round, game.phase]);
  useEffect(() => { log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" }); }, [game.events.length, game.revision]);
  async function submit(target?: number) {
    setError("");
    try {
      onUpdate(await api(`/games/${game.id}/actions`, { revision: game.revision, action: target === undefined ? "speak" : "vote", text: draft, target }));
      setDraft(""); setVote(null);
    } catch (err) { setError(message(err)); }
  }
  const shown = game.events.filter(e => filter === 0 || e.round === filter);
  return <div className="word-layout">
    <div className="word-main">
      <div className="word-statusbar"><span>第 {game.round} 轮</span><strong>{game.result ? "身份揭晓" : PHASES[game.phase]}</strong><span>{game.alive.length} 人在场</span></div>
      <div className="secret-word"><div><span>仅你可见的词语</span><strong>{hidden ? "••••••" : game.word}</strong></div><button title={hidden ? "显示我的词" : "隐藏我的词"} aria-label={hidden ? "显示我的词" : "隐藏我的词"} onClick={() => setHidden(value => !value)}>{hidden ? <Eye size={20} /> : <EyeOff size={20} />}</button><span className="word-confidential">三位 AI 各自持有独立信息</span></div>
      <div className="players-row">{game.players.map(player => <div className={`word-player ${game.alive.includes(player.id) ? "" : "eliminated"}`} key={player.id}><div className={`player-avatar ${player.id > 0 ? "fish-avatar" : ""}`}>{player.id === 0 ? <CircleUserRound size={24} /> : <img src={FISH[player.id]} alt={`${player.name}的大肥鱼头像`} />}{game.next_actor === player.id && <i />}</div><strong>{player.name}</strong><span>{!game.alive.includes(player.id) ? "已出局" : player.id === 0 ? "人类玩家" : "DeepSeek 子 Agent"}</span></div>)}</div>
      <div className="discussion-toolbar"><span>公开发言</span><select aria-label="筛选轮次" value={filter} onChange={e => setFilter(Number(e.target.value))}><option value={0}>全部轮次</option>{Array.from({ length: game.round }, (_, i) => <option key={i} value={i + 1}>第 {i + 1} 轮</option>)}</select></div>
      <div className="discussion" ref={log} aria-live="polite">{shown.length === 0 ? <div className="empty-discussion"><Users size={33} /><p>每人依次描述自己的词</p><span>接着讨论，最后投票找出卧底</span></div> : shown.map((event, i) => <div className={`word-event ${event.kind === "system" || event.kind === "eliminated" ? "system-event" : ""}`} key={`${game.round}-${i}`}>
        {event.kind === "ballot" ? <><div className="event-heading"><Vote size={14} />投票已公开</div>{Object.entries(event.votes || {}).map(([p, t]) => <p key={p}>{NAMES[Number(p)]} → {NAMES[t]}</p>)}</>
          : event.player === null ? <p>{event.text}</p>
          : <><div className="event-heading"><strong>{NAMES[event.player]}</strong><span>第 {event.round} 轮 · {PHASES[event.kind as keyof typeof PHASES]}</span></div><p>{event.text}</p>{event.kind === "eliminated" && <span className="event-tag">身份已公开</span>}</>}
      </div>)}</div>
      {game.result && <div className={`game-verdict ${game.result === "civilians" && game.reveal?.[0].role === "平民" ? "won" : ""}`}><h3>{outcome(game)}</h3><div>{game.reveal?.map(p => <span key={p.name}><b>{p.name}</b>{p.role} · {p.word}</span>)}</div></div>}
      {error && <div className="game-error" role="alert">{error}</div>}
      {!game.result && game.phase !== "vote" && <form className="word-composer" onSubmit={e => { e.preventDefault(); void submit(); }}><textarea aria-label="我的发言" placeholder={game.human_alive ? "用一句不泄词的话描述你的词语…" : "你已出局，可继续观战"} disabled={!game.human_alive || game.next_actor !== 0 || busy} maxLength={160} rows={2} value={draft} onChange={e => setDraft(e.target.value)} /><button type="submit" disabled={!draft.trim() || game.next_actor !== 0 || busy}>{game.phase === "defend" ? "提交辩护" : "提交发言"}<Play size={15} /></button></form>}
      {!game.result && game.phase === "vote" && <div className="voting"><div><Vote size={17} /><strong>选择要淘汰的玩家</strong><span>{game.votes_cast}/{game.alive.length} 已投票</span></div><div className="vote-options">{game.candidates.filter(p => p !== 0).map(p => <button key={p} className={vote === p ? "voted" : ""} disabled={!game.human_alive || game.next_actor !== 0 || busy} onClick={() => setVote(p)}>{NAMES[p]}{vote === p && <Check size={15} />}</button>)}</div><button className="vote-submit" onClick={() => vote !== null && void submit(vote)} disabled={vote === null || game.next_actor !== 0 || busy}>确认投票</button></div>}
    </div>
    <aside className="word-sidebar"><div className="agent-heading"><span>本局玩家</span><span>4 人局</span></div>
      <div className="agent-rule"><span>平民</span><strong>3 人</strong></div><div className="agent-rule"><span>卧底</span><strong>1 人</strong></div><div className="agent-rule"><span>白板</span><strong>无</strong></div>
      <div className="round-guide"><span className={game.phase === "describe" ? "current" : ""}>01<span>描述</span></span><span className={game.phase === "discuss" ? "current" : ""}>02<span>讨论</span></span><span className={game.phase === "vote" ? "current" : ""}>03<span>投票</span></span></div>
      <div className="private-note"><EyeOff size={16} /><p>你的身份不会提前显示。结合发言判断自己与他人是否同阵营。</p></div>
      {game.result && <button className="new-inline" onClick={() => { localStorage.removeItem(KEY); location.reload(); }}><RotateCcw size={15} />再来一局</button>}
    </aside>
  </div>;
}

function Lobby({ onStart, busy }: { onStart: (kind: string, color: string) => void; busy: boolean }) {
  const [color, setColor] = useState("white");
  return <main className="lobby"><div className="lobby-heading"><p>小溟 · 轻松一局</p><h1>想好接下来玩什么了吗？</h1></div>
    <div className="game-choices">
      <section className="game-choice chess-choice"><div className="choice-visual"><div className="mini-board">{[..."rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR"].map((p, i) => <span key={i} className={p === "." ? "" : p === p.toUpperCase() ? "white" : "black"}>{PIECES[p]}</span>)}</div></div><div className="choice-body"><span className="choice-number">01 / CHESS</span><h2>国际象棋</h2><p>你来执棋，DeepSeek 应战。标准规则，慢慢思考。</p><div className="color-select" aria-label="选择棋色"><button className={color === "white" ? "selected" : ""} onClick={() => setColor("white")}><span className="white-piece">♙</span>执白先行</button><button className={color === "black" ? "selected" : ""} onClick={() => setColor("black")}><span>♟</span>执黑后行</button></div><button className="start-game" disabled={busy} onClick={() => onStart("chess", color)}>{busy ? <LoaderCircle className="pet-spin" size={18} /> : <Play size={18} />}开始对局</button></div></section>
      <section className="game-choice word-choice"><div className="choice-visual"><img className="lobby-fish lobby-fish-happy" src={FISH[1]} alt="" /><img className="lobby-fish lobby-fish-dreamy" src={FISH[2]} alt="" /><img className="lobby-fish lobby-fish-clever" src={FISH[3]} alt="" /><div className="word-pair"><div><span>你的词</span><strong>?</strong></div><div><span>大家的词</span><strong>?</strong></div><div><span>谁的词</span><strong>?</strong></div><div><span>卧底是谁</span><strong>?</strong></div></div></div><div className="choice-body"><span className="choice-number">02 / UNDERCOVER</span><h2>谁是卧底</h2><p>你与三位独立 AI 玩家，用描述、推理和投票找出词语不同的人。</p><div className="game-facts"><span>4 人局</span><span>1 名卧底</span><span>无白板</span></div><button className="start-game" disabled={busy} onClick={() => onStart("undercover", color)}>{busy ? <LoaderCircle className="pet-spin" size={18} /> : <Play size={18} />}开始游戏</button></div></section>
    </div>
  </main>;
}

export default function GameApp() {
  const [game, setGame] = useState<Game | null>(null);
  const [config, setConfig] = useState<SettingsView | null>(null);
  const [settings, setSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const pending = useRef(false);
  const refresh = async (id: string) => { const current = await api<Game>(`/games/${id}`, undefined, "GET"); setGame(current); return current; };
  useEffect(() => {
    document.title = "游戏间";
    void bootstrap().then(setConfig).catch(err => setError(message(err)));
    const id = localStorage.getItem(KEY);
    if (id) void refresh(id).catch(() => localStorage.removeItem(KEY));
  }, []);
  useEffect(() => {
    if (!game || game.result || game.next_actor === null || game.next_actor === 0 || pending.current) return;
    pending.current = true; setBusy(true); setError("");
    void (async () => {
      try {
        let current = game;
        // Keep a resumable checkpoint after each independent player action.
        for (let i = 0; i < 4 && current.next_actor !== null && current.next_actor !== 0 && !current.result; i++) {
          current = await api<Game>(`/games/${current.id}/actions`, { revision: current.revision, action: "advance" });
          setGame(current);
        }
        setError("");
      } catch (err) {
        setError(message(err));
        // Reconcile the saved turn after a lost connection or failed AI action.
        try { setGame(await refresh(game.id)); } catch { /* The user can retry explicitly. */ }
      } finally { pending.current = false; setBusy(false); }
    })();
  }, [game]);
  async function start(kind: string, color: string) {
    setBusy(true); setError("");
    try { const next = await api<Game>("/games", { kind, color }); localStorage.setItem(KEY, next.id); setGame(next); }
    catch (err) { setError(message(err)); }
    finally { setBusy(false); }
  }
  function update(next: Game) { setGame(next); setError(""); }
  return <main className="games-app"><header className="games-header"><div className="games-brand"><span><Gamepad2 size={21} /></span><strong>游戏间</strong></div><div className="games-header-right"><span className="connection"><i />本地游戏 · DeepSeek 在线</span>{game && <button onClick={() => setGame(null)} title="返回游戏选择"><ArrowLeft size={16} />游戏选择</button>}<button className="icon-button" title="DeepSeek 设置" aria-label="DeepSeek 设置" onClick={() => setSettings(true)}><Settings2 size={19} /></button></div></header>
    <div className="games-content">{error && <div className="top-error" role="alert">{error}{!busy && <button onClick={() => location.reload()}>重试</button>}<button aria-label="关闭提示" onClick={() => setError("")}><X size={15} /></button></div>}{!game ? <Lobby onStart={start} busy={busy} /> : <>
      <div className="game-topline"><div><span className="game-eyebrow">{game.kind === "chess" ? "CHESS" : "UNDERCOVER"}</span><h1>{game.kind === "chess" ? "国际象棋" : "谁是卧底"}</h1></div>{game.result && <span className="verdict-label">{outcome(game)}</span>}{!game.result && game.next_actor !== 0 && <span className="next-turn">{busy ? <LoaderCircle size={16} className="pet-spin" /> : <Play size={16} />}{busy ? "等待 AI" : "自动继续"}</span>}<button className="reset-button" title="开始新游戏" onClick={() => { if (game.result || confirm("当前对局会保存在本机，确定开始新游戏？")) void start(game.kind, game.kind === "chess" ? (game as ChessGame).human : "white"); }}><RotateCcw size={16} />新对局</button></div>
      {game.kind === "chess" ? <Chess game={game as ChessGame} onUpdate={update} busy={busy} /> : <Undercover game={game as WordGame} onUpdate={update} busy={busy} />}
    </>}</div>
    {settings && config && <Settings initial={config} onSaved={next => { setConfig(next); setSettings(false); }} onClose={() => setSettings(false)} onPreview={() => {}} />}
  </main>;
}
