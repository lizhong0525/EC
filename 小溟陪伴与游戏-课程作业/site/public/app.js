"use strict";
const $ = (selector) => document.querySelector(selector);
const main = $("#main");
const escapeHTML = value => String(value ?? "").replace(/[&<>"']/g, char =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
const fish = [3,2,5,4];
const pieces = {P:"♟",R:"♜",N:"♞",B:"♝",Q:"♛",K:"♚",p:"♟",r:"♜",n:"♞",b:"♝",q:"♛",k:"♚"};
const pieceNames = {p:"兵",r:"车",n:"马",b:"象",q:"后",k:"王"};
const phases = {describe:"描述词语",discuss:"自由讨论",defend:"平票辩护",vote:"匿名投票"};
const titles = {home:"首页",chat:"小溟对话",games:"游戏间",chess:"国际象棋",undercover:"谁是卧底",settings:"连接设置",about:"使用说明"};
let state, busy = false, selected = "", promotions = [], problem = "", chatDraft = "", wordDraft = "";
let chatMode, token = "", retryText = "", noticeTimer;
const route = () => Object.hasOwn(titles, location.hash.slice(1)) ? location.hash.slice(1) : "home";
const modeLabel = mode => mode === "cloud" ? "DeepSeek 联网" : "本地规则";
function message(text) {
  $("#notice").textContent = text;
  clearTimeout(noticeTimer);
  if (text) noticeTimer = setTimeout(() => { $("#notice").textContent = ""; }, 5500);
}
async function api(path, body) {
  const options = body ? {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)} : {cache:"no-store"};
  let res;
  try { res = await fetch("/api/" + path, options); }
  catch { throw Error("无法连接本机服务，请检查启动窗口并重试。"); }
  const data = await res.json();
  if (!res.ok) throw Error(data.error || "请求失败，请重试。");
  return data;
}
const disabled = () => busy ? " disabled" : "";
const altMode = () => state.settings.cloud ? "cloud" : "local";
function modeSelect(value, name="mode") {
  return '<label>运行方式 <select name="' + name + '"' + disabled() + '><option value="local"' +
    (value === "local" ? " selected" : "") + '>本地规则</option><option value="cloud"' +
    (state.settings.cloud ? "" : " disabled") + (value === "cloud" ? " selected" : "") + '>DeepSeek 联网</option></select></label>';
}
function settingsForm() {
  const cfg = state.settings;
  return '<form id="settingsForm"><label class="stack">DeepSeek API 密钥<input id="key" type="password" maxlength="512" autocomplete="off" placeholder="' +
    (cfg.configured ? "已配置，可留空沿用" : "输入 API Key") + '"></label><label class="check"><input id="consent" type="checkbox"' +
    (cfg.consent ? " checked" : "") + '>允许将本次联网对话或游戏回合发送给 DeepSeek</label><button' + disabled() + '>保存连接设置</button></form>';
}
function miniBoard() {
  const rows = "rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR";
  return '<div class="mini" aria-hidden="true">' + Array.from(rows, c => '<i>'+(pieces[c] || "")+'</i>').join("") + '</div>';
}
function gameCards() {
  return '<div class="cards"><article class="card"><div class="art">' + miniBoard() + '</div><h2>国际象棋</h2><p>选白先行，或执黑应战。查看合法落子，在棋谱里回顾每一步。</p><a class="action second" href="#chess">进入棋桌</a></article>' +
    '<article class="card"><div class="art word">' + fish.map(n => '<img alt="" src="assets/fish-0'+n+'.png">').join("") + '</div><h2>谁是卧底</h2><p>四个人，两种词。描述自己的线索，听取发言，投票找出卧底。</p><a class="action second" href="#undercover">进入词语局</a></article></div>';
}
function home() {
  const setup = state.settings.cloud ? "" : '<section class="panel setupmini"><div><h2>第一步 · 设置联网对话</h2><p>填写密钥可启用 DeepSeek，也可直接体验本地游戏。</p></div><details><summary>填写密钥与授权</summary>' + settingsForm() + '</details></section>';
  return setup + '<section class="hero"><div class="heroCopy"><h1>和小溟聊一会儿</h1><p>聊聊今天发生的小事，或坐到游戏桌边，开始一局。</p><div class="row"><a class="action" href="#chat">开始对话</a><a class="action second" href="#games">去游戏间</a></div></div><span class="sun"></span><img class="heroArt" src="assets/portrait.png" alt="小溟的角色肖像"></section><h2 class="space">游戏间</h2>' + gameCards();
}
function chat() {
  const logs = state.chat.map(t => '<div class="bubble user"><b>你</b>' + escapeHTML(t.user) + '</div><div class="bubble"><b>小溟 · ' + modeLabel(t.mode) + '</b>' + escapeHTML(t.answer) + '</div>').join("");
  return '<h1>小溟对话</h1><div class="chatlayout"><section class="panel"><div class="topline"><h2>对话记录</h2>'+modeSelect(chatMode,"chatMode")+'</div><div class="chatlog" id="log" role="log">'+ (logs || '<p>小溟在这里。今天想从哪里说起？</p>') + '</div><form id="chatForm"><label for="chatText">写给小溟</label><textarea id="chatText" rows="3" maxlength="1500" placeholder="说说今天的心情…" required'+disabled()+'>'+escapeHTML(chatDraft)+'</textarea><button'+disabled()+'>'+(busy?"等待回复…":problem?"重试发送":"发送")+'</button></form></section><aside class="panel"><img class="sidepic" src="assets/portrait.png" alt="小溟"><h2>当前陪伴方式</h2><p>'+(chatMode==="cloud"?"DeepSeek 根据你最近六轮联网对话生成回复。":"本地规则按问候、日常感受和游戏等话题选择简短回应。")+'</p><a href="#settings">管理联网授权</a></aside></div>';
}
function gameStart(kind) {
  const g = state.games[kind];
  const humanColor = g?.human || "white";
  const whiteSelected = humanColor === "white" ? " selected" : "";
  const blackSelected = humanColor === "black" ? " selected" : "";
  const color = kind === "chess"
    ? '<label>执棋颜色 <select name="color"><option value="white"' + whiteSelected +
      '>执白先行</option><option value="black"' + blackSelected +
      '>执黑后行</option></select></label>'
    : "";
  return '<form id="startForm" class="row actions">' + color + modeSelect(g?.mode || altMode()) + '<button'+disabled()+'>'+(g?"重新开局":"开始游戏")+'</button></form>';
}
function statusLine(g) {
  if (g.result) {
    const outcomes = {"1-0":"白方获胜","0-1":"黑方获胜","1/2-1/2":"和棋", civilians:"平民阵营获胜",undercover:"卧底阵营获胜",draw:"平局"};
    return '<p class="result">本局结束 · ' + escapeHTML(outcomes[g.result] || g.result) + '</p>';
  }
  return '<p>'+modeLabel(g.mode)+' · '+ (g.next_actor === 0 ? "现在轮到你" : busy ? "其他玩家正在行动…" : "等待其他玩家行动") + (g.check ? " · 将军" : "")+'</p>';
}
function chessContent(g) {
  let files = [..."abcdefgh"], ranks = [8,7,6,5,4,3,2,1];
  if (g.human === "black") {files.reverse(); ranks.reverse();}
  let cells = "";
  for (const rank of ranks) for (const file of files) {
    const sq = file + rank, p = g.board[sq], target = selected && g.legal.some(m=>m.startsWith(selected+sq));
    const color = ((file.charCodeAt(0)-97 + rank)%2 === 1) ? " dark" : "";
    const label = sq + (p ? (p === p.toUpperCase() ? " 白" : " 黑") + pieceNames[p.toLowerCase()] : " 空格");
    cells += '<button class="sq'+color+(selected===sq?" selected":"")+(target?" valid":"")+'" data-square="'+sq+'" data-white="'+Boolean(p&&p===p.toUpperCase())+'" aria-label="'+label+'" aria-pressed="'+(selected===sq)+'">'+(pieces[p]||"")+'<small>'+sq+'</small></button>';
  }
  const promo = promotions.length ? '<div class="promote"><strong>选择升变棋子 </strong>'+promotions.map(m=>'<button data-promote="'+m+'">'+pieceNames[m[4]]+'</button>').join("")+'</div>' : '';
  const log = g.history.map((h,i)=>'<li>' + (i%2?"黑":"白")+' · '+escapeHTML(h.san)+'</li>').join("");
  return '<div class="boardlayout"><div><div class="board" aria-label="国际象棋棋盘">'+cells+'</div>'+promo+'</div><aside class="panel"><h2>对局记录</h2>'+statusLine(g)+'<p>你执'+(g.human==="white"?"白":"黑")+'。先选棋子，再选带圆点的目标格。</p><ol class="history">'+log+'</ol>'+(!g.result ? '<div class="row"><button class="second" data-act="resign"'+disabled()+'>认输</button>' + (g.can_claim&&g.next_actor===0?'<button data-act="claim">提和</button>':'')+'</div>' : '')+'</aside></div>';
}
function wordContent(g) {
  const people = g.players.map(p=>'<div class="player '+(g.alive.includes(p.id)?"":"out")+'"><img src="assets/fish-0'+fish[p.id]+'.png" alt=""><b>'+p.name+'</b><span>'+(g.alive.includes(p.id)?"在场":"出局")+'</span></div>').join("");
  const events = g.events.map(e=>{const ballot = e.votes ? " · " + Object.entries(e.votes).map(([from,to])=>g.players[from].name+" → "+g.players[to].name).join("，") : "";return '<li><small>第 '+e.round+' 轮</small>　<b>'+(e.player===null?"主持":g.players[e.player].name)+'</b>　'+escapeHTML(e.text+ballot)+'</li>';}).join("");
  let action = "";
  if (!g.result && g.next_actor===0) {
    if (g.phase === "vote") action = '<strong>投给哪位玩家？</strong><div class="row">'+g.candidates.filter(i=>i!==0).map(i=>'<button data-vote="'+i+'"'+disabled()+'>'+g.players[i].name+'</button>').join("")+'</div>';
    else action = '<form id="speakForm"><label for="wordText">你的发言</label><textarea id="wordText" rows="2" maxlength="160" required'+disabled()+'>'+escapeHTML(wordDraft)+'</textarea><button'+disabled()+'>提交发言</button></form>';
  }
  const reveal = g.reveal ? '<h3>身份揭晓</h3>'+g.reveal.map(p=>'<p>'+p.name+' · '+p.role+' · '+p.word+'</p>').join("") : '';
  return '<div class="columns"><section class="panel"><h2>公开发言</h2>'+statusLine(g)+'<ol class="wordlog">'+events+'</ol>'+action+'</section><aside class="panel"><h2>第 '+g.round+' 轮 · '+phases[g.phase]+'</h2><p>你的词语　<strong>'+escapeHTML(g.word)+'</strong></p><div class="playerlist">'+people+'</div><p class="space">三名平民，一名卧底。描述后讨论，再匿名投票。平票者辩护并重新投票。</p>'+reveal+'</aside></div>';
}
function render() {
  const r = route(), inGame = r === "chess" || r === "undercover";
  document.title = titles[r] + " · 小溟的房间";
  for (const a of document.querySelectorAll("nav a")) {
    if(a.hash === "#"+(inGame?"games":r)) a.setAttribute("aria-current","page"); else a.removeAttribute("aria-current");
  }
  $("#connection").textContent = state.settings.cloud ? "已授权 · 设置" : "连接设置";
  let content;
  if (r==="home") content = home();
  if (r==="games") content = '<h1>游戏间</h1><p class="sectionLead">选一张游戏桌，和小溟的伙伴们开始一局。</p>'+gameCards();
  if (r==="chat") content = chat();
  if (r==="settings") content = '<h1>连接设置</h1><section class="panel narrow"><h2>DeepSeek 联网授权</h2><p>Windows 在本机加密保存密钥。授权后可以在对话与游戏中选择联网模式。</p>'+settingsForm()+'</section>';
  if (r==="about") content = '<h1>使用说明与素材来源</h1><section class="panel narrow"><h2>在本机使用</h2><p>首页进入聊天或游戏，可选择本地规则体验。联网使用 DeepSeek，必须先配置密钥并授权。聊天和对局进度由本地服务保存。</p><h2>角色与游戏</h2><p>角色溟月原画由上善无形创作，取自 dsh-pet 社区。肖像、动作与游戏头像沿用 EcoCompanion 项目。</p><p><a href="assets/ATTRIBUTION.md">查看角色素材原始署名</a></p><p>棋盘合法性由 python-chess 1.11.2 判定；游戏阶段沿用 EcoCompanion 的四人卧底规则。</p></section>';
  if (inGame) {
    const g = state.games[r];
    content = '<a class="breadcrumbs" href="#games">← 返回游戏间</a><h1>'+titles[r]+'</h1>'+gameStart(r);
    if (g) {
      content += r==="chess"?chessContent(g):wordContent(g);
      if (!g.result && g.next_actor!==0 && problem) content+='<button data-act="advance">重试当前回合</button>';
    } else content += '<section class="panel narrow"><h2>开始之前</h2><p>'+(r==="chess"?"支持王车易位、吃过路兵、升变和终局判断。选择执棋颜色，再开始对局。":"每人只看见自己的词。轮流发言，讨论后投票找出与多数人词语不同的玩家。")+'</p></section>';
  }
  main.innerHTML = content + (problem?'<p class="panel feedback" role="alert">'+escapeHTML(problem)+'</p>':'');
  const log = $("#log") || $(".wordlog"); if(log) log.scrollTop = log.scrollHeight;
}
async function run(work) {
  if (busy) return;
  busy = true; problem = ""; render();
  try { await work(); }
  catch (err) { problem = err.message; }
  finally { busy=false; render(); }
}
async function action(kind, verb, extra={}) {
  const g = state.games[kind];
  state.games[kind] = await api("action",{kind,id:g.id,revision:g.revision,action:verb,...extra});
  selected=""; promotions=[];
}
async function advanceOthers(kind) {
  let g=state.games[kind];
  while (g && route()===kind && !g.result && g.next_actor !== 0) {
    await action(kind,"advance"); render(); g=state.games[kind];
  }
}
main.addEventListener("input", e=>{
  if(e.target.id==="chatText") chatDraft=e.target.value;
  if(e.target.id==="wordText") wordDraft=e.target.value;
});
main.addEventListener("change", e=>{
  if(e.target.name==="chatMode"){ chatMode=e.target.value; problem=""; render(); }
});
main.addEventListener("submit",async e=>{
  e.preventDefault();
  const form=e.target, kind=route();
  if(form.id==="settingsForm") {
    const key=$("#key").value.trim(), consent=$("#consent").checked;
    await run(async()=>{state.settings=await api("settings",{key,consent});chatMode=altMode();message("连接设置已保存。");});
  } else if(form.id==="chatForm") {
    const text=chatDraft.trim(), mode=chatMode;
    if(!text) return;
    if(text!==retryText || !token){token=crypto.randomUUID();retryText=text;}
    await run(async()=>{state.chat=await api("chat",{text,mode,request_id:token});chatDraft="";token="";retryText="";});
  } else if(form.id==="startForm") {
    if(state.games[kind] && !state.games[kind].result && !confirm("重新开局将替换这一局的进度。继续吗？"))return;
    const data=Object.fromEntries(new FormData(form));
    await run(async()=>{state.games[kind]=await api("game",{kind,...data});selected="";promotions=[];wordDraft="";render();await advanceOthers(kind);});
  } else if(form.id==="speakForm") {
    const text=wordDraft.trim();
    if(text) await run(async()=>{await action(kind,"speak",{text});wordDraft="";render();await advanceOthers(kind);});
  }
});
main.addEventListener("click", async e=>{
  const btn=e.target.closest("button"); if(!btn || busy) return;
  const kind=route(), g=state.games[kind];
  if(btn.dataset.square && g && !g.result && g.next_actor===0) {
    const sq=btn.dataset.square, candidates=selected?g.legal.filter(m=>m.startsWith(selected+sq)):[];
    if(candidates.length>1){promotions=candidates;render();return;}
    if(candidates.length===1){await run(async()=>{await action(kind,"move",{text:candidates[0]});render();await advanceOthers(kind);});return;}
    selected=g.legal.some(m=>m.startsWith(sq))?sq:"";promotions=[];render();document.querySelector('[data-square="'+sq+'"]')?.focus({preventScroll:true});
  } else if(btn.dataset.promote) {
    const move=btn.dataset.promote;await run(async()=>{await action(kind,"move",{text:move});await advanceOthers(kind);});
  } else if(btn.dataset.vote!==undefined) {
    await run(async()=>{await action(kind,"vote",{target:Number(btn.dataset.vote)});render();await advanceOthers(kind);});
  } else if(btn.dataset.act) {
    const verb=btn.dataset.act;
    if(verb==="resign" && !confirm("确定认输并结束本局吗？")) return;
    await run(async()=>{if(verb!=="advance")await action(kind,verb);await advanceOthers(kind);});
  }
});
function resume() {
  if(!state) return;
  selected="";promotions=[];problem="";render();
  const kind=route(), g=state.games[kind];
  if(!busy && g && !g.result && g.next_actor!==0) run(()=>advanceOthers(kind));
}
window.addEventListener("hashchange",()=>{resume();window.scrollTo(0,0);});
api("state").then(data=>{state=data;chatMode=altMode();resume();}).catch(err=>{
  main.textContent=err.message+" 请从启动脚本显示的本机地址进入。";
});
