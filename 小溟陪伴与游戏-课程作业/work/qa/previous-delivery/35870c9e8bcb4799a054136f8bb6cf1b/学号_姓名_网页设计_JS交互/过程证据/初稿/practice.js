"use strict";

const files = "abcdefgh";
const initialPieces = { a8: "k", c7: "Q", c6: "K" };
const pieces = { k: { name: "黑王", symbol: "♚" }, Q: { name: "白后", symbol: "♕" }, K: { name: "白王", symbol: "♔" } };
const board = document.getElementById("board");
const turn = document.getElementById("turn");
const feedback = document.getElementById("feedback");
const notation = document.getElementById("notation");
const squares = new Map();
let position, selected, played;
let focusSquare = "c7";

function textSpan(text, className) {
  const span = document.createElement("span");
  span.className = className;
  span.textContent = text;
  span.setAttribute("aria-hidden", "true");
  return span;
}

for (let rank = 8; rank >= 1; rank--) {
  for (let file = 0; file < files.length; file++) {
    const name = files[file] + rank;
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "square " + ((file + rank) % 2 ? "dark" : "light");
    cell.dataset.square = name;
    cell.addEventListener("click", () => choose(name));
    cell.addEventListener("focus", () => setTabStop(name));
    squares.set(name, cell);
    board.appendChild(cell);
  }
}

function setTabStop(name) {
  squares.get(focusSquare).tabIndex = -1;
  squares.get(name).tabIndex = 0;
  focusSquare = name;
}

function render() {
  for (const [name, cell] of squares) {
    const piece = pieces[position[name]];
    const target = Boolean(selected && puzzleMoves[selected + name]);
    cell.replaceChildren();
    if (piece) cell.appendChild(textSpan(piece.symbol, "piece"));
    if (name[0] === "a") cell.appendChild(textSpan(name[1], "rank"));
    if (name[1] === "1") cell.appendChild(textSpan(name[0], "file"));
    cell.tabIndex = name === focusSquare ? 0 : -1;
    cell.classList.toggle("legal", target);
    cell.classList.toggle("selected", name === selected);
    cell.classList.toggle("recent", Boolean(played && played.slice(2) === name));
    cell.setAttribute("aria-label", [name, piece ? piece.name : "空格", target ? "可落子" : ""].filter(Boolean).join(" "));
    cell.setAttribute("aria-pressed", String(name === selected));
  }
}

function reset() {
  position = { ...initialPieces };
  selected = null;
  played = null;
  focusSquare = "c7";
  turn.textContent = "白方走棋";
  notation.textContent = "尚未落子";
  feedback.textContent = "选择一个白子，查看它可以到达的格子。";
  render();
}

const resultText = {
  mate: "将死！白后控制了黑王的逃生格，白王保护白后。",
  check: "黑王被将军，仍有解围手段。重开后试试封住全部逃生格。",
  stalemate: "逼和。黑王没有合法走法，但它没有被将军。可以重新尝试。",
  ongoing: "这一步合法，黑方还有应对。重开后试试更有威胁的落点。"
};

function choose(name) {
  if (played) return;
  if (name === selected) {
    selected = null;
    feedback.textContent = "已取消选择。";
  } else if (position[name] === "Q" || position[name] === "K") {
    selected = name;
    feedback.textContent = "已选中 " + name + " 的" + pieces[position[name]].name + "。请选择带圆点的目标格。";
  } else if (selected && puzzleMoves[selected + name]) {
    played = selected + name;
    const move = puzzleMoves[played];
    position[name] = position[selected];
    delete position[selected];
    selected = null;
    notation.textContent = "1. " + move.san;
    turn.textContent = move.state === "mate" ? "练习完成" : "本次尝试结束";
    feedback.textContent = resultText[move.state];
  } else {
    feedback.textContent = selected ? "该棋子不能走到 " + name + "，请另选带圆点的目标格。" : "先选择白后或白王。";
  }
  render();
}

board.addEventListener("keydown", event => {
  const offsets = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  if (!Object.hasOwn(offsets, event.key)) return;
  event.preventDefault();
  const name = event.target.dataset.square;
  if (!name) return;
  const [dx, dy] = offsets[event.key];
  const nextFile = Math.max(0, Math.min(7, files.indexOf(name[0]) + dx));
  const nextRank = Math.max(1, Math.min(8, Number(name[1]) + dy));
  squares.get(files[nextFile] + nextRank).focus();
});

document.getElementById("reset").addEventListener("click", reset);
reset();
