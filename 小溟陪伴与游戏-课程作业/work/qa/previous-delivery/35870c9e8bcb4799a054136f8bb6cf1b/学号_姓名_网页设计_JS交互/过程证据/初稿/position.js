// 局面经 python-chess 1.11.2 校验。逐一尝试起始局面的每种合法走法，记录 SAN 与终局状态。
// FEN: k7/2Q5/2K5/8/8/8/8/8 w - - 0 1
const puzzleMoves = Object.freeze({
  "c7d8": { san: "Qd8+", state: "check" },
  "c7c8": { san: "Qc8+", state: "check" },
  "c7b8": { san: "Qb8+", state: "check" },
  "c7h7": { san: "Qh7", state: "ongoing" },
  "c7g7": { san: "Qg7", state: "ongoing" },
  "c7f7": { san: "Qf7", state: "ongoing" },
  "c7e7": { san: "Qe7", state: "ongoing" },
  "c7d7": { san: "Qd7", state: "ongoing" },
  "c7b7": { san: "Qb7#", state: "mate" },
  "c7a7": { san: "Qa7+", state: "check" },
  "c7d6": { san: "Qd6", state: "ongoing" },
  "c7b6": { san: "Qb6", state: "stalemate" },
  "c7e5": { san: "Qe5", state: "ongoing" },
  "c7a5": { san: "Qa5+", state: "check" },
  "c7f4": { san: "Qf4", state: "ongoing" },
  "c7g3": { san: "Qg3", state: "ongoing" },
  "c7h2": { san: "Qh2", state: "ongoing" },
  "c6d7": { san: "Kd7", state: "stalemate" },
  "c6d6": { san: "Kd6", state: "stalemate" },
  "c6b6": { san: "Kb6", state: "stalemate" },
  "c6d5": { san: "Kd5", state: "stalemate" },
  "c6c5": { san: "Kc5", state: "stalemate" },
  "c6b5": { san: "Kb5", state: "stalemate" }
});
