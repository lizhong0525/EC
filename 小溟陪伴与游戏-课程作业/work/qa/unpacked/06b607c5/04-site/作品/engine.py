"""Authoritative rules and player-specific information boundaries."""

from __future__ import annotations

import secrets
from collections import Counter

import chess
import chess.pgn

NAMES = ["你", "林舟", "阿柚", "知夏"]
STYLES = ["", "审慎，关注逻辑一致性", "活泼，善于生活化联想", "简洁，留意措辞差异"]
PAIRS = [
    ("咖啡", "奶茶"),
    ("地铁", "公交"),
    ("月亮", "太阳"),
    ("耳机", "音箱"),
    ("冰箱", "空调"),
    ("饺子", "馄饨"),
    ("钢琴", "吉他"),
    ("书店", "图书馆"),
    ("雨衣", "雨伞"),
    ("电影", "电视剧"),
    ("海豚", "鲸鱼"),
    ("足球", "篮球"),
    ("蛋糕", "面包"),
    ("火锅", "麻辣烫"),
    ("围巾", "手套"),
    ("瀑布", "喷泉"),
    ("日记", "博客"),
    ("沙发", "躺椅"),
    ("草莓", "樱桃"),
    ("相机", "手机"),
    ("风筝", "气球"),
    ("电梯", "扶梯"),
    ("牙膏", "漱口水"),
    ("露营", "野餐"),
]


class RuleError(ValueError):
    pass


def new_chess(color: str) -> dict:
    return {
        "kind": "chess",
        "human": color,
        "moves": [],
        "result": None,
        "reason": "",
        "comment": "",
        "revision": 0,
    }


def board_for(state: dict) -> chess.Board:
    board = chess.Board()
    for move in state["moves"]:
        board.push_uci(move)
    return board


def chess_turn(state: dict) -> str:
    return "white" if board_for(state).turn else "black"


def push_move(state: dict, uci: str) -> None:
    board = board_for(state)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as exc:
        raise RuleError("无效的走子格式") from exc
    if move not in board.legal_moves:
        raise RuleError("这一步不合法，请重新选择")
    board.push(move)
    state["moves"].append(uci)
    outcome = board.outcome()
    if outcome:
        state["result"] = outcome.result()
        state["reason"] = outcome.termination.name


def chess_view(state: dict) -> dict:
    board = board_for(state)
    replay = chess.Board()
    history = []
    for uci in state["moves"]:
        move = chess.Move.from_uci(uci)
        history.append({"uci": uci, "san": replay.san(move)})
        replay.push(move)
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "You" if state["human"] == "white" else "DeepSeek"
    game.headers["Black"] = "You" if state["human"] == "black" else "DeepSeek"
    game.headers["Result"] = state["result"] or "*"
    return {
        **state,
        "board": {chess.square_name(sq): piece.symbol() for sq, piece in board.piece_map().items()},
        "fen": board.fen(),
        "turn": chess_turn(state),
        "check": board.is_check(),
        "legal": [move.uci() for move in board.legal_moves] if not state["result"] else [],
        "can_claim": not state["result"] and board.can_claim_draw(),
        "history": history,
        "pgn": str(game),
        "next_actor": None
        if state["result"]
        else (0 if chess_turn(state) == state["human"] else 1),
    }


def new_undercover() -> dict:
    pair = list(secrets.choice(PAIRS))
    if secrets.randbelow(2):
        pair.reverse()
    undercover = secrets.randbelow(4)
    start = secrets.randbelow(4)
    return {
        "id": "",
        "kind": "undercover",
        "revision": 0,
        "round": 1,
        "phase": "describe",
        "words": [pair[1] if i == undercover else pair[0] for i in range(4)],
        "undercover": undercover,
        "alive": [0, 1, 2, 3],
        "order": [(start + i) % 4 for i in range(4)],
        "cursor": 0,
        "votes": {},
        "candidates": [0, 1, 2, 3],
        "runoff": False,
        "events": [],
        "result": None,
    }


def actor(state: dict) -> int | None:
    if state["result"] or state["cursor"] >= len(state["order"]):
        return None
    return state["order"][state["cursor"]]


def event(state: dict, kind: str, player: int | None, text: str, **extra) -> None:
    state["events"].append(
        {
            "round": state["round"],
            "kind": kind,
            "player": player,
            "text": text,
            **extra,
        }
    )


def valid_text(text: str, words: list[str]) -> bool:
    compact = "".join(c for c in text if c.isalnum()).casefold()
    return bool(text.strip()) and len(text) <= 160 and not any(w in compact for w in words)


def next_round(state: dict) -> None:
    if state["round"] >= 8:
        state["result"] = "draw"
        event(state, "system", None, "达到八轮上限，本局平局。")
        return
    state["round"] += 1
    state["phase"] = "describe"
    state["cursor"] = 0
    alive = state["alive"]
    offset = (state["round"] - 1) % len(alive)
    state["order"] = alive[offset:] + alive[:offset]
    state["votes"] = {}
    state["candidates"] = list(alive)
    state["runoff"] = False


def resolve_votes(state: dict) -> None:
    counts = Counter(state["votes"].values())
    leaders = [p for p, count in counts.items() if count == max(counts.values())]
    event(state, "ballot", None, "投票结果", votes=dict(state["votes"]))
    if len(leaders) > 1:
        if state["runoff"]:
            event(state, "system", None, "再次平票，无人出局，进入下一轮。")
            next_round(state)
        else:
            state.update(
                phase="defend", order=leaders, cursor=0, candidates=leaders, votes={}, runoff=True
            )
            event(
                state,
                "system",
                None,
                "平票：" + "、".join(NAMES[p] for p in leaders) + "加赛陈述。",
            )
        return
    eliminated = leaders[0]
    state["alive"].remove(eliminated)
    role = "卧底" if eliminated == state["undercover"] else "平民"
    event(state, "eliminated", eliminated, f"{NAMES[eliminated]}出局，身份为{role}。")
    if eliminated == state["undercover"]:
        state["result"] = "civilians"
    elif len(state["alive"]) == 2:
        state["result"] = "undercover"
    else:
        next_round(state)


def speak(state: dict, player: int, text: str) -> None:
    if state["phase"] not in {"describe", "discuss", "defend"} or actor(state) != player:
        raise RuleError("现在不是该玩家的发言回合")
    if not valid_text(text, [state["words"][player]]):
        raise RuleError("发言应为 1 至 160 字，且不能直接包含自己的词语")
    event(state, state["phase"], player, text.strip())
    state["cursor"] += 1
    if state["cursor"] == len(state["order"]):
        state["cursor"] = 0
        if state["phase"] == "describe":
            state["phase"] = "discuss"
        else:
            state["phase"] = "vote"
            # Human votes first; no ballots are revealed to subsequent voters.
            state["order"] = list(state["alive"])


def vote(state: dict, player: int, target: int) -> None:
    if state["phase"] != "vote" or actor(state) != player:
        raise RuleError("现在不是该玩家的投票回合")
    if target not in state["candidates"] or target == player:
        raise RuleError("只能投给候选名单中除自己外的存活玩家")
    state["votes"][str(player)] = target
    state["cursor"] += 1
    if state["cursor"] == len(state["order"]):
        resolve_votes(state)


def undercover_view(state: dict) -> dict:
    visible = {
        key: state[key]
        for key in (
            "id",
            "kind",
            "revision",
            "round",
            "phase",
            "alive",
            "events",
            "result",
            "candidates",
        )
    }
    visible.update(
        players=[{"id": i, "name": name} for i, name in enumerate(NAMES)],
        word=state["words"][0],
        next_actor=actor(state),
        votes_cast=len(state["votes"]),
        human_alive=0 in state["alive"],
    )
    if state["result"]:
        visible["reveal"] = [
            {
                "name": name,
                "word": state["words"][i],
                "role": "卧底" if i == state["undercover"] else "平民",
            }
            for i, name in enumerate(NAMES)
        ]
    return visible


def player_context(state: dict, player: int) -> dict:
    """Explicit allowlist: never serialize the omniscient state into a prompt."""
    return {
        "player_id": player,
        "name": NAMES[player],
        "style": STYLES[player],
        "private_word": state["words"][player],
        "round": state["round"],
        "phase": state["phase"],
        "alive": state["alive"],
        "players": [{"id": i, "name": name} for i, name in enumerate(NAMES)],
        "public_events": state["events"],
        "eligible_targets": [p for p in state["candidates"] if p != player],
    }


def view(state: dict) -> dict:
    return chess_view(state) if state["kind"] == "chess" else undercover_view(state)
