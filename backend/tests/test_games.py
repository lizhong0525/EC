from __future__ import annotations

import asyncio
import json

import chess
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from ecocompanion.games import engine
from ecocompanion.games.service import Games
from ecocompanion.pet.api import create_pet_app


def test_chess_illegal_move_and_fools_mate():
    state = engine.new_chess("white")
    with pytest.raises(engine.RuleError):
        engine.push_move(state, "e2e5")
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        engine.push_move(state, move)
    assert state["result"] == "0-1"
    assert state["reason"] == "CHECKMATE"
    assert engine.chess_view(state)["legal"] == []


def test_chess_draw_and_castling_are_library_rules():
    for moves in (
        ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
        ["g1f3", "g8f6", "f3g1", "f6g8"] * 2,
    ):
        state = engine.new_chess("white")
        for move in moves:
            engine.push_move(state, move)
        if "e2e4" in moves:
            engine.push_move(state, "e1g1")
            assert engine.board_for(state).piece_at(chess.G1).symbol() == "K"
        else:
            assert engine.board_for(state).can_claim_threefold_repetition()


def test_only_players_private_word_is_in_prompt_context():
    state = engine.new_undercover()
    state["words"] = ["咖啡", "奶茶", "咖啡", "咖啡"]
    context = engine.player_context(state, 1)
    assert context["private_word"] == "奶茶"
    assert "undercover" not in context
    assert "words" not in context
    view = engine.undercover_view(state)
    assert view["word"] == "咖啡"
    assert "奶茶" not in json.dumps(view)


def test_human_speech_cannot_leak_word():
    state = engine.new_undercover()
    state["order"], state["cursor"] = [0, 1, 2, 3], 0
    with pytest.raises(engine.RuleError):
        engine.speak(state, 0, f"我是{state['words'][0]}")


def test_undercover_voting_hides_ballots_until_complete():
    state = engine.new_undercover()
    state.update(phase="vote", undercover=2, order=[0, 1, 2, 3], cursor=0)
    for p in range(4):
        engine.vote(state, p, 2 if p != 2 else 0)
        if p < 3:
            assert engine.undercover_view(state)["votes_cast"] == p + 1
    assert state["result"] == "civilians"
    assert state["events"][-2]["votes"] == {"0": 2, "1": 2, "2": 0, "3": 2}


def test_tie_requires_defense_and_runoff():
    state = engine.new_undercover()
    state.update(phase="vote", undercover=3, order=[0, 1, 2, 3], cursor=0)
    for p, target in [(0, 1), (1, 2), (2, 1), (3, 2)]:
        engine.vote(state, p, target)
    assert state["phase"] == "defend" and state["candidates"] == [1, 2]
    for p in [1, 2]:
        engine.speak(state, p, "我会结合前面发言解释。")
    assert state["phase"] == "vote"
    for p, target in [(0, 1), (1, 2), (2, 1), (3, 2)]:
        engine.vote(state, p, target)
    assert state["phase"] == "describe" and state["round"] == 2


def test_failed_ai_call_does_not_mutate_or_commit(tmp_path):
    games = Games(tmp_path, tmp_path)
    game = games.create("chess", "black")

    async def fail(*args, **kwargs):
        raise HTTPException(502, "provider unavailable")

    games.ask = fail
    with pytest.raises(HTTPException):
        asyncio.run(games.action(game["id"], 0, "advance"))
    assert games.load(game["id"])["revision"] == 0
    assert games.load(game["id"])["moves"] == []


def test_stale_revision_and_persistence(tmp_path):
    games = Games(tmp_path, tmp_path)
    game = games.create("chess")
    asyncio.run(games.action(game["id"], 0, "move", "e2e4"))
    with pytest.raises(HTTPException):
        asyncio.run(games.action(game["id"], 0, "move", "d2d4"))
    assert games.load(game["id"])["moves"] == ["e2e4"]


def test_web_routes_and_secret_isolation(tmp_path):
    with TestClient(create_pet_app(tmp_path)) as client:
        assert client.get("/games").status_code == 200
        token = client.get("/api/pet/bootstrap").json()["token"]
        game = client.post(
            "/api/pet/games", json={"kind": "undercover"}, headers={"X-Pet-Token": token}
        ).json()
        state = client.app.state.games.load(game["id"])
        assert "words" not in game
        assert game["word"] == state["words"][0]
        assert all(
            other not in str(game)
            for i, other in enumerate(set(state["words"]))
            if other != game["word"]
        )
        assert client.post("/api/pet/games", json={"kind": "chess"}).status_code == 403


def test_deterministic_undercover_turns_with_mock_agents(tmp_path):
    games = Games(tmp_path, tmp_path)
    game = games.create("undercover")
    state = games.load(game["id"])
    state.update(undercover=3, words=["咖啡", "咖啡", "咖啡", "奶茶"], order=[1, 2, 3, 0])
    with games.connect() as db:
        db.execute("UPDATE games SET state = ? WHERE id = ?", (json.dumps(state), game["id"]))

    async def agent(prompt, context, validate):
        if context["phase"] == "vote":
            return {"target": context["eligible_targets"][0]}
        return {"text": "适合和朋友一起分享。"}

    games.ask = agent

    def advance_agents(current):
        while current["next_actor"] not in {None, 0} and not current["result"]:
            current = asyncio.run(games.action(current["id"], current["revision"], "advance"))
        return current

    for _ in range(3):
        game = asyncio.run(games.action(game["id"], game["revision"], "advance"))
    assert game["next_actor"] == 0
    game = asyncio.run(games.action(game["id"], game["revision"], "speak", text="日常会用到。"))
    game = advance_agents(game)
    assert game["phase"] == "discuss"
    game = asyncio.run(
        games.action(game["id"], game["revision"], "speak", text="我会注意谁描述得太含糊。")
    )
    game = advance_agents(game)
    assert game["phase"] == "vote" and game["next_actor"] == 0
    game = asyncio.run(games.action(game["id"], game["revision"], "vote", target=3))
    game = advance_agents(game)
    ballot = next(event for event in game["events"] if event["kind"] == "ballot")
    eliminated = max(
        ballot["votes"].values(), key=lambda p: list(ballot["votes"].values()).count(p)
    )
    assert game["result"] == ("civilians" if eliminated == 3 else "undercover")
    assert len(game["reveal"]) == 4
