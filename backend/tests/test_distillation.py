"""Network-isolated teacher-data pipeline tests; approvals are TEST_FIXTURE only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from scripts import distillation as d

SAFE_REPLY = "晚上好，我们可以聊聊你刚结束的课程。你想从哪件小事说起？"


@pytest.fixture
def packet(tmp_path: Path) -> Path:
    sources = tmp_path / "sources"
    sources.mkdir()
    records = {}
    for name, url in d.SOURCES.items():
        text = (
            "TEST_FIXTURE 训练其他模型（如模型蒸馏）"
            if name == "terms"
            else "TEST_FIXTURE deepseek-flash"
        )
        raw = ("<p>" + text + "</p>").encode()
        (sources / f"{name}.html").write_bytes(raw)
        (sources / f"{name}.txt").write_text(text, encoding="utf-8")
        records[name] = {
            "url": url,
            "sha256": d.digest(raw),
            "text_sha256": d.digest(text.encode()),
        }
    d.write_json(sources / "manifest.json", {"records": records})
    output = tmp_path / "packet"
    d.prepare(output, sources, d.ROOT / "docs/DISTILLATION_PILOT_SEEDS_V1.json")
    return output


def ok_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["max_tokens"] == 384
    assert str(request.url) == d.ENDPOINT
    assert payload["model"] == "deepseek-flash"
    assert "id" not in payload
    return httpx.Response(
        200,
        json={
            "model": "TEST_FIXTURE_teacher",
            "choices": [{"message": {"content": SAFE_REPLY}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        },
    )


def make_run(packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "TEST_FIXTURE_NOT_A_REAL_KEY")
    output = tmp_path / "teacher"
    result = d.generate(
        packet, output, limit=1, allow_paid=True, transport=httpx.MockTransport(ok_response)
    )
    assert result["attempted"] == 1
    assert result["statuses"] == {"PENDING_HUMAN_REVIEW": 1}
    assert result["training_approved"] == 0
    return output


def approve(run: Path, reference: str = "TEST_FIXTURE_REVIEW") -> Path:
    with (run / "review.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields is not None
    rows[0].update(
        {
            "decision": "APPROVE",
            "reviewer_id": "TEST_FIXTURE_NOT_A_REAL_REVIEWER",
            "review_reference": reference,
        }
    )
    output = run / "test-fixture-review.csv"
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def test_source_snapshots_have_portable_multiline_hashes(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        content = "<p>训练其他模型（如模型蒸馏）</p><p>deepseek-flash</p>"
        return httpx.Response(200, text=content)

    output = tmp_path / "sources"
    manifest = d.freeze_sources(output, transport=httpx.MockTransport(handler))
    assert d.verify_sources(output) == manifest
    for name in d.SOURCES:
        raw = (output / f"{name}.txt").read_bytes()
        assert b"\n" in raw
        assert b"\r" not in raw
        assert d.digest(raw) == manifest["records"][name]["text_sha256"]


def test_failed_source_capture_cannot_prepare_packet(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://example.invalid/"})

    output = tmp_path / "sources"
    with pytest.raises(httpx.HTTPStatusError):
        d.freeze_sources(output, transport=httpx.MockTransport(handler))
    assert not (output / "manifest.json").exists()


def test_prepare_freezes_grouped_splits_and_visible_only_requests(packet: Path) -> None:
    plan, items = d.load_packet(packet)
    assert plan["counts"] == {"train": 8, "dev": 8, "heldout": 8}
    assert plan["expert_gold"] is False
    assert plan["independent_blind_test"] is False
    assert len({item["group"] for item in items}) == 24
    for item in items:
        text = json.dumps(item["messages"], ensure_ascii=False)
        assert "category" not in text
        assert "heldout" not in text
        assert item["group"] not in text
        assert not d.UNSUPPORTED.search(item["text"])
    with pytest.raises(FileExistsError):
        d.prepare(packet, Path(plan["sources"]), d.ROOT / "docs/DISTILLATION_PILOT_SEEDS_V1.json")


def test_no_key_or_no_paid_authorization_sends_nothing(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValueError, match="allow-paid"):
        d.generate(packet, tmp_path / "run")
    with pytest.raises(ValueError, match="no request sent"):
        d.generate(packet, tmp_path / "run", allow_paid=True)
    assert not (tmp_path / "run").exists()


def test_requests_only_train_partition_and_capped_call_count(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "TEST_FIXTURE_NOT_A_REAL_KEY")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return ok_response(request)

    output = tmp_path / "run"
    report = d.generate(packet, output, allow_paid=True, transport=httpx.MockTransport(handler))
    assert len(seen) == report["attempted"] == 8
    inputs = d.jsonl(packet / "inputs.jsonl")
    assert seen == [
        {
            "model": "deepseek-flash",
            "messages": item["messages"],
            "temperature": 0.5,
            "max_tokens": 384,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        for item in inputs
        if item["split"] == "train"
    ]
    assert "TEST_FIXTURE_NOT_A_REAL_KEY" not in "".join(
        path.read_text(encoding="utf-8-sig") for path in output.iterdir()
    )
    with pytest.raises(ValueError, match="at most 8"):
        d.generate(packet, tmp_path / "too-many", limit=9, allow_paid=True)


def test_provider_metadata_cannot_persist_credentials(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "TEST_FIXTURE_NOT_A_REAL_KEY"
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)

    def handler(request: httpx.Request) -> httpx.Response:
        body = ok_response(request).json()
        body["model"] = key
        body["system_fingerprint"] = {"untrusted": key}
        body["choices"][0]["finish_reason"] = key
        return httpx.Response(200, json=body)

    output = tmp_path / "run"
    d.generate(packet, output, limit=1, allow_paid=True, transport=httpx.MockTransport(handler))
    assert key not in "".join(path.read_text(encoding="utf-8-sig") for path in output.iterdir())


@pytest.mark.parametrize("kind", ["http", "timeout", "usage", "secret", "truncated"])
def test_failure_is_preserved_without_retry(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "TEST_FIXTURE_NOT_A_REAL_KEY")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if kind == "http":
            return httpx.Response(429, text="TEST_FIXTURE_NOT_A_REAL_KEY")
        if kind == "timeout":
            raise httpx.ReadTimeout("TEST_FIXTURE_NOT_A_REAL_KEY")
        body = ok_response(request).json()
        if kind == "usage":
            del body["usage"]
        elif kind == "secret":
            body["choices"][0]["message"]["content"] = "TEST_FIXTURE_NOT_A_REAL_KEY"
        else:
            body["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=body)

    output = tmp_path / "run"
    result = d.generate(
        packet, output, limit=1, allow_paid=True, transport=httpx.MockTransport(handler)
    )
    assert result["attempted"] == calls == 1
    assert result["training_approved"] == 0
    text = "".join(path.read_text(encoding="utf-8-sig") for path in output.iterdir())
    assert "TEST_FIXTURE_NOT_A_REAL_KEY" not in text
    assert "PENDING_HUMAN_REVIEW" not in result["statuses"]


def test_pending_reviews_cannot_become_training_data(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(packet, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="no human-approved"):
        d.export_reviewed(packet, run, run / "review.csv", tmp_path / "training")
    assert not (tmp_path / "training").exists()


def test_traceable_fixture_approval_exports_only_messages(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(packet, tmp_path, monkeypatch)
    reviews = approve(run)
    output = tmp_path / "training"
    summary = d.export_reviewed(packet, run, reviews, output)
    records = d.jsonl(output / "train.jsonl")
    assert summary["samples"] == len(records) == 1
    assert set(records[0]) == {"messages"}
    assert records[0]["messages"][-1] == {"role": "assistant", "content": SAFE_REPLY}
    assert summary["weights_trained"] is False


def test_tampered_candidate_or_missing_review_reference_is_rejected(
    packet: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = make_run(packet, tmp_path, monkeypatch)
    reviews = approve(run, reference="")
    with pytest.raises(ValueError, match="traceable human approval"):
        d.export_reviewed(packet, run, reviews, tmp_path / "training")
    response_path = run / "g01.response.json"
    response_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher record changed"):
        d.export_reviewed(packet, run, reviews, tmp_path / "training")


def test_tampered_inputs_sources_or_budget_fail_before_call(packet: Path, tmp_path: Path) -> None:
    plan = d.read_json(packet / "plan.json")
    plan["max_tokens"] = 100000
    (packet / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="pilot limits changed"):
        d.load_packet(packet)
    (packet / "inputs.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="packet inputs changed"):
        d.load_packet(packet)


def test_new_crisis_seed_is_not_in_first_remote_pilot(packet: Path, tmp_path: Path) -> None:
    plan = d.read_json(packet / "plan.json")
    seeds = tmp_path / "crisis.json"
    d.write_json(
        seeds,
        [
            {
                "id": "crisis",
                "group": "crisis",
                "split": "train",
                "text": "我想死了",
            }
        ],
    )
    with pytest.raises(ValueError, match="excludes crisis"):
        d.prepare(tmp_path / "crisis-packet", Path(plan["sources"]), seeds)
    assert not (tmp_path / "crisis-packet").exists()
