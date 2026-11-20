"""Local, role-isolated, append-only expert annotation and adjudication."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score  # type: ignore[import-untyped]

from ecocompanion.domain.enums import Action, RiskLevel
from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research import VERSION
from ecocompanion.research.contracts import Judgment, ResearchCase


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def dataset_digest(cases: tuple[ResearchCase, ...]) -> str:
    return digest([case.model_dump(mode="json") for case in cases])


def _material(judgment: Judgment) -> dict[str, Any]:
    result = judgment.model_dump(mode="json", exclude={"confidence", "reason"})
    for name in ("acceptable_actions", "prohibited_actions", "missing_information"):
        result[name] = sorted(result[name])
    result["evidence"] = sorted(result["evidence"], key=canonical)
    return result


def _kappa(left: list[Any], right: list[Any], *, ordered: bool = False) -> float | None:
    import math

    if not left or len(set(left + right)) < 2:
        return None
    value = float(
        cohen_kappa_score(
            left,
            right,
            weights="quadratic" if ordered else None,
            labels=[0, 1, 2, 3, 4] if ordered else None,
        )
    )
    return value if math.isfinite(value) else None


def needs_adjudication(case_id: str, step_id: int, a: Judgment, b: Judgment) -> bool:
    return (
        a.risk_level in {RiskLevel.L3, RiskLevel.L4}
        or b.risk_level in {RiskLevel.L3, RiskLevel.L4}
        or _material(a) != _material(b)
        or int(digest([case_id, step_id])[:8], 16) % 10 == 0
    )


class AnnotationStore:
    """A local research boundary, not protection against a malicious OS administrator."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @classmethod
    def create(
        cls, path: Path, cases: tuple[ResearchCase, ...], *, manual_digest: str
    ) -> tuple[AnnotationStore, dict[str, str]]:
        if path.exists():
            raise FileExistsError("annotation store already exists")
        if not cases or len({c.case_id for c in cases}) != len(cases):
            raise ValueError("empty or duplicate case set")
        observations = [o.current_observation.observation_id for c in cases for o in c.observations]
        if len(set(observations)) != len(observations):
            raise ValueError("observation IDs must be globally unique")
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
        try:
            connection.executescript("""
                CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE reviewers(
                    reviewer TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL,
                    role TEXT NOT NULL);
                CREATE TABLE attestations(
                    reviewer TEXT PRIMARY KEY, reference TEXT NOT NULL,
                    administrator TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    person_id TEXT UNIQUE NOT NULL, qualification TEXT NOT NULL);
                CREATE TABLE judgments(
                    reviewer TEXT NOT NULL, case_id TEXT NOT NULL, step_id INTEGER NOT NULL,
                    payload TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    PRIMARY KEY(reviewer,case_id,step_id));
                CREATE TABLE events(
                    seq INTEGER PRIMARY KEY, payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
                CREATE TRIGGER no_edit_judgment BEFORE UPDATE ON judgments
                    BEGIN SELECT RAISE(ABORT,'immutable judgment'); END;
                CREATE TRIGGER no_delete_judgment BEFORE DELETE ON judgments
                    BEGIN SELECT RAISE(ABORT,'immutable judgment'); END;
                CREATE TRIGGER no_edit_event BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT,'immutable event'); END;
                CREATE TRIGGER no_delete_event BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT,'immutable event'); END;
                CREATE TRIGGER no_edit_attestation BEFORE UPDATE ON attestations
                    BEGIN SELECT RAISE(ABORT,'immutable attestation'); END;
                CREATE TRIGGER no_delete_attestation BEFORE DELETE ON attestations
                    BEGIN SELECT RAISE(ABORT,'immutable attestation'); END;
                CREATE TRIGGER no_edit_meta BEFORE UPDATE ON meta
                    BEGIN SELECT RAISE(ABORT,'immutable dataset'); END;
                CREATE TRIGGER no_delete_meta BEFORE DELETE ON meta
                    BEGIN SELECT RAISE(ABORT,'immutable dataset'); END;
                CREATE TRIGGER no_edit_reviewer BEFORE UPDATE ON reviewers
                    BEGIN SELECT RAISE(ABORT,'immutable role'); END;
                CREATE TRIGGER no_delete_reviewer BEFORE DELETE ON reviewers
                    BEGIN SELECT RAISE(ABORT,'immutable role'); END;
            """)
            values = {
                "dataset": [c.model_dump(mode="json") for c in cases],
                "dataset_digest": dataset_digest(cases),
                "manual_digest": manual_digest,
                "version": VERSION,
            }
            connection.executemany(
                "INSERT INTO meta VALUES (?,?)", [(k, canonical(v)) for k, v in values.items()]
            )
            tokens = {
                role: secrets.token_urlsafe(32) for role in ("expert-a", "expert-b", "adjudicator")
            }
            connection.executemany(
                "INSERT INTO reviewers VALUES (?,?,?)",
                [
                    (role, hashlib.sha256(token.encode()).hexdigest(), role)
                    for role, token in tokens.items()
                ],
            )
            connection.commit()
        finally:
            connection.close()
        store = cls(path)
        with store.connection() as connection:
            store._event(
                connection,
                {
                    "kind": "created",
                    **{k: v for k, v in values.items() if k != "dataset"},
                    "reviewer_hashes": {
                        role: hashlib.sha256(token.encode()).hexdigest()
                        for role, token in tokens.items()
                    },
                },
            )
        return store, tokens

    @staticmethod
    def _meta(connection: sqlite3.Connection, key: str) -> Any:
        row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            raise ValueError("missing store metadata")
        return json.loads(row["value"])

    def cases(self) -> tuple[ResearchCase, ...]:
        with self.connection() as connection:
            return self._cases(connection)

    def _cases(self, connection: sqlite3.Connection) -> tuple[ResearchCase, ...]:
        return tuple(ResearchCase.model_validate(c) for c in self._meta(connection, "dataset"))

    @staticmethod
    def _event(connection: sqlite3.Connection, payload: dict[str, Any]) -> None:
        last = connection.execute(
            "SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        previous = last["event_hash"] if last else "0" * 64
        body = {**payload, "recorded_at": datetime.now(UTC).isoformat()}
        event_hash = digest([previous, body])
        connection.execute(
            "INSERT INTO events(payload,previous_hash,event_hash) VALUES (?,?,?)",
            (canonical(body), previous, event_hash),
        )

    def _authenticate(self, connection: sqlite3.Connection, token: str) -> str:
        row = connection.execute(
            "SELECT reviewer FROM reviewers WHERE token_hash=?",
            (hashlib.sha256(token.encode()).hexdigest(),),
        ).fetchone()
        if row is None:
            raise PermissionError("invalid annotation access token")
        return str(row["reviewer"])

    def attest(
        self,
        reviewer: str,
        *,
        reference: str,
        administrator: str,
        person_id: str,
        qualification: str,
    ) -> None:
        if not all(s.strip() for s in (reference, administrator, person_id, qualification)):
            raise ValueError("human identity/qualification verification reference required")
        with self.connection() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM reviewers WHERE reviewer=?", (reviewer,)
                ).fetchone()
                is None
            ):
                raise ValueError("unknown reviewer")
            connection.execute(
                "INSERT INTO attestations VALUES (?,?,?,?,?,?)",
                (
                    reviewer,
                    reference,
                    administrator,
                    datetime.now(UTC).isoformat(),
                    person_id,
                    qualification,
                ),
            )
            self._event(
                connection,
                {
                    "kind": "attestation",
                    "reviewer": reviewer,
                    "reference": reference,
                    "administrator": administrator,
                    "person_id": person_id,
                    "qualification": qualification,
                },
            )

    @staticmethod
    def _labels(connection: sqlite3.Connection) -> dict[tuple[str, str, int], Judgment]:
        return {
            (
                str(r["reviewer"]),
                str(r["case_id"]),
                int(r["step_id"]),
            ): Judgment.model_validate_json(r["payload"])
            for r in connection.execute("SELECT * FROM judgments")
        }

    def next_task(self, token: str) -> dict[str, Any]:
        with self.connection() as connection:
            reviewer = self._authenticate(connection, token)
            return self._task(connection, reviewer)

    def _task(self, connection: sqlite3.Connection, reviewer: str) -> dict[str, Any]:
        cases, labels = self._cases(connection), self._labels(connection)
        total = sum(len(c.observations) for c in cases)
        independent_complete = all(
            (role, c.case_id, o.step_id) in labels
            for role in ("expert-a", "expert-b")
            for c in cases
            for o in c.observations
        )
        if reviewer == "adjudicator" and not independent_complete:
            return {"status": "waiting_for_independent_labels"}
        for case in cases:
            for observation in case.observations:
                key = (reviewer, case.case_id, observation.step_id)
                if key in labels:
                    continue
                if reviewer == "adjudicator":
                    a = labels[("expert-a", case.case_id, observation.step_id)]
                    b = labels[("expert-b", case.case_id, observation.step_id)]
                    if not needs_adjudication(case.case_id, observation.step_id, a, b):
                        continue
                task = {
                    "status": "ready",
                    "task_id": digest([reviewer, case.case_id, observation.step_id]),
                    "case_id": case.case_id,
                    "step_id": observation.step_id,
                    "observation": observation.model_dump(mode="json"),
                    "completed_nodes": sum(k[0] == reviewer for k in labels),
                    "assigned_nodes": total,
                }
                if reviewer == "adjudicator":
                    task["independent_labels"] = {
                        "a": a.model_dump(mode="json"),
                        "b": b.model_dump(mode="json"),
                    }
                return task
        return {"status": "complete"}

    def submit(self, token: str, task_id: str, judgment: Judgment) -> dict[str, Any]:
        with self.connection() as connection:
            reviewer = self._authenticate(connection, token)
            task = self._task(connection, reviewer)
            if task.get("task_id") != task_id:
                raise ValueError("stale, future or already locked annotation task")
            observation = ObservationV1.model_validate(task["observation"])
            allowed_ids = {m.message_id for m in observation.visible_context.conversation_history}
            allowed_ids.add(observation.current_observation.observation_id)
            if any(e.observation_id not in allowed_ids for e in judgment.evidence):
                raise ValueError("evidence references future or unknown observations")
            if not set(judgment.acceptable_actions + judgment.prohibited_actions) <= set(
                observation.allowed_primary_actions
            ):
                raise ValueError("annotation contains unknown actions")
            body = judgment.model_dump(mode="json")
            connection.execute(
                "INSERT INTO judgments VALUES (?,?,?,?,?)",
                (
                    reviewer,
                    task["case_id"],
                    task["step_id"],
                    canonical(body),
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._event(
                connection,
                {
                    "kind": "judgment",
                    "reviewer": reviewer,
                    "case_id": task["case_id"],
                    "step_id": task["step_id"],
                    "judgment": body,
                    "task_id": task_id,
                },
            )
            return {"status": "locked"}

    def audit(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._audit(connection)

    def _audit(self, connection: sqlite3.Connection) -> dict[str, Any]:
        previous = "0" * 64
        seen: dict[tuple[str, str, int], dict[str, Any]] = {}
        attested: dict[str, tuple[str, ...]] = {}
        rows = list(connection.execute("SELECT * FROM events ORDER BY seq"))
        for sequence, row in enumerate(rows, 1):
            body = json.loads(row["payload"])
            if (
                row["seq"] != sequence
                or row["previous_hash"] != previous
                or digest([previous, body]) != row["event_hash"]
            ):
                raise ValueError("annotation audit chain mismatch")
            previous = row["event_hash"]
            if body["kind"] == "judgment":
                seen[(body["reviewer"], body["case_id"], body["step_id"])] = body["judgment"]
            elif body["kind"] == "attestation":
                attested[body["reviewer"]] = (
                    body["reference"],
                    body["administrator"],
                    body["person_id"],
                    body["qualification"],
                )
        labels = self._labels(connection)
        if {k: v.model_dump(mode="json") for k, v in labels.items()} != seen:
            raise ValueError("judgment table differs from append-only audit")
        actual_attestations = {
            r["reviewer"]: (r["reference"], r["administrator"], r["person_id"], r["qualification"])
            for r in connection.execute("SELECT * FROM attestations")
        }
        if actual_attestations != attested:
            raise ValueError("attestation table differs from audit")
        if not rows or json.loads(rows[0]["payload"])["dataset_digest"] != dataset_digest(
            self._cases(connection)
        ):
            raise ValueError("dataset differs from audit root")
        root = json.loads(rows[0]["payload"])
        if any(
            root[k] != self._meta(connection, k)
            for k in ("manual_digest", "version", "dataset_digest")
        ):
            raise ValueError("metadata differs from audit root")
        reviewers = list(connection.execute("SELECT * FROM reviewers"))
        if {r["reviewer"]: r["token_hash"] for r in reviewers} != root["reviewer_hashes"] or any(
            r["role"] != r["reviewer"] for r in reviewers
        ):
            raise ValueError("reviewer roles differ from audit root")
        return {
            "status": "verified",
            "events": len(rows),
            "head": previous,
            "warning": "Hash chaining detects accidental edits; OS administrators remain trusted.",
        }

    def agreement(self) -> dict[str, Any]:
        with self.connection() as connection:
            return self._agreement(connection)

    def label_status(self) -> dict[str, Any]:
        with self.connection() as connection:
            self._audit(connection)
            labels = self._labels(connection)
            frozen = any(
                json.loads(r["payload"])["kind"] == "gold_export"
                for r in connection.execute("SELECT payload FROM events")
            )
            records = []
            for case in self._cases(connection):
                for observation in case.observations:
                    pair = [
                        labels.get((role, case.case_id, observation.step_id))
                        for role in ("expert-a", "expert-b")
                    ]
                    a, b = pair
                    unresolved = any(
                        j is not None and j.resolution == "unresolved" for j in pair
                    ) or (
                        a is not None
                        and b is not None
                        and needs_adjudication(case.case_id, observation.step_id, a, b)
                        and ("adjudicator", case.case_id, observation.step_id) not in labels
                    )
                    records.append(
                        {
                            "case_id": case.case_id,
                            "step_id": observation.step_id,
                            "status": "GOLD"
                            if frozen
                            else "UNRESOLVED"
                            if unresolved
                            else "SILVER"
                            if a or b
                            else "UNLABELED",
                            "source": "human_frozen"
                            if frozen
                            else "human_unfrozen"
                            if a or b
                            else "unannotated_input",
                            "gold_promotion": "ONLY_BY_FREEZE_GATE",
                        }
                    )
            return {"records": records, "gold_status": "FROZEN" if frozen else "NOT_FROZEN"}

    def record_retrospective(
        self,
        *,
        case_id: str,
        windows: list[dict[str, Any]],
        reason: str,
        reviewer_reference: str,
    ) -> None:
        if not reason.strip() or not reviewer_reference.strip():
            raise ValueError("retrospective review requires a reason and human reference")
        with self.connection() as connection:
            self._audit(connection)
            cases = {c.case_id: c for c in self._cases(connection)}
            if case_id not in cases:
                raise ValueError("unknown retrospective case")
            for window in windows:
                start, end = window["earliest_step"], window["latest_step"]
                if not (0 <= start <= end < len(cases[case_id].observations)):
                    raise ValueError("retrospective interval outside the case")
                Action(window["action"])
            self._event(
                connection,
                {
                    "kind": "retrospective_window",
                    "case_id": case_id,
                    "windows": windows,
                    "reason": reason,
                    "reviewer_reference": reviewer_reference,
                    "used_for_prospective_scoring": False,
                },
            )

    def _agreement(self, connection: sqlite3.Connection) -> dict[str, Any]:
        labels, cases = self._labels(connection), self._cases(connection)
        expected = {(c.case_id, o.step_id) for c in cases for o in c.observations}
        for reviewer in ("expert-a", "expert-b"):
            own = {(c, s): j for (r, c, s), j in labels.items() if r == reviewer}
            if set(own) != expected:
                return {
                    "status": "INCOMPLETE",
                    "paired_metrics": None,
                    "reason": "all assigned nodes must be independently annotated",
                }
        pairs = [
            (labels[("expert-a", c, s)], labels[("expert-b", c, s)]) for c, s in sorted(expected)
        ]
        ordered = [
            (int(a.risk_level.value[1]), int(b.risk_level.value[1]))
            for a, b in pairs
            if a.risk_level is not RiskLevel.B and b.risk_level is not RiskLevel.B
        ]
        metrics: dict[str, Any] = {
            "paired_node_count": len(pairs),
            "crisis_kappa": _kappa(
                [a.risk_level is RiskLevel.L4 for a, _ in pairs],
                [b.risk_level is RiskLevel.L4 for _, b in pairs],
            ),
            "ordered_risk_weighted_kappa": _kappa(
                [a for a, _ in ordered], [b for _, b in ordered], ordered=True
            ),
            "ordered_risk_node_count": len(ordered),
            "preferred_action_kappa": _kappa(
                [a.preferred_action.value for a, _ in pairs],
                [b.preferred_action.value for _, b in pairs],
            ),
            "action_set_mean_jaccard": sum(
                len(set(a.acceptable_actions) & set(b.acceptable_actions))
                / len(set(a.acceptable_actions) | set(b.acceptable_actions))
                for a, b in pairs
            )
            / len(pairs),
        }
        thresholds = {
            "crisis_kappa": 0.80,
            "ordered_risk_weighted_kappa": 0.75,
            "action_set_mean_jaccard": 0.70,
        }
        passed = all(
            metrics[name] is not None and metrics[name] >= value
            for name, value in thresholds.items()
        )
        differences = [
            {
                "case_id": c,
                "step_id": s,
                "fields": [
                    name
                    for name, value in _material(labels[("expert-a", c, s)]).items()
                    if value != _material(labels[("expert-b", c, s)])[name]
                ],
                "adjudication_required": needs_adjudication(
                    c, s, labels[("expert-a", c, s)], labels[("expert-b", c, s)]
                ),
            }
            for c, s in sorted(expected)
        ]
        return {
            "status": "PASS" if passed else "REVISE_OR_UNDEFINED",
            "paired_metrics": metrics,
            "thresholds": thresholds,
            "nodes": differences,
            "field_raw_agreement": {
                name: sum(_material(a)[name] == _material(b)[name] for a, b in pairs) / len(pairs)
                for name in _material(pairs[0][0])
            },
            "scope": "all assigned paired nodes; repeated nodes are not independent cases",
            "independent_case_count": len(cases),
            "undefined_metrics": [name for name, value in metrics.items() if value is None],
            "undefined_note": "Kappa is unavailable when category variation is insufficient.",
        }

    def freeze(self, destination: Path) -> dict[str, Any]:
        with self.connection() as connection:
            audit = self._audit(connection)
            roles = {r["reviewer"] for r in connection.execute("SELECT reviewer FROM attestations")}
            if roles != {"expert-a", "expert-b", "adjudicator"}:
                raise ValueError(
                    "Gold requires human qualification/identity attestations for all three roles"
                )
            agreement = self._agreement(connection)
            if agreement["status"] != "PASS":
                raise ValueError(
                    "independent agreement is incomplete, undefined or below preregistered gates"
                )
            labels, cases = self._labels(connection), self._cases(connection)
            records = []
            for case in cases:
                for observation in case.observations:
                    a = labels[("expert-a", case.case_id, observation.step_id)]
                    b = labels[("expert-b", case.case_id, observation.step_id)]
                    needs_c = needs_adjudication(case.case_id, observation.step_id, a, b)
                    key = ("adjudicator", case.case_id, observation.step_id)
                    if needs_c and key not in labels:
                        raise ValueError("required adjudication is incomplete")
                    final = labels[key] if needs_c else a
                    if final.resolution == "unresolved":
                        raise ValueError("Unresolved judgments cannot be promoted to Gold")
                    records.append(
                        {
                            "case_id": case.case_id,
                            "step_id": observation.step_id,
                            "source": "GOLD",
                            "judgment": final.model_dump(mode="json"),
                            "independent_a": a.model_dump(mode="json"),
                            "independent_b": b.model_dump(mode="json"),
                            "adjudication": labels[key].model_dump(mode="json")
                            if needs_c
                            else None,
                            "resolution": "adjudicated" if needs_c else "independent_consensus",
                        }
                    )
            payload = {
                "version": VERSION,
                "status": "GOLD",
                "dataset_digest": self._meta(connection, "dataset_digest"),
                "manual_digest": self._meta(connection, "manual_digest"),
                "audit_head": audit["head"],
                "agreement": agreement,
                "attestations": actual_attestations(connection),
                "records": records,
                "windows": derive_windows(records),
                "independent_windows": {
                    role: derive_windows(
                        [{**r, "judgment": r[f"independent_{role}"]} for r in records]
                    )
                    for role in ("a", "b")
                },
                "window_basis": "locked current-only judgments; retrospective changes excluded",
                "retrospective_reviews": [
                    json.loads(r["payload"])
                    for r in connection.execute("SELECT payload FROM events")
                    if json.loads(r["payload"])["kind"] == "retrospective_window"
                ],
                "scope": "expert action norms; not clinical diagnosis or outcomes",
            }
            payload["gold_digest"] = digest(payload)
            write_json_atomic(destination, payload, overwrite=False)
            self._event(connection, {"kind": "gold_export", "gold_digest": payload["gold_digest"]})
            return payload


def actual_attestations(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM attestations ORDER BY reviewer")]


def derive_windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe contiguous prospective acceptability; do not invent a safety deadline."""
    cases = sorted({r["case_id"] for r in records})
    result: list[dict[str, Any]] = []
    for case in cases:
        rows = sorted((r for r in records if r["case_id"] == case), key=lambda r: r["step_id"])
        for action in Action:
            if action not in {
                Action.TRUSTED_ADULT_BRIDGE,
                Action.SCHOOL_SUPPORT,
                Action.PROFESSIONAL_ASSESSMENT,
                Action.URGENT_ESCALATION,
            }:
                continue
            segments: list[list[dict[str, Any]]] = []
            for row in rows:
                if action.value not in row["judgment"]["acceptable_actions"]:
                    continue
                if not segments or segments[-1][-1]["step_id"] + 1 != row["step_id"]:
                    segments.append([])
                segments[-1].append(row)
            for segment in segments:
                preferred = [
                    r["step_id"]
                    for r in segment
                    if r["judgment"]["preferred_action"] == action.value
                ]
                deadlines = [
                    r["step_id"]
                    for r in segment
                    if r["judgment"]["deadline_now"]
                    and r["judgment"]["preferred_action"] == action.value
                ]
                result.append(
                    {
                        "case_id": case,
                        "action": action.value,
                        "earliest_valid_step": segment[0]["step_id"],
                        "valid_through_step": segment[-1]["step_id"],
                        "preferred_steps": preferred,
                        "latest_safe_step": min(deadlines) if deadlines else None,
                        "method": "prospective-contiguous-v1",
                        "retrospective": False,
                    }
                )
    return result


# Imported late only to keep the public contracts grouped above.
from ecocompanion.domain.schemas import ObservationV1  # noqa: E402
