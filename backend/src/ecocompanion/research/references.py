"""Private guideline ledger. Its Silver output cannot enter the expert Gold scorer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from ecocompanion.experiments.io import write_json_atomic
from ecocompanion.research.annotation import AnnotationStore, canonical, dataset_digest, digest
from ecocompanion.research.contracts import Prediction, Protocol, ResearchCase
from ecocompanion.research.measurement import (
    contract_issues,
    metric_summary,
    score_measurement,
)
from ecocompanion.research.reference_contracts import (
    REFERENCE_VERSION,
    GuidelineCatalog,
    GuidelineMapping,
    MappingReview,
    RuleReview,
)
from ecocompanion.research.workflows import require_private_output


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.suppressed = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.suppressed += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.suppressed = max(0, self.suppressed - 1)

    def handle_data(self, data: str) -> None:
        if not self.suppressed and data.strip():
            self.parts.append(data.strip())


def fetch_sources(catalog: GuidelineCatalog, output: Path) -> dict[str, Any]:
    """Explicit public HTTPS downloads; no model credentials, uploads or silent mirrors."""
    if output.exists():
        raise FileExistsError("choose a new source snapshot directory")
    output.mkdir(parents=True)
    records = []
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for source in catalog.sources:
            record: dict[str, Any] = {
                "source_id": source.source_id,
                "url": source.url,
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            try:
                response = client.get(source.url)
                record["http_status"] = response.status_code
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", ""):
                    raise ValueError(
                        "expected HTML; import other formats in a new reviewed catalog"
                    )
                parser = _Text()
                parser.feed(response.text)
                text = "\n".join(parser.parts)
                filename = f"{digest(source.source_id)}.json"
                write_json_atomic(
                    output / filename,
                    {
                        "source_id": source.source_id,
                        "url": source.url,
                        "raw_sha256": hashlib.sha256(response.content).hexdigest(),
                        "retrieved_at": record["retrieved_at"],
                        "text": text,
                    },
                    overwrite=False,
                )
                record.update(
                    status="FETCHED_NOT_EXPERT_REVIEWED",
                    file=filename,
                    file_sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                )
            except (httpx.HTTPError, ValueError) as exc:
                record.update(status="UNAVAILABLE", error=type(exc).__name__)
            records.append(record)
    manifest = {
        "version": REFERENCE_VERSION,
        "catalog_digest": digest(catalog.model_dump(mode="json")),
        "sources": records,
        "warning": "Retrieval is not verification of clinical applicability or authority.",
    }
    write_json_atomic(output / "manifest.json", manifest, overwrite=False)
    return manifest


def read_snapshots(directory: Path, catalog: GuidelineCatalog) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["catalog_digest"] != digest(catalog.model_dump(mode="json")):
        raise ValueError("source snapshots do not match catalog")
    expected = {s.source_id for s in catalog.sources}
    records = manifest["sources"]
    if len(records) != len(expected) or {r["source_id"] for r in records} != expected:
        raise ValueError("source snapshot coverage is not exact")
    result = {}
    for record in records:
        if record["status"] == "UNAVAILABLE":
            result[record["source_id"]] = record
            continue
        path = (directory / record["file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("snapshot path escapes source directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["file_sha256"]:
            raise ValueError("source snapshot hash mismatch")
        body = json.loads(path.read_text(encoding="utf-8"))
        source = next(s for s in catalog.sources if s.source_id == record["source_id"])
        if body["source_id"] != source.source_id or body["url"] != source.url:
            raise ValueError("source snapshot identity mismatch")
        result[source.source_id] = {**record, "snapshot": body}
    return result


class ReferenceStore:
    """Append-only local curator ledger; the OS administrator is trusted."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
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
        cls,
        output: Path,
        packet: Path,
        catalog: GuidelineCatalog,
        snapshots: Path,
        manual: Path,
    ) -> ReferenceStore:
        annotations = AnnotationStore(packet / "private" / "annotations.sqlite3")
        annotations.audit()
        cases = annotations.cases()
        require_private_output(cases, packet, output)
        if not output.resolve().is_relative_to((packet / "private").resolve()):
            raise ValueError("reference ledgers must remain in the packet private subtree")
        if output.exists():
            raise FileExistsError("choose a new versioned reference directory")
        sources = read_snapshots(snapshots, catalog)
        manual_text = manual.read_text(encoding="utf-8")
        output.mkdir(parents=True)
        path = output / "references.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TABLE events(seq INTEGER PRIMARY KEY, payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
                CREATE TRIGGER no_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT,'immutable reference event'); END;
                CREATE TRIGGER no_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT,'immutable reference event'); END;
            """)
        store = cls(path)
        with store.connection() as connection:
            store._append(
                connection,
                {
                    "kind": "created",
                    "version": REFERENCE_VERSION,
                    "cases": [c.model_dump(mode="json") for c in cases],
                    "dataset_digest": dataset_digest(cases),
                    "catalog": catalog.model_dump(mode="json"),
                    "catalog_digest": digest(catalog.model_dump(mode="json")),
                    "sources": sources,
                    "manual": manual_text,
                    "manual_digest": hashlib.sha256(manual.read_bytes()).hexdigest(),
                    "manual_version": "guideline-reference-operations-v1",
                    "parent_annotation_audit": annotations.audit(),
                    "scope": "partial guideline-derived Silver, never expert Gold",
                },
            )
        write_json_atomic(output / "status-initial.json", store.status(), overwrite=False)
        for model in (GuidelineCatalog, RuleReview, GuidelineMapping, MappingReview):
            write_json_atomic(
                output / "schemas" / f"{model.__name__}.schema.json",
                model.model_json_schema(),
                overwrite=False,
            )
        return store

    @staticmethod
    def _append(connection: sqlite3.Connection, payload: dict[str, Any]) -> str:
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
        return event_hash

    @staticmethod
    def _events(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        previous = "0" * 64
        result = []
        for seq, row in enumerate(connection.execute("SELECT * FROM events ORDER BY seq"), 1):
            body = json.loads(row["payload"])
            if (
                row["seq"] != seq
                or row["previous_hash"] != previous
                or digest([previous, body]) != row["event_hash"]
            ):
                raise ValueError("reference audit chain mismatch")
            previous = row["event_hash"]
            result.append({**body, "_event_hash": previous})
        if not result or result[0].get("kind") != "created":
            raise ValueError("reference ledger has no creation root")
        root = result[0]
        cases = tuple(ResearchCase.model_validate(c) for c in root["cases"])
        catalog = GuidelineCatalog.model_validate(root["catalog"])
        if root["dataset_digest"] != dataset_digest(cases) or root["catalog_digest"] != digest(
            catalog.model_dump(mode="json")
        ):
            raise ValueError("reference root digest mismatch")
        return result

    def audit(self) -> dict[str, Any]:
        with self.connection() as connection:
            events = self._events(connection)
        return {
            "status": "verified",
            "events": len(events),
            "head": events[-1]["_event_hash"],
            "warning": "Local integrity check, not proof of reviewer identity or expertise.",
        }

    def review_rule(self, review: RuleReview) -> str:
        review = RuleReview.model_validate(review.model_dump(mode="json"))
        with self.connection() as connection:
            events = self._events(connection)
            root = events[0]
            rule = next(
                (r for r in root["catalog"]["rules"] if r["rule_id"] == review.rule_id), None
            )
            if rule is None:
                raise ValueError("unknown guideline rule")
            source = root["sources"][rule["source_id"]]
            if review.disposition == "allow_silver_mapping":
                text = source.get("snapshot", {}).get("text", "")
                if source["status"] == "UNAVAILABLE" or rule["source_anchor"] not in text:
                    raise ValueError("rule has no retrievable matching source anchor")
            if any(
                e["kind"] == "rule_review" and e["review"]["rule_id"] == review.rule_id
                for e in events
            ):
                raise ValueError("rule review is locked; change catalog in a new ledger")
            return self._append(
                connection, {"kind": "rule_review", "review": review.model_dump(mode="json")}
            )

    def next_task(self, author_id: str) -> dict[str, Any]:
        author_id = author_id.strip()
        if not author_id.strip():
            raise ValueError("curator identity required")
        with self.connection() as connection:
            events = self._events(connection)
            completed = {
                (e["mapping"]["case_id"], e["mapping"]["step_id"])
                for e in events
                if e["kind"] == "mapping" and e["mapping"]["author_id"] == author_id
            }
            for case in events[0]["cases"]:
                for observation in case["observations"]:
                    if (case["case_id"], observation["step_id"]) not in completed:
                        return {
                            "status": "ready",
                            "case_id": case["case_id"],
                            "step_id": observation["step_id"],
                            "observation": observation,
                            "observation_digest": digest(observation),
                            "manual_digest": events[0]["manual_digest"],
                            "catalog_digest": events[0]["catalog_digest"],
                            "rule_review_status": {
                                r["rule_id"]: next(
                                    (
                                        e["review"]["disposition"]
                                        for e in events
                                        if e["kind"] == "rule_review"
                                        and e["review"]["rule_id"] == r["rule_id"]
                                    ),
                                    "PENDING",
                                )
                                for r in events[0]["catalog"]["rules"]
                            },
                        }
        return {"status": "complete"}

    def submit(self, mapping: GuidelineMapping) -> str:
        mapping = GuidelineMapping.model_validate(mapping.model_dump(mode="json"))
        with self.connection() as connection:
            events = self._events(connection)
            root = events[0]
            case = next((c for c in root["cases"] if c["case_id"] == mapping.case_id), None)
            if case is None or mapping.step_id >= len(case["observations"]):
                raise ValueError("unknown reference node")
            observation = case["observations"][mapping.step_id]
            if mapping.observation_digest != digest(observation):
                raise ValueError("reference observation digest mismatch")
            visible = {
                observation["current_observation"]["observation_id"]: observation[
                    "current_observation"
                ]["message"],
                **{
                    m["message_id"]: m["content"]
                    for m in observation["visible_context"]["conversation_history"]
                },
            }
            if any(
                e.observation_id not in visible or e.quote not in visible[e.observation_id]
                for e in mapping.evidence
            ):
                raise ValueError("evidence is future, unknown or not an exact visible quote")
            prior = [
                e
                for e in events
                if e["kind"] == "mapping"
                and e["mapping"]["author_id"] == mapping.author_id
                and (e["mapping"]["case_id"], e["mapping"]["step_id"])
                == (mapping.case_id, mapping.step_id)
            ]
            if prior:
                if mapping.supersedes != prior[-1]["_event_hash"]:
                    raise ValueError("correction must explicitly supersede the latest original")
                if mapping.origin != prior[-1]["mapping"]["origin"]:
                    raise ValueError("corrections cannot relabel an AI author as a human")
            elif mapping.supersedes:
                raise ValueError("cannot supersede a nonexistent mapping")
            else:
                seen = {
                    (e["mapping"]["case_id"], e["mapping"]["step_id"])
                    for e in events
                    if e["kind"] == "mapping" and e["mapping"]["author_id"] == mapping.author_id
                }
                order = [
                    (c["case_id"], o["step_id"]) for c in root["cases"] for o in c["observations"]
                ]
                if next((key for key in order if key not in seen), None) != (
                    mapping.case_id,
                    mapping.step_id,
                ):
                    raise ValueError("map the current node before opening the next")
            rules = {r["rule_id"]: r for r in root["catalog"]["rules"]}
            fields = mapping.norm.model_dump(mode="json", exclude_none=True)
            supported = set()
            for application in mapping.applications:
                rule = rules.get(application.rule_id)
                if rule is None:
                    raise ValueError("unknown mapping rule")
                reviews = [
                    e["review"]
                    for e in events
                    if e["kind"] == "rule_review" and e["review"]["rule_id"] == application.rule_id
                ]
                if not reviews or reviews[-1]["disposition"] != "allow_silver_mapping":
                    raise ValueError("rule source/scope review is pending or blocked")
                if (
                    set(application.conditions) != set(rule["conditions"])
                    or set(application.exclusions) != set(rule["exclusions"])
                    or set(application.conditions.values()) != {"met"}
                    or set(application.exclusions.values()) != {"absent"}
                ):
                    raise ValueError("all conditions must be met and exclusions absent")
                if len(set(application.fields)) != len(application.fields):
                    raise ValueError("duplicate supported fields")
                for field in application.fields:
                    if field not in fields or fields[field] != rule["conclusion"].get(field):
                        raise ValueError("mapping invents a field/value unsupported by the rule")
                    supported.add(field)
            if supported != set(fields):
                raise ValueError("every partial label needs explicit matching rule support")
            return self._append(
                connection,
                {
                    "kind": "mapping",
                    "mapping": mapping.model_dump(mode="json"),
                    "correction_is_retrospective": bool(mapping.supersedes),
                },
            )

    def review_mapping(self, review: MappingReview) -> str:
        review = MappingReview.model_validate(review.model_dump(mode="json"))
        with self.connection() as connection:
            events = self._events(connection)
            mapping = next(
                (
                    e
                    for e in events
                    if e["kind"] == "mapping" and e["_event_hash"] == review.mapping_digest
                ),
                None,
            )
            if mapping is None:
                raise ValueError("unknown mapping digest")
            if mapping["mapping"]["author_id"] == review.reviewer_id:
                raise ValueError("mapping authors cannot review their own mapping")
            if any(
                e["kind"] == "mapping_review"
                and e["review"]["mapping_digest"] == review.mapping_digest
                for e in events
            ):
                raise ValueError("review is locked; append a corrected mapping instead")
            return self._append(
                connection, {"kind": "mapping_review", "review": review.model_dump(mode="json")}
            )

    def status(self) -> dict[str, Any]:
        with self.connection() as connection:
            events = self._events(connection)
        root = events[0]
        records = []
        for case in root["cases"]:
            for observation in case["observations"]:
                key = case["case_id"], observation["step_id"]
                latest: dict[str, dict[str, Any]] = {}
                for event in events:
                    if (
                        event["kind"] == "mapping"
                        and (event["mapping"]["case_id"], event["mapping"]["step_id"]) == key
                    ):
                        latest[event["mapping"]["author_id"]] = event
                confirmed = []
                conflict = False
                for event in latest.values():
                    review = next(
                        (
                            e["review"]
                            for e in events
                            if e["kind"] == "mapping_review"
                            and e["review"]["mapping_digest"] == event["_event_hash"]
                        ),
                        None,
                    )
                    if review and review["disposition"] == "disagree":
                        conflict = True
                    if (
                        review
                        and review["disposition"] == "confirm"
                        and event["mapping"]["resolution"] == "mapped"
                        and event["mapping"]["origin"] == "human_curator"
                    ):
                        confirmed.append(event)
                norms = {canonical(e["mapping"]["norm"]) for e in confirmed}
                if len(norms) > 1:
                    conflict = True
                silver = bool(confirmed) and not conflict
                final = confirmed[0] if silver else None
                records.append(
                    {
                        "case_id": key[0],
                        "step_id": key[1],
                        "status": "SILVER" if silver else "UNRESOLVED",
                        "source": "GUIDELINE_DERIVED_SILVER"
                        if silver
                        else "PENDING_GUIDELINE_REVIEW",
                        "reason": "confirmed_partial_mapping"
                        if silver
                        else "mapping_disagreement"
                        if conflict
                        else "missing_or_unconfirmed_mapping",
                        "norm": final["mapping"]["norm"] if final else None,
                        "mapping_digest": final["_event_hash"] if final else None,
                        "lineage": [
                            e
                            for e in events
                            if (
                                e["kind"] == "mapping"
                                and (e["mapping"]["case_id"], e["mapping"]["step_id"]) == key
                            )
                            or (
                                e["kind"] == "mapping_review"
                                and e["review"]["mapping_digest"]
                                in {
                                    m["_event_hash"]
                                    for m in events
                                    if m["kind"] == "mapping"
                                    and (m["mapping"]["case_id"], m["mapping"]["step_id"]) == key
                                }
                            )
                        ],
                        "time_windows": None,
                    }
                )
        return {
            "version": REFERENCE_VERSION,
            "status": "SILVER_REFERENCE_LEDGER_NOT_GOLD",
            "dataset_digest": root["dataset_digest"],
            "catalog_digest": root["catalog_digest"],
            "manual_digest": root["manual_digest"],
            "audit_head": events[-1]["_event_hash"],
            "coverage": {
                "GOLD": 0,
                "SILVER": sum(r["status"] == "SILVER" for r in records),
                "UNRESOLVED": sum(r["status"] == "UNRESOLVED" for r in records),
                "total_nodes": len(records),
            },
            "source_status": {k: s["status"] for k, s in root["sources"].items()},
            "records": records,
            "scope": "Curator-reviewed partial guideline adaptation; not expert consensus.",
            "prospective_claim": "NOT_ESTABLISHED_CURATOR_HAS_WORKSPACE_ACCESS",
            "promotion_policy": (
                "No auto-promotion. Expert Gold requires the separate blinded workflow."
            ),
        }

    def cases(self) -> tuple[ResearchCase, ...]:
        with self.connection() as connection:
            return tuple(
                ResearchCase.model_validate(c) for c in self._events(connection)[0]["cases"]
            )


def reference_report(
    store: ReferenceStore,
    protocol: Protocol,
    predictions: Sequence[Prediction],
    *,
    simulation_truth: Mapping[tuple[str, int], tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Report Silver constraints separately; no Silver-to-Gold conversion or imputation."""
    cases, status = store.cases(), store.status()
    # Existing evaluator validates budgets, identities, duplicates and failed output denominators.
    report = score_measurement(cases, protocol, predictions, simulation_truth=simulation_truth)
    observations = {(c.case_id, o.step_id): o for c in cases for o in c.observations}
    references = {
        (r["case_id"], r["step_id"]): r for r in status["records"] if r["status"] == "SILVER"
    }
    by_key = {(p.policy, p.case_id, p.step_id, p.repeat): p for p in predictions}
    for policy in protocol.policy_names:
        values: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
        for key, row in references.items():
            norm = row["norm"]
            for repeat in range(protocol.repeats):
                p = by_key.get((policy, *key, repeat))
                decision = p.decision if p and not contract_issues(p, observations[key]) else None
                primary = decision.decision.primary_action.value if decision else None
                actions = (
                    {primary, *(a.value for a in decision.decision.secondary_actions)}
                    if decision
                    else set()
                )
                if norm["direct_safety_check_required"] is True:
                    values["required_safety_action_recall"].append(
                        (
                            key[0],
                            float(bool(actions & {"DIRECT_SAFETY_CHECK", "URGENT_ESCALATION"})),
                        )
                    )
                if norm["acceptable_actions"] is not None:
                    values["acceptable_primary_action_rate"].append(
                        (key[0], float(primary in norm["acceptable_actions"]))
                    )
                if norm["preferred_action"] is not None:
                    values["preferred_action_rate"].append(
                        (key[0], float(primary == norm["preferred_action"]))
                    )
                if norm["prohibited_actions"] is not None:
                    values["prohibited_action_rate"].append(
                        (key[0], float(bool(actions & set(norm["prohibited_actions"]))))
                    )
                if norm["risk_level"] is not None:
                    values["risk_action_level_accuracy"].append(
                        (
                            key[0],
                            float(
                                decision is not None
                                and decision.decision.escalation_level.value == norm["risk_level"]
                            ),
                        )
                    )
                if norm["referral_target"] is not None:
                    values["referral_target_accuracy"].append(
                        (
                            key[0],
                            float(
                                decision is not None
                                and decision.decision.escalation_target.value
                                == norm["referral_target"]
                            ),
                        )
                    )
        report["policies"][policy]["guideline_silver_constraints"] = {
            "status": "SILVER_PARTIAL_ONLY" if references else "UNAVAILABLE_NO_REVIEWED_SILVER",
            "metrics": metric_summary(cases, values, protocol) if values else None,
            "template_sensitivity": metric_summary(cases, values, protocol, template_groups=True)
            if values
            else None,
            "covered_nodes": len(references),
            "total_nodes": len(observations),
            "warning": (
                "Partial constraints only; absence of a rule is not evidence of safe conduct."
            ),
            "time_windows": "UNAVAILABLE_NO_EXPERT_PROSPECTIVE_LABELS",
        }
    report["reference_provenance"] = {k: v for k, v in status.items() if k != "records"}
    report["silver_field_support"] = dict(
        Counter(
            k
            for row in references.values()
            for k, value in row["norm"].items()
            if value is not None
        )
    )
    return report
