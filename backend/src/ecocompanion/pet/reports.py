"""Local, user-directed wellbeing records. No diagnoses or inferred scores."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ecocompanion.core.policy import PolicyOutcome
from ecocompanion.domain.enums import ActorRole
from ecocompanion.pet.pipeline import turn_context


class CheckIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    mood: int = Field(ge=1, le=5)
    energy: int = Field(ge=1, le=5)
    sleep: int = Field(ge=1, le=5)
    stress: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=2000)


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = ""
    source: Literal["local", "personal_wechat", "qq", "wechat"] = "local"
    message_ids: list[str] = Field(default_factory=list, max_length=40)
    check_in: CheckIn | None = None
    summary: str = Field(default="", max_length=4000)
    preview_id: str | None = None
    include_assessment: bool = True


@dataclass(frozen=True)
class Evidence:
    message_id: str
    content: str
    created_at: str


class _Base(DeclarativeBase):
    pass


class _CheckInRow(_Base):
    __tablename__ = "check_ins"
    local_date: Mapped[str] = mapped_column(String(10), primary_key=True)
    mood: Mapped[int] = mapped_column(Integer, nullable=False)
    energy: Mapped[int] = mapped_column(Integer, nullable=False)
    sleep: Mapped[int] = mapped_column(Integer, nullable=False)
    stress: Mapped[int] = mapped_column(Integer, nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class _ReportRow(_Base):
    __tablename__ = "report_snapshots"
    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    document_json: Mapped[str] = mapped_column(Text, nullable=False)


class _EvidenceRow(_Base):
    __tablename__ = "report_evidence"
    report_id: Mapped[str] = mapped_column(
        ForeignKey("report_snapshots.report_id", ondelete="CASCADE"),
        primary_key=True,
    )
    message_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ReportStore:
    def __init__(self, database: Path) -> None:
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
        self.sessions = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(_Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def save(
        self, check_in: CheckIn | None, document: dict, evidence: list[Evidence]
    ) -> dict:
        now = datetime.now(UTC)
        async with self.sessions.begin() as db:
            if check_in is not None:
                row = await db.get(_CheckInRow, check_in.local_date)
                values = check_in.model_dump()
                if row is None:
                    db.add(_CheckInRow(**values, updated_at=now))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
                    row.updated_at = now
            db.add(
                _ReportRow(
                    report_id=document["report_id"],
                    created_at=now,
                    document_json=json.dumps(document, ensure_ascii=False),
                )
            )
            db.add_all(
                _EvidenceRow(
                    report_id=document["report_id"],
                    message_id=item.message_id,
                    session_id=document["session_id"],
                    content=item.content,
                    message_created_at=item.created_at,
                )
                for item in evidence
            )
        return document

    async def list(self) -> list[dict]:
        async with self.sessions() as db:
            rows = (
                await db.scalars(select(_ReportRow).order_by(_ReportRow.created_at.desc()))
            ).all()
        return [self._summary(row) for row in rows]

    async def get(self, report_id: str) -> dict | None:
        async with self.sessions() as db:
            row = await db.get(_ReportRow, report_id)
            return None if row is None else json.loads(row.document_json)

    async def delete(self, report_id: str) -> bool:
        async with self.sessions.begin() as db:
            row = await db.get(_ReportRow, report_id)
            if row is None:
                return False
            await db.execute(delete(_EvidenceRow).where(_EvidenceRow.report_id == report_id))
            await db.delete(row)
            return True

    async def remove_session_evidence(self, session_id: str) -> None:
        async with self.sessions.begin() as db:
            report_ids = set(
                await db.scalars(
                    select(_EvidenceRow.report_id).where(_EvidenceRow.session_id == session_id)
                )
            )
            await db.execute(delete(_EvidenceRow).where(_EvidenceRow.session_id == session_id))
            for report_id in report_ids:
                row = await db.get(_ReportRow, report_id)
                if row is None:
                    continue
                document = json.loads(row.document_json)
                document["evidence"] = []
                document["assessments"] = []
                if document.get("summary_source") != "user":
                    document["summary"] = ""
                    document["summary_source"] = "removed"
                document["source_note"] = "关联会话已删除，相关原话与自动摘要已移除。"
                row.document_json = json.dumps(document, ensure_ascii=False)

    @staticmethod
    def _summary(row: _ReportRow) -> dict:
        document = json.loads(row.document_json)
        return {
            "report_id": row.report_id,
            "created_at": document["created_at"],
            "check_in": document.get("check_in"),
            "has_summary": bool(document.get("summary")),
        }


def assessment_snapshot(
    user_message_id: str, used_fallback: bool = False, outcome: PolicyOutcome | None = None
) -> dict:
    context = turn_context.get()
    decision = context.decision or {
        "action": "local_safety_policy",
        "level": "preserved_by_local_policy",
        "reason_codes": ["LOCAL_SAFETY_SHORT_CIRCUIT"],
    }
    if outcome is not None:
        decision = {
            **decision,
            "action": outcome.plan.primary_action.value,
            "level": outcome.plan.escalation_level.value,
            "reason_codes": list(outcome.reason_codes),
        }
    return {
        "schema": "ecocompanion-turn-assessment-v1",
        "message_id": user_message_id,
        "qwen": (
            {
                "status": "ok",
                "model": context.settings.qwen_model,
                **context.assessment.model_dump(),
            }
            if context.assessment is not None
            else {
                "status": next(
                    (
                        stage["state"]
                        for stage in reversed(context.stages)
                        if stage["name"] == "qwen"
                    ),
                    "unavailable",
                )
            }
        ),
        "decision": decision,
        "stages": context.stages,
        "reply_source": "local_safety_fallback" if used_fallback else "deepseek",
    }


async def evidence_for_request(repository, request: ReportRequest) -> list[Evidence]:
    if len(set(request.message_ids)) != len(request.message_ids):
        raise ValueError("请勿重复选择同一条消息")
    if not request.session_id or not request.message_ids:
        if request.message_ids:
            raise ValueError("请选择有效的会话")
        return []
    messages = {
        item.message_id: item for item in await repository.list_messages(request.session_id)
    }
    evidence = []
    for message_id in request.message_ids:
        message = messages.get(message_id)
        if message is None or message.role is not ActorRole.USER:
            raise ValueError("只能选择当前会话中存在的本人消息")
        evidence.append(
            Evidence(message.message_id, message.content, message.created_at.isoformat())
        )
    return evidence


def extractive_summary(evidence: list[Evidence]) -> str:
    if not evidence:
        return ""
    excerpts = [item.content.strip()[:180] for item in evidence[:4]]
    return "本次记录选取了以下本人表达：" + "；".join(f"“{text}”" for text in excerpts)


def new_document(
    request: ReportRequest,
    evidence: list[Evidence],
    summary: str,
    source: str,
    assessments: list[dict] | None = None,
) -> dict:
    snapshots = assessments or []
    return {
        "schema": "ecocompanion-wellbeing-report-v2",
        "report_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "session_id": request.session_id or None,
        "source": request.source,
        "check_in": request.check_in.model_dump() if request.check_in else None,
        "summary": summary,
        "summary_source": source,
        "evidence": [asdict(item) for item in evidence],
        "assessments": snapshots if request.include_assessment else [],
        "source_scope": "selected_messages",
        "method_note": (
            "工作分层来自所选消息对应的 Qwen 结构化评估；Jev 只提供经程序校验的动作建议。"
            "未启用、失败或信息不足时如实标注，不以缺失结果推断心理状态。"
        ),
        "notice": (
            "这是用户主动创建的自我记录，不是医学诊断、量表结果或治疗建议。1-5 分仅用于自我回顾。"
        ),
    }


def report_print_html(document: dict) -> str:
    check_in = document.get("check_in")
    ratings = (
        ""
        if not check_in
        else "".join(
            f"<div><dt>{escape(label)}</dt><dd>{check_in[key]} / 5</dd></div>"
            for key, label in (
                ("mood", "心情"),
                ("energy", "精力"),
                ("sleep", "睡眠"),
                ("stress", "压力"),
            )
        )
    )
    note = (
        ""
        if not check_in or not check_in["note"]
        else f"<h2>今日备注</h2><p>{escape(check_in['note'])}</p>"
    )
    summary = (
        ""
        if not document.get("summary")
        else f"<h2>对话摘要</h2><p>{escape(document['summary'])}</p>"
    )
    evidence_html = "".join(
        f"<blockquote>{escape(item['content'])}</blockquote>" for item in document["evidence"]
    )
    quotes_html = "" if not evidence_html else f"<h2>选取的本人原话</h2>{evidence_html}"
    source_note = (
        "" if not document.get("source_note") else f"<p>{escape(document['source_note'])}</p>"
    )
    assessment_html = ""
    if document.get("assessments"):
        blocks = []
        for item in document["assessments"]:
            qwen = item.get("qwen") or {}
            decision = item.get("decision") or {}
            if qwen.get("status") == "ok" and "wellbeing_level" not in qwen:
                qwen = {**qwen, "status": "legacy_without_level"}
            if qwen.get("status") != "ok":
                blocks.append(
                    "<section class='assessment'><h3>本轮评估未完成</h3>"
                    "<p>模型未返回可用的结构化结果，不能据此给出心理等级。</p></section>"
                )
                continue
            level = {
                "steady": "日常稳定", "strained": "轻度困扰",
                "distressed": "明显困扰", "uncertain": "信息不足",
            }.get(qwen.get("wellbeing_level"), "信息不足")
            safety = {
                "none": "未提取到当前安全线索", "uncertain": "当前安全信息不足",
                "current": "提取到当前安全线索，需按安全流程关注",
            }.get(qwen.get("safety_signal"), "未知")
            quotes = "".join(
                f"<blockquote>{escape(str(quote))}</blockquote>"
                for quote in qwen.get("evidence_quotes", [])
            )
            jev = decision.get("jev") or {}
            jev_text = (
                f"Jev 建议：{escape(str(jev.get('choice', '未提供')))}；"
                f"最终动作：{escape(str(decision.get('action', '未记录')))}；"
                f"安全等级：{escape(str(decision.get('level', '未记录')))}"
            )
            blocks.append(
                f"<section class='assessment'><h3>支持需要：{level}</h3>"
                "<p>"
                + escape(
                    str(qwen.get("rationale") or qwen.get("support_focus") or "未提供说明")
                )
                + "</p>"
                f"<p>{safety}</p>{quotes}<p class='method'>{jev_text}</p></section>"
            )
        assessment_html = "<h2>模型评估与决策依据</h2>" + "".join(blocks)
        assessment_html += f"<p class='notice'>{escape(document.get('method_note', ''))}</p>"
    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='referrer' content='no-referrer'><title>心伴 · 自我记录</title>
<style>
body {{font:16px/1.9 system-ui,sans-serif;color:#263b31;max-width:720px;
margin:40px auto;padding:0 24px}}
h1 {{font-size:25px}}
h2 {{font-size:16px;margin-top:28px;border-bottom:1px solid #ccd9cf;padding-bottom:7px}}
dl {{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
dl div {{border:1px solid #dbe5dd;padding:12px}}
dt {{font-size:13px;color:#617b67}}
dd {{margin:4px 0 0;font-size:21px}}
blockquote {{margin:12px 0;padding:10px 15px;border-left:3px solid #8aab96;
background:#f5f8f5}}
p,blockquote {{white-space:pre-wrap;overflow-wrap:anywhere}}
.notice {{font-size:12px;color:#66776b;margin-top:32px}}
.print-action {{border:1px solid #8aab96;background:#fff;color:#263b31;padding:8px 14px;
border-radius:4px;cursor:pointer}}
@media print {{body {{margin:0}} .print-action {{display:none}} @page {{margin:18mm}}}}
</style></head><body>
<button class='print-action' type='button' onclick='window.print()'>打印 / 导出 PDF</button>
<h1>心伴 · 自我记录</h1><time>{escape(document["created_at"])}</time>{source_note}
<h2>今日自评</h2><dl>{ratings}</dl>{note}{assessment_html}{summary}{quotes_html}
<p class='notice'>{escape(document["notice"])}</p></body></html>"""
