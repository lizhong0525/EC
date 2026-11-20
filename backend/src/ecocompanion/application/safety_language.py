"""Scoped language cues for Web routing, separate from frozen benchmark scores."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ecocompanion.core.risk import DEFAULT_SIGNAL_RULES

SAFETY_CODES = frozenset(
    {"DIRECT_SELF_HARM_OR_SUICIDE_SIGNAL", "SELF_HARM_SIGNAL", "PASSIVE_DEATH_WISH", "HOPELESSNESS"}
)
_SIGNALS = tuple(
    phrase
    for rule in DEFAULT_SIGNAL_RULES
    if rule.evidence_code in SAFETY_CODES
    for phrase in rule.phrases
)
_HARM = r"(?:自杀|自残|伤害自己|伤害自身|自我伤害|结束(?:自己的)?生命|去死)"
_DENIED = re.compile(
    r"(?<!不是)(?<!并非)(?:没有|并无|没)(?:任何)?"
    r"(?:想死|想去死|想要自杀|自杀|自残|伤害自己|伤害自身|自我伤害|具体计划|自杀计划|计划)"
    r"(?:的?(?:念头|想法|冲动))?"
    r"|(?<!不是)(?<!并非)(?:不会|不打算|不准备|不想)"
    r"(?:在|正在)?(?:现在|马上|此刻|今晚|今天)?"
    r"(?:伤害自己|自杀|自残|结束(?:自己的)?生命|死)"
)


@dataclass(frozen=True, slots=True)
class SafetyLanguage:
    self_text: str
    third_party_text: str
    excluded_signal: bool

    @property
    def self_signal(self) -> bool:
        return any(phrase in self.self_text for phrase in _SIGNALS)

    @property
    def third_party_signal(self) -> bool:
        return any(phrase in self.third_party_text for phrase in _SIGNALS)


def safety_language(text: str) -> SafetyLanguage:
    """Keep affirmed clauses after a denial; never discard the whole message."""

    compact = re.sub(r"\s+", "", text)
    literary = bool(re.search(r"小说|文学|角色|台词|剧本|引用|这段文字", compact))
    parts = re.split(r"[，。！？!?；;\n]|但是|不过|可是|然而|但", compact)
    own: list[str] = []
    others: list[str] = []
    subject = "self"
    historical = False
    for part in parts:
        if not part:
            continue
        if re.search(r"去年|前年|以前|过去|曾经|当时|几年前|小时候", part):
            historical = True
        if re.search(r"现在|此刻|眼下|今晚|今天|最近|目前|还是|仍然", part):
            historical = False
        third = re.search(r"朋友|同学|同事|室友|家人|他|她|角色|台词", part)
        own_disclosure = re.search(
            rf"我(?:自己)?(?:也|现在|此刻|已经|正在|还是|仍然)?"
            rf"(?:想死|想去死|不想活|准备|打算|{_HARM})",
            part,
        )
        if third is not None:
            subject = "third"
        elif re.search(r"我(?:自己|也)?", part):
            subject = "self"
        # A separate own disclosure outranks a literary/third-person preface.
        if own_disclosure and (third is None or own_disclosure.start() < third.start()):
            subject = "self"
        if literary and subject == "third":
            continue
        cleaned = _DENIED.sub("", part)
        if not historical:
            (others if subject == "third" else own).append(cleaned)
    self_text = "，".join(own)
    third_text = "，".join(others)
    original_signal = any(phrase in text for phrase in _SIGNALS)
    return SafetyLanguage(
        self_text=self_text,
        third_party_text=third_text,
        excluded_signal=original_signal
        and not any(phrase in self_text or phrase in third_text for phrase in _SIGNALS),
    )


def immediate_language(text: str) -> bool:
    has_intent = bool(
        re.search(
            r"(?:已经|已)(?:想好|做好|准备|计划)"
            r"|(?:有|写好|准备好)(?:明确|具体|自杀)?(?:计划|遗书)"
            r"|(?:刀|药|绳子|工具|天台|楼顶).{0,8}(?:准备好|放在|在手里|拿着)"
            rf"|(?:打算|准备|计划).{{0,6}}(?:现在|今晚|今天).{{0,8}}{_HARM}"
            rf"|(?:马上|立刻|今晚|今天).{{0,8}}{_HARM}"
            rf"|(?:现在|此刻)(?:也|还是|仍然)?(?:就|要|准备|打算).{{0,8}}{_HARM}"
            rf"|(?:已经|正在).{{0,4}}{_HARM}"
            rf"|{_HARM}.{{0,8}}(?:马上|立刻|今晚|今天)",
            text,
        )
    )
    return has_intent and bool(re.search(rf"{_HARM}|不想活|想死|结束一切|撑不下去|活不下去", text))
