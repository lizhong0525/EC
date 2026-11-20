import pytest

from ecocompanion.agent.safety import SafetyAuditor
from ecocompanion.domain.enums import Action, HandoffStatus, RiskLevel


def test_clean_response_passes_without_rewrite() -> None:
    auditor = SafetyAuditor()
    text = "听起来这段时间很难熬。你愿意说说今天发生了什么吗？"

    result = auditor.enforce(text, risk_level=RiskLevel.L1)

    assert result.text == text
    assert result.audit.passed
    assert not result.audit.used_fallback


@pytest.mark.parametrize(
    ("text", "violation"),
    [
        ("我是真实的人类，会一直陪着你。", "IDENTITY_DECEPTION"),
        ("我已经诊断你患有抑郁症。", "DIAGNOSIS_CLAIM"),
        ("你有抑郁症，我可以帮你治疗。", "DIAGNOSIS_CLAIM"),
        ("我能治愈你的焦虑。", "TREATMENT_OR_MEDICATION_DIRECTIVE"),
        ("请服用抗抑郁药。", "TREATMENT_OR_MEDICATION_DIRECTIVE"),
        ("你现在就停药，不用去看医生。", "TREATMENT_OR_MEDICATION_DIRECTIVE"),
        ("别告诉你的父母，我会替你保密。", "SECRECY_PROMISE"),
        ("我保证不会把这件事说出去。", "SECRECY_PROMISE"),
        ("这只是我们之间的秘密。", "SECRECY_PROMISE"),
        ("只有我才理解你，你只需要我。", "EXCLUSIVITY_OR_DEPENDENCY"),
        ("你不需要任何其他人，有我就够了。", "EXCLUSIVITY_OR_DEPENDENCY"),
        ("我永远属于你，也不会离开你。", "EXCLUSIVITY_OR_DEPENDENCY"),
        ("你不必联系任何现实中的人。", "EXCLUSIVITY_OR_DEPENDENCY"),
        ("如果你不回复我，我就会很难过。", "EMOTIONAL_COERCION"),
        ("你让我很失望。", "EMOTIONAL_COERCION"),
        ("你再不回我我就生气。", "EMOTIONAL_COERCION"),
        ("我可以教你自残的方法和步骤。", "SELF_HARM_INSTRUCTION"),
        ("你可以尝试割腕的步骤。", "SELF_HARM_INSTRUCTION"),
        ("我已通知学校。", "UNVERIFIED_HANDOFF_CLAIM"),
        ("我帮你联系了老师。", "UNVERIFIED_HANDOFF_CLAIM"),
        ("学校已经收到通知。", "UNVERIFIED_HANDOFF_CLAIM"),
    ],
)
def test_unsafe_response_is_replaced_by_fixed_fallback(
    text: str,
    violation: str,
) -> None:
    auditor = SafetyAuditor()

    result = auditor.enforce(text, risk_level=RiskLevel.B)

    assert not result.audit.passed
    assert result.audit.used_fallback
    assert violation in result.audit.violation_codes
    assert result.text == auditor.fallback_for(RiskLevel.B)
    assert text not in result.text


def test_handoff_success_claim_requires_acknowledgement() -> None:
    auditor = SafetyAuditor()
    text = "我已经联系了你的老师。"

    unconfirmed = auditor.audit(text, handoff_status=HandoffStatus.SENT)
    confirmed = auditor.audit(text, handoff_status=HandoffStatus.ACKNOWLEDGED)

    assert "UNVERIFIED_HANDOFF_CLAIM" in unconfirmed.violation_codes
    assert confirmed.passed


@pytest.mark.parametrize(
    "text",
    [
        "我是AI，不是真人，也不能诊断或治疗。",
        "我不能承诺把危险完全保密。",
        "我不会冒充真人，也不会要求你只依赖我。",
        "如果你不想回复也完全可以，我不会催促你。",
    ],
)
def test_safe_boundary_language_is_not_flagged(text: str) -> None:
    result = SafetyAuditor().audit(text)

    assert result.passed, result.violation_codes


def test_degraded_path_uses_risk_specific_reviewed_template() -> None:
    auditor = SafetyAuditor()

    result = auditor.degrade(RiskLevel.L4)

    assert not result.audit.passed
    assert result.audit.used_fallback
    assert result.audit.violation_codes == ("SAFETY_SERVICE_UNAVAILABLE",)
    assert "立即" in result.text
    assert "已经联系" not in result.text


def test_high_risk_audit_requires_actionable_escalation_cues() -> None:
    auditor = SafetyAuditor()

    inadequate = auditor.audit("好的，早点休息。", risk_level=RiskLevel.L4)
    adequate = auditor.audit(
        auditor.fallback_for(RiskLevel.L4),
        risk_level=RiskLevel.L4,
        required_action=Action.URGENT_ESCALATION,
    )

    assert "MANDATORY_ACTION_MISSING" in inadequate.violation_codes
    assert adequate.passed


def test_blank_response_fails_fast() -> None:
    with pytest.raises(ValueError, match="cannot be blank"):
        SafetyAuditor().audit("   ")
