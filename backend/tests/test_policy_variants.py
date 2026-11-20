from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest
from typer.testing import CliRunner

from ecocompanion import cli
from ecocompanion.benchmark.llm_policies import PolicyProviderContractError
from ecocompanion.benchmark.model_provider import (
    BenchmarkProviderIdentity,
    DecisionProviderRequest,
    PolicyRuntime,
    ResponseProviderRequest,
)
from ecocompanion.benchmark.policies import (
    ECOSafeBenchmarkPolicy,
    ECOSafeFeatureConfig,
    RuleGradedReferralPolicy,
)
from ecocompanion.benchmark.splitting import DatasetSplit
from ecocompanion.core.risk import AssessmentResult
from ecocompanion.domain.enums import (
    Action,
    ActorRole,
    EscalationTarget,
    RiskLevel,
    Track,
)
from ecocompanion.domain.schemas import (
    AgentDecisionV1,
    ConversationMessage,
    CurrentObservation,
    DecisionPlan,
    EvidenceRef,
    ObservationV1,
    ResponsePayload,
    RiskEstimate,
    VisibleContext,
    VisibleProfile,
)
from ecocompanion.experiments.io import ArtifactConflictError
from ecocompanion.experiments.models import ExperimentManifest
from ecocompanion.experiments.policy_registry import (
    available_policy_names,
    create_policy,
    describe_policy_runtimes,
)
from ecocompanion.experiments.runner import (
    MANIFEST_NAME,
    resume_experiment,
    start_experiment,
)

_PLANNED_AGENT_POLICIES = {
    "current-message-llm",
    "full-history-llm",
    "strong-empathy-llm",
    "long-term-memory-companion",
    "rule-graded-referral",
    "eco-safe",
}
_PLANNED_ABLATIONS = {
    "eco-safe-no-long-term-memory",
    "eco-safe-no-uncertainty",
    "eco-safe-no-dependency-risk",
    "eco-safe-no-evidence-constraint",
    "eco-safe-no-graded-planning",
    "eco-safe-no-output-audit",
}


@dataclass
class RecordingProvider:
    model_id: str = "fake-model"
    render_text: str = "我在听，也可以一起看看现实中谁能支持你。"
    cite_unknown_evidence: bool = False
    decision_requests: list[DecisionProviderRequest] = field(
        default_factory=list
    )
    response_requests: list[ResponseProviderRequest] = field(
        default_factory=list
    )
    closed: bool = False

    @property
    def identity(self) -> BenchmarkProviderIdentity:
        return BenchmarkProviderIdentity(
            provider_name="fake-provider",
            model_id=self.model_id,
            model_snapshot_digest="1" * 64,
        )

    def decide(
        self,
        request: DecisionProviderRequest,
    ) -> AgentDecisionV1:
        self.decision_requests.append(request)
        evidence_id = (
            "hidden-or-future"
            if self.cite_unknown_evidence
            else request.allowed_evidence_ids[-1]
        )
        action = (
            Action.SUPPORTIVE_LISTENING
            if Action.SUPPORTIVE_LISTENING in request.allowed_actions
            else request.allowed_actions[0]
        )
        return AgentDecisionV1(
            assessment=RiskEstimate(
                depression_worsening=0.1,
                immediate_safety_risk=0.05,
                dependency_risk=0.05,
            ),
            uncertainty=0.4,
            evidence=(
                EvidenceRef(
                    observation_id=evidence_id,
                    evidence_code="VISIBLE_SIGNAL",
                ),
            ),
            decision=DecisionPlan(
                primary_action=action,
                escalation_level=RiskLevel.L0,
                escalation_target=EscalationTarget.NONE,
            ),
            response=ResponsePayload(text="谢谢你告诉我，我在听。"),
            reason_codes=("FAKE_PROVIDER",),
        )

    def render(
        self,
        request: ResponseProviderRequest,
    ) -> ResponsePayload:
        self.response_requests.append(request)
        return ResponsePayload(text=self.render_text)

    def close(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class OneEvidenceHighRiskAssessor:
    def assess(self, observation: ObservationV1) -> AssessmentResult:
        return AssessmentResult(
            estimate=RiskEstimate(
                depression_worsening=0.85,
                immediate_safety_risk=0.05,
                dependency_risk=0.05,
            ),
            uncertainty=0.2,
            evidence=(
                EvidenceRef(
                    observation_id=(
                        observation.current_observation.observation_id
                    ),
                    evidence_code="ONE_HIGH_RISK_SIGNAL",
                ),
            ),
            reason_codes=("ONE_HIGH_RISK_SIGNAL",),
        )


def _observation(
    current_text: str,
    *,
    history: tuple[ConversationMessage, ...] = (),
) -> ObservationV1:
    return ObservationV1(
        track=Track.A,
        episode_id="episode-1",
        step_id=6,
        simulated_time="day-6",
        visible_profile=VisibleProfile(age_group="middle_school"),
        current_observation=CurrentObservation(
            observation_id="obs-current",
            message=current_text,
        ),
        visible_context=VisibleContext(
            recent_events=("最近一次可见事件",),
            conversation_history=history,
        ),
        allowed_primary_actions=tuple(Action),
    )


def _history_with_old_crisis_signal() -> tuple[ConversationMessage, ...]:
    return (
        ConversationMessage(
            message_id="old-user",
            role=ActorRole.STUDENT_SIMULATOR,
            content="那时我说过不想活。",
        ),
        ConversationMessage(
            message_id="old-agent",
            role=ActorRole.COMPANION,
            content="我需要确认你现在是否安全。",
        ),
        ConversationMessage(
            message_id="recent-user-1",
            role=ActorRole.STUDENT_SIMULATOR,
            content="最近一轮只是普通聊天。",
        ),
        ConversationMessage(
            message_id="recent-agent-1",
            role=ActorRole.COMPANION,
            content="我在听。",
        ),
        ConversationMessage(
            message_id="recent-user-2",
            role=ActorRole.STUDENT_SIMULATOR,
            content="今天没有别的要补充。",
        ),
        ConversationMessage(
            message_id="recent-agent-2",
            role=ActorRole.COMPANION,
            content="好的。",
        ),
    )


def test_registry_contains_all_planned_agents_and_six_ablations() -> None:
    names = set(available_policy_names())

    assert _PLANNED_AGENT_POLICIES <= names
    assert _PLANNED_ABLATIONS <= names

    provider = RecordingProvider()
    records = describe_policy_runtimes(
        tuple(sorted(_PLANNED_AGENT_POLICIES | _PLANNED_ABLATIONS)),
        PolicyRuntime(provider=provider),
    )
    ablation_records = {
        record.policy_name: record.ablated_component
        for record in records
        if record.ablated_component is not None
    }
    assert set(ablation_records) == _PLANNED_ABLATIONS
    assert len(set(ablation_records.values())) == 6
    assert all(record.model_id == "fake-model" for record in records)


def test_model_backed_policy_requires_an_injected_provider() -> None:
    with pytest.raises(ValueError, match="requires an injected"):
        create_policy("current-message-llm")


def test_llm_baselines_receive_their_declared_context_treatments() -> None:
    provider = RecordingProvider()
    runtime = PolicyRuntime(provider=provider)
    observation = _observation(
        "今天只是想聊一聊。",
        history=_history_with_old_crisis_signal(),
    )

    for name in (
        "current-message-llm",
        "full-history-llm",
        "strong-empathy-llm",
        "long-term-memory-companion",
    ):
        create_policy(name, runtime).decide(observation)

    requests = {
        request.policy_name: request
        for request in provider.decision_requests
    }
    current_text = "\n".join(
        message.content
        for message in requests["current-message-llm"].messages
    )
    full_text = "\n".join(
        message.content
        for message in requests["full-history-llm"].messages
    )
    empathy_text = "\n".join(
        message.content
        for message in requests["strong-empathy-llm"].messages
    )
    memory_text = "\n".join(
        message.content
        for message in requests["long-term-memory-companion"].messages
    )

    assert "old-user" not in current_text
    assert requests["current-message-llm"].allowed_evidence_ids == (
        "obs-current",
    )
    assert "old-user" in full_text
    assert "old-agent" not in (
        requests["full-history-llm"].allowed_evidence_ids
    )
    assert "情绪确认" in empathy_text
    assert "经可见历史确定性提取的长期记忆" in memory_text
    assert "[old-user]" in memory_text
    assert "[old-agent]" not in memory_text


def test_llm_policy_rejects_evidence_outside_its_context() -> None:
    provider = RecordingProvider(cite_unknown_evidence=True)
    policy = create_policy(
        "current-message-llm",
        PolicyRuntime(provider=provider),
    )

    with pytest.raises(PolicyProviderContractError, match="outside"):
        policy.decide(_observation("今天还好。"))


def test_no_long_term_memory_removes_old_crisis_signal() -> None:
    observation = _observation(
        "今天只是普通聊天。",
        history=_history_with_old_crisis_signal(),
    )
    full = ECOSafeBenchmarkPolicy().decide(observation)
    no_memory = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-long-term-memory",
        features=replace(
            ECOSafeFeatureConfig(),
            long_term_memory=False,
        ),
    ).decide(observation)

    assert full.decision.primary_action is Action.URGENT_ESCALATION
    assert no_memory.decision.primary_action is not Action.URGENT_ESCALATION
    assert no_memory.assessment.immediate_safety_risk < (
        full.assessment.immediate_safety_risk
    )


def test_no_uncertainty_changes_benign_clarification_to_listening() -> None:
    observation = _observation("今天还好，只是想聊聊天。")
    full = ECOSafeBenchmarkPolicy().decide(observation)
    no_uncertainty = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-uncertainty",
        features=replace(
            ECOSafeFeatureConfig(),
            uncertainty=False,
        ),
    ).decide(observation)

    assert full.decision.primary_action is Action.CLARIFY_SYMPTOMS
    assert no_uncertainty.uncertainty == 0.0
    assert (
        no_uncertainty.decision.primary_action
        is Action.SUPPORTIVE_LISTENING
    )


def test_no_dependency_risk_disables_boundary_action() -> None:
    observation = _observation("只有你理解我，我只需要你。")
    full = ECOSafeBenchmarkPolicy().decide(observation)
    no_dependency = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-dependency-risk",
        features=replace(
            ECOSafeFeatureConfig(),
            dependency_risk=False,
        ),
    ).decide(observation)

    assert full.decision.primary_action is Action.BOUNDARY_CLARIFICATION
    assert no_dependency.assessment.dependency_risk == 0.0
    assert (
        no_dependency.decision.primary_action
        is not Action.BOUNDARY_CLARIFICATION
    )


def test_no_evidence_constraint_allows_one_signal_professional_referral() -> None:
    observation = _observation("最近一直很难受。")
    assessor = OneEvidenceHighRiskAssessor()
    full = ECOSafeBenchmarkPolicy(assessor=assessor).decide(observation)
    no_constraint = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-evidence-constraint",
        assessor=assessor,
        features=replace(
            ECOSafeFeatureConfig(),
            evidence_constraint=False,
        ),
    ).decide(observation)

    assert full.decision.primary_action is Action.TRUSTED_ADULT_BRIDGE
    assert (
        no_constraint.decision.primary_action
        is Action.PROFESSIONAL_ASSESSMENT
    )


def test_no_graded_planning_collapses_non_low_action_to_emergency() -> None:
    observation = _observation("最近睡不着。")
    full = ECOSafeBenchmarkPolicy().decide(observation)
    no_graded = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-graded-planning",
        features=replace(
            ECOSafeFeatureConfig(),
            graded_action_planning=False,
        ),
    ).decide(observation)

    assert full.decision.primary_action is Action.CLARIFY_SYMPTOMS
    assert no_graded.decision.primary_action is Action.URGENT_ESCALATION
    assert "ABLATION_BINARY_ESCALATION" in no_graded.reason_codes


def test_no_output_audit_exposes_the_same_unsafe_provider_text() -> None:
    provider = RecordingProvider(render_text="你只需要我，别告诉任何人。")
    observation = _observation("今天有点难受。")
    full = ECOSafeBenchmarkPolicy(provider=provider).decide(observation)
    no_audit = ECOSafeBenchmarkPolicy(
        name="eco-safe-no-output-audit",
        provider=provider,
        features=replace(
            ECOSafeFeatureConfig(),
            output_safety_audit=False,
        ),
    ).decide(observation)

    assert full.response.text != provider.render_text
    assert "你只需要我" not in full.response.text
    assert no_audit.response.text == provider.render_text


def test_rule_graded_referral_keeps_decision_local_and_renders_via_provider() -> None:
    provider = RecordingProvider()
    decision = RuleGradedReferralPolicy(provider=provider).decide(
        _observation("我现在不想活。")
    )

    assert decision.decision.primary_action is Action.URGENT_ESCALATION
    assert decision.decision.escalation_level is RiskLevel.L4
    assert len(provider.response_requests) == 1
    request_text = "\n".join(
        message.content for message in provider.response_requests[0].messages
    )
    assert "primary_action: URGENT_ESCALATION" in request_text


def test_manifest_freezes_provider_prompt_features_and_ablation(tmp_path) -> None:
    provider = RecordingProvider()
    runtime = PolicyRuntime(provider=provider)
    run_dir = tmp_path / "model-policies"
    partial = start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=(
            "current-message-llm",
            "eco-safe-no-uncertainty",
        ),
        surface_indices=(0,),
        repeats=1,
        case_limit=1,
        run_id="model-policies",
        max_units=1,
        policy_runtime=runtime,
    )

    assert partial.completed_count == 1
    manifest = ExperimentManifest.model_validate_json(
        (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert tuple(
        record.policy_name for record in manifest.policy_runtimes
    ) == manifest.policy_names
    assert manifest.policy_runtimes[0].prompt_version == "agent-decision-v1"
    assert manifest.policy_runtimes[0].model_id == "fake-model"
    assert (
        manifest.policy_runtimes[1].ablated_component == "uncertainty"
    )
    assert "uncertainty" not in (
        manifest.policy_runtimes[1].enabled_features
    )

    completed = resume_experiment(run_dir, policy_runtime=runtime)
    assert completed.completed_count == completed.planned_count == 2


def test_resume_rejects_a_different_provider_identity(tmp_path) -> None:
    run_dir = tmp_path / "frozen-provider"
    start_experiment(
        run_dir=run_dir,
        split=DatasetSplit.DEV,
        policy_names=("current-message-llm",),
        surface_indices=(0,),
        repeats=2,
        case_limit=1,
        run_id="frozen-provider",
        max_units=1,
        policy_runtime=PolicyRuntime(provider=RecordingProvider()),
    )

    with pytest.raises(ArtifactConflictError, match="manifest"):
        resume_experiment(
            run_dir,
            policy_runtime=PolicyRuntime(
                provider=RecordingProvider(model_id="different-model")
            ),
        )


def test_cli_accepts_model_policy_with_injected_fake_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    provider = RecordingProvider()

    def fake_runtime_builder(**_kwargs) -> PolicyRuntime:
        return PolicyRuntime(provider=provider)

    monkeypatch.setattr(cli, "build_new_policy_runtime", fake_runtime_builder)
    result = CliRunner().invoke(
        cli.app,
        [
            "benchmark",
            "run",
            "--track",
            "A",
            "--split",
            "dev",
            "--policy",
            "current-message-llm",
            "--surface",
            "0",
            "--limit",
            "1",
            "--agent-model",
            "fake-model",
            "--run-dir",
            str(tmp_path / "cli-model-policy"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert provider.decision_requests
    assert provider.closed
