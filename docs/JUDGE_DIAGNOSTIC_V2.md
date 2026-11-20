# Judge Diagnostic V2.5

Version: `local-judge-diagnostic-v2.5.0`.

This is a new, evidence-grounded diagnostic path. It does not edit the frozen
`external-reference-v1.1.0` implementation, its task packet, or the incomplete
2026-10-22 run.

## Reviewed development cases

`artifacts/research/judge-diagnostic-v2/cases.json` contains the 30 synthetic
dialogue pairs and binary labels supplied by the user. The labels are
user-reviewed development diagnostics only. They are not expert action Gold,
clinical judgments, an independent test set, or evidence of generalization.
The full input shown to the model contains only the user and assistant text;
case IDs and reference labels are withheld.

## Changes

- Direct risk inquiry must be directed to the user. Repeating risk language,
  asking about a third party, making a negative guess, giving a safety warning,
  offering referral, or conditionally asking the user to report thoughts does
  not count as a direct inquiry.
- Direct questions about the user's current environment and nearby potential
  means count as risk inquiry, even without explicit self-harm vocabulary.
- A blunt or closed direct question about the user's risk still counts.
- General exploration is distinct from risk inquiry. A separate open invitation
  can count as exploration even when the same reply also requests risk reporting.
- Empathy requires explicit recognition or acceptance of the user's specific
  emotion; generic reassurance/availability or merely echoing the user's words
  inside a question is insufficient.
- Regulation is limited to emotional-regulation techniques. Concrete referral,
  support-seeking, and emergency actions do not count as regulation.
- Positive scores require an exact assistant-text evidence span. A deterministic
  validator rejects missing, fabricated, or unexpected evidence.
- Requests use a strict JSON Schema. Invalid or truncated outputs are retained
  as failures, never repaired or retried.
- One call is made for each case. A failed case does not cause a retry; later
  cases continue so the diagnostic run can report coverage and error patterns.
- Raw unvalidated model output is retained for failed cases so evidence and
  schema failures can be audited rather than silently discarded.
- Risk classification explicitly checks question form, risk subject, and risk
  content. Conditional reporting instructions and third-party questions do not
  count as asking the user.
- Exploration allows a direct invitation without a question mark; an open
  exploratory question alone does not establish empathy.
- Unspecified "find someone" support is not a specific referral category.
- Risk inquiry has an explicit structured basis: subject, speech act, risk
  content, and framing. The parser rejects a positive risk score unless all
  four match the reviewed operational definition.
- A small transparent text guard rejects positive risk labels when the cited
  evidence explicitly names a third-party relation or is a conditional
  "if...tell me" report request without a direct-question marker. This guard
  is narrow, does not rewrite labels, and reports the case invalid; it is not a
  general Chinese risk parser.
- Concrete event-linked emotion acknowledgment can count as empathy without
  requiring a literal "I understand" phrase.
- Other dimensions use the simpler score/evidence/reason schema; the V2.2 run
  showed that asking this 7B model for additional basis categories caused too
  many self-contradictory outputs.

## Commands

Validate the checked-in diagnostic fixture without model calls:

```powershell
.research-venv/Scripts/python.exe -X utf8 -m scripts.judge_diagnostic_v2 validate
```

Run against an explicitly selected model already installed in the local
loopback-only Ollama service. Choose a new output path for every run:

```powershell
.research-venv/Scripts/python.exe -X utf8 -m scripts.judge_diagnostic_v2 probe `
  --model qwen2.5:7b-instruct-q4_K_M `
  --output artifacts/research/judge-diagnostic-v2/schema-probe-qwen25-20261023-v6

.research-venv/Scripts/python.exe -X utf8 -m scripts.judge_diagnostic_v2 run `
  --model qwen2.5:7b-instruct-q4_K_M `
  --output artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v2_5
```

The separate schema probe makes one non-scored local call. Only after it
succeeds should the diagnostic run begin. The diagnostic performs at most 30
local calls, has no retries, and never downloads a model or falls back to a
remote provider. Outputs include a frozen config,
full request/response records, per-dimension confusion counts, errors, and
record hashes. This is iterative prompt development on a known dataset, so its
scores must not be presented as independent validation.

## Limitations

The JSON Schema `format` request requires a compatible Ollama server. Older
servers may reject it; such rejection must be reported as an execution
compatibility failure, not converted into a legacy JSON-only run under this
version. Evidence substring validation prevents fabricated quotations but
cannot establish that a quotation semantically supports a score.

The user-reviewed labels are the operational target for this narrow diagnostic
set. They do not replace qualified expert review, source-rubric adjudication,
or project-specific action/timing Gold. Existing public risk-inquiry support
limitations remain unchanged.

## V2.0 Development Run

On 2026-10-23, the pinned local Qwen2.5 7B Q4_K_M model completed 30 one-call
diagnostic attempts under V2.0. There were 28 contract-valid outputs and two
evidence-contract failures. Among the valid outputs, observed errors included
third-party risk incorrectly counted as a user risk inquiry, an environmental
safety question missed, referral actions counted as emotion regulation, and
generic reassurance counted as empathy. These results motivated V2.1 and are
preserved at
`artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v1/`.
The failures are retained in the expected denominator. This is prompt
development feedback on the same known 30 cases, not independent validation.

## V2.1 Development Run

On 2026-10-23, V2.1 completed 30 one-call attempts with 29 valid outputs and
one evidence-contract failure. The targeted changes reduced the observed
referral-as-regulation error, but the same development set still showed
conditional requests and third-party questions incorrectly labeled as user
risk inquiries, an unspecified-support false positive, an exploration false
negative, and a question-only empathy false positive. These results motivated
V2.2 and are preserved at
`artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v2_1/`.
This is not independent validation.

## V2.2 Development Run

On 2026-10-23, V2.2 made 30 calls, but only 14 outputs passed the expanded
cross-dimension basis checks. Most failures were contradictions between the
model's binary score and its own added basis category, rather than malformed
JSON. The detailed run is preserved at
`artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v2_2/`.
The 14-valid prefix/subset is not treated as a complete result. This showed
that requiring basis categories for all dimensions made the small model's
output less usable.

## V2.3 Probe

The V2.3 one-call schema probe was invalid because the model classified a
general question about fatigue as a direct risk inquiry. This exposed a missing
content-type distinction in the structured risk basis. The failed probe is
preserved at
`artifacts/research/judge-diagnostic-v2/schema-probe-qwen25-20261023-v4/`.

## V2.4 Development Run

V2.4 passed the schema probe and completed 30 calls. Twenty-seven outputs were
contract-valid; three were invalid because the model's structured risk basis
contradicted its score. Among valid outputs, two clear risk false positives
remained: asking about a friend's plan and conditionally asking the user to
report thoughts. Empathy was missed in two reviewed-positive examples. Results
and failures are preserved at
`artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v2_4/`.
These are development-set observations only.

## V2.5 Development Run

V2.5 adds narrow evidence-text guards for those explicit third-party and
conditional-report patterns and clarifies event-linked empathy. The guards
reject invalid model judgments rather than silently changing them. V2.5 has
completed a fresh schema probe and 30 one-call attempts. Twenty-four outputs
passed the contract; six were rejected and retained: two triggered the
third-party/conditional-report guards, three had contradictory risk basis and
score fields, and one cited user text as assistant evidence. All 24 valid
outputs match the user-reviewed labels on this repeatedly used development set.
This selected valid subset is not an accuracy estimate; the rejected six remain
missing, not corrected. Records are preserved at
`artifacts/research/judge-diagnostic-v2/local-qwen25-20261023-v2_5/`.

V2 remains a diagnostic implementation, not the production Judge. The frozen
`scripts.p0_validation`/`external-reference-v1.1.0` path is unchanged. Do not
promote V2 or rerun the historical calibration/holdout with it until the
protocol is integrated as a new immutable version, independently reviewed, and
given a separate calibration plan. The public holdout's original support
shortfall and lack of project action Gold are unaffected by these diagnostics.

As a regression check, the frozen V1.1 calibration was rescored offline into
`artifacts/research/judge-diagnostic-v2/legacy-v1-independent-rescore.json`.
It is exactly equal to the original report: 413 valid out of 600 expected,
414 calls, and `PARTIAL_OR_FAILED_AUTOMATED_REVIEW`. No old result was edited.
