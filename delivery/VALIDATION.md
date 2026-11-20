# Validation Record

Date: 2026-10-27, Windows, Python 3.13 and Node.js in the developer workspace.

## Automated Verification

- 105 targeted backend tests passed: pet/model behavior, application persistence,
  conversation search, QQ/WeChat protocol simulation, settings and gateway runtime.
- 18 frontend unit tests passed, including real API turn behavior, provider failure,
  isolated server drafts, history race handling, rename/delete and channel secrets.
- Channel form hotfix: 18 frontend tests passed. Real-browser save -> start ->
  preflight produced PUT/POST 200 responses, retained one page and the settings
  dialog, and displayed the credential failure in the visible feedback region.
- `npm run build`: TypeScript and Vite production build passed.
- Ruff: passed for the changed pet, repository, messaging modules and targeted tests.
- Browser/native QA covers WebM alpha and motion, desktop startup, chat, 44 sessions,
  search, pagination, draft restoration, rename, settings and mobile layouts.
- Native report preview, saved-record selection and the actual Electron print/PDF window
  are exercised. AI summary consent/edit/save, session-evidence deletion and report
  deletion have API tests. Focus trapping includes selects and enabled textareas.
- Official channel preflight is tested with mocked QQ/WeChat endpoints and a real local
  listener. Tokens are checked not to appear in responses; this is not live account QA.
- Full backend suite: 831 passed, 1 skipped for Windows symlink privilege.
- Ollama 0.40.11 and qwen2.5:7b were downloaded. Real 29-case local baseline:
  29/29 schema-valid, 28/29 grounded quotes, 9/29 signal labels exact. This is an
  engineering fixture and not clinical validation; the unmodified model is not a
  psychological diagnostic or screening service.
- Real DeepSeek was previously verified once using an isolated synthetic turn;
  this packaging pass did not use saved credentials or send private conversations.

## Messaging Scope

QQ AccessToken/gateway/Identify/Resume/C2C/reply contracts and WeChat XML signature,
quick ACK, async customer-service text, idempotency, consent, isolation and deletion
are tested with synthetic messages and mocked official endpoints. Gateway startup
and cancellation are exercised locally. No real QQ or WeChat credentials, public
callback or account sandbox was available. This is not a claim of live platform
verification, permission approval, uptime, clinical effectiveness or crisis response.

The AstrBot-style channel pass adds UI-managed gateway lifecycle, an authenticated
loopback personal-WeChat JSON subset, automatic DPAPI bridge-token creation,
wxid allowlists, stable-ID deduplication and a separate opt-in UI transport.
53 targeted messaging/pet tests passed after the final lifecycle changes. Local
HTTP consent and duplicate suppression were exercised. Missing optional bridge
dependencies stop at preflight without importing or accessing WeChat. No real
personal account, database or window was accessed; real WeChat send/receive and
QQ sandbox delivery remain unverified. AstrBot/whale_gril source is not bundled.

## Exclusions And Residual Work

- No developer credentials, conversations, databases, traces, model weights,
  environments, `node_modules`, research datasets or full research suite are included.
- No clean-machine installer test. The launcher is a development quick launcher.
- WeChat AES callback compatibility, QQ groups/channels and personal WeChat groups
  are not implemented. Live passive-reply limits require platform sandbox validation.
  Personal WeChat UI automation requires the separately installed optional dependency
  and a dedicated test account; it is not an official personal-account API.
- Qwen 7B unquantized weights and Jev Chinese decision evaluation require further work.
  A separate Qwen2.5-0.5B-Instruct QLoRA smoke run completed 17 train/3 validation
  examples, 1 epoch, train_loss 3.3771 and eval_loss 2.3668. Adapter hashes and
  test-set exclusion were verified. It is not a validated psychological model,
  not a production 7B fine-tune, and is not deployed. See docs/QWEN_FINETUNING.md.
- Character-art permissions must be confirmed before public distribution.
