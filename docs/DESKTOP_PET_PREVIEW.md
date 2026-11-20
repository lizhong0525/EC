# Desktop Pet Preview v0.1

## Run

From the project directory on Windows:

Double-click `EcoCompanion.exe`, or the `EcoCompanion` desktop shortcut.
This no-console quick launcher reuses the built UI and starts a local service
automatically. It does not rebuild the frontend or require npm at launch.
Keep the exe in this project folder: it depends on `.research-venv`,
`web/node_modules/electron`, the backend sources and `web/dist`.
It is a local quick launcher, not a self-contained distributable installer.

Rebuild the launcher with the Windows .NET Framework compiler:

```powershell
.\Build-DesktopPetLauncher.ps1
# Add -DesktopShortcut to create a desktop shortcut (will not overwrite one).
.\EcoCompanion.exe --check
```

`--check` exits without opening any window: 0 means required files exist,
2 means a required file is missing. To rebuild the frontend too, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-DesktopPet.ps1
```

The launcher builds the React UI, opens the Electron desktop shell, and starts
its own loopback FastAPI server on an available port. Quitting the pet also
stops the server it owns. It does not start or download Ollama.

For a browser-only preview:

```powershell
.\.research-venv\Scripts\python.exe -m uvicorn ecocompanion.pet.api:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001 after building `web` with `npm run build`.
If the port is already occupied, choose another port. Existing research API,
commands, frontend and benchmark interfaces are retained in their original files.

## Behavior

- Transparent, draggable desktop character; click to toggle a separate chat window.
- Settings: DeepSeek API key, official endpoint, model, generation parameters,
  personality prompt, context size, timeouts, local Qwen, pinned Jev version,
  confidence gate, size, animations and always-on-top.
- Default cloud consent is OFF. Keys are not read from existing research
  environment variables, returned by the API, or put into browser storage.
- Windows keys are protected with current-user DPAPI. Non-Windows secrets are
  process-local only.
- Conversation, traces and settings live under `%LOCALAPPDATA%\EcoCompanion\pet`.
  `ECO_PET_HOME` overrides this location. Conversation/traces are not encrypted.
  This adult-test preview is not suitable for sensitive real clinical data.
- New conversation creates a separate session; it does not delete older records.
- The collapsible conversation sidebar lists existing sessions, newest activity
  first. Titles come from the first user message (no extra model calls).
  Search matches that opening message; pages load 40 items at a time.
- Switching loads each session's own messages and safety state. In-flight turns
  block switching. Unsent drafts are retained per session in memory until exit;
  they are not written to browser storage or sent to a model.
- The active session is remembered by the backend, including when the desktop
  runtime restarts with a different available port.
- No model training, paid API calls or large model downloads occur on startup.

## Model Flow

Local rule assessment and conversation policy run first. Qwen uses Ollama's
structured `/api/chat` output to extract evidence, not produce a clinical
diagnosis. Evidence quotes must match the current message.

Jev uses the actual TypeSafe `/v1/systemone` API with a pinned version and
Choice primitive. It receives original text, recent context and Qwen evidence.
Confidence gates ordinary suggestions; it is not a clinical accuracy measure.

Program code restricts suggestions to listening, clarification and safety
checks. Neither model can lower mandatory local actions. Local L3/L4 states
bypass all model calls and immediately release an audited local support reply.
Failed/invalid model output is shown as unavailable, never as a safe result.

DeepSeek renders the permitted action with personality preferences subordinate
to the existing system safety prompt. Existing deterministic safety/quality
audits may request one rewrite, then fall back locally. No raw unreviewed model
tokens are displayed.

DeepSeek uses the current official `deepseek-flash` or `deepseek-v4-pro` IDs.
Existing `deepseek-chat` / `deepseek-reasoner` settings are mapped to Flash,
with thinking disabled / enabled respectively. This adapter is pet-specific:
research model identity pinning is unchanged. New settings default to Flash
with thinking disabled for short, responsive conversations. The token limit
includes reasoning tokens; thinking may need a higher limit.

The explicit generation check sends one fixed synthetic greeting (no saved
dialogue or personality), verifies returned text and model identity, and incurs
a small API charge. It catches auth, balance and generation failures that a
model-list check cannot. UI diagnostics never echo provider error bodies.
Official references: https://api-docs.deepseek.com/quick_start/pricing and
https://api-docs.deepseek.com/guides/thinking_mode.

This is an integration preview: Qwen has not been domain-fine-tuned, Jev's
Chinese mental-health performance is unvalidated, and external human handoff
remains simulated. Do not describe the result as clinical detection.

## Animation Assets

Seven byte-for-byte original WebM clips are included, selected from the 106
reference clips. There is one character, not one character per model.
Serious safety states suppress random entertainment. Asset attribution and
the original restriction excerpt are in `web/public/pet/ATTRIBUTION.md`.

## Verification

```powershell
.\.research-venv\Scripts\python.exe -m pytest backend/tests/test_pet.py backend/tests/test_web_conversation.py backend/tests/test_application_service.py
cd web
npm test
npm run build
node desktop/verify.cjs --native
```
