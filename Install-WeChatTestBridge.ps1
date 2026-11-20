$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".research-venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Missing locked project Python runtime." }
if (-not [Environment]::Is64BitOperatingSystem) { throw "A 64-bit Windows environment is required." }
& $python -m pip install "wechatauto-replica==1.2.7.0"
if ($LASTEXITCODE -ne 0) { throw "Optional bridge dependency installation failed." }
# Account/chat validation belongs to bridge preflight, not component installation.
& $python -c "import wechatauto; from wechatauto import WeChatDB, WeChatGUI; print('WeChat bridge component import OK:', wechatauto.__version__)"
if ($LASTEXITCODE -ne 0) { throw "The installed bridge component could not be imported." }
