$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".research-venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Missing locked project Python runtime." }
$env:PYTHONPATH = Join-Path $root "backend/src"
$env:PYTHONUTF8 = "1"
Write-Host "Personal WeChat test bridge. Whitelisted text only; Ctrl+C stops."
& $python -m scripts.wechat_test_bridge --run
