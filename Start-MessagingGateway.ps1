$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".research-venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Install the locked Python dependencies into .research-venv first."
}
$env:PYTHONPATH = Join-Path $root "backend/src"
$env:PYTHONUTF8 = "1"
Write-Host "Starting explicitly enabled official messaging channels. Ctrl+C stops the gateway."
Write-Host "QQ C2C, Official Account callbacks and the authenticated local test bridge."
& $python -m ecocompanion.messaging.runtime
