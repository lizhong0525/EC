$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".research-venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .research-venv. Use the project's Python environment first."
}
Push-Location -LiteralPath (Join-Path $root "web")
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally {
    Pop-Location
}
$env:PYTHONPATH = Join-Path $root "backend/src"
$env:PYTHONUTF8 = "1"
& $python -m uvicorn ecocompanion.games.app:app --host 127.0.0.1 --port 8765
