$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$web = Join-Path $root "web"
$electron = Join-Path $web "node_modules/electron/dist/electron.exe"
if (-not (Test-Path -LiteralPath $electron)) {
    throw "Run npm install in the web directory first."
}
Push-Location -LiteralPath $web
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    $main = Join-Path $web "desktop/main.cjs"
    Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
    Start-Process -FilePath $electron -ArgumentList @("`"$main`"") -WorkingDirectory $root -WindowStyle Hidden
} finally {
    Pop-Location
}
