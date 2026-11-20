param([switch]$InstallOnly)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/ollama/ollama/releases/latest" -TimeoutSec 30
$asset = $release.assets | Where-Object name -eq "OllamaSetup.exe"
if (-not $asset) { throw "Official Ollama Windows installer was not found." }
$directory = Join-Path $root "tmp/runtime"
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$checksums = Join-Path $directory "ollama-sha256.txt"
$installer = Join-Path $directory "OllamaSetup.exe"
Invoke-WebRequest -Uri ($asset.browser_download_url -replace "OllamaSetup.exe$", "sha256sum.txt") -OutFile $checksums -TimeoutSec 60
$expected = ((Get-Content -LiteralPath $checksums | Where-Object { $_ -match "OllamaSetup.exe$" }) -split "\s+")[0]
if ((Test-Path -LiteralPath $installer) -and (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.ToLowerInvariant()) {
    throw "The cached installer hash is invalid. Inspect it before downloading a replacement."
}
if (-not (Test-Path -LiteralPath $installer)) {
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $installer -TimeoutSec 1800
}
if ((Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.ToLowerInvariant()) {
    throw "Official installer checksum mismatch. Installation has not started."
}
Write-Host "Verified official Ollama $($release.tag_name). Starting its per-user installer."
Start-Process -FilePath $installer -ArgumentList "/VERYSILENT","/NORESTART" -Wait -WindowStyle Hidden
$ollama = Join-Path $env:LOCALAPPDATA "Programs/Ollama/ollama.exe"
if (-not (Test-Path -LiteralPath $ollama)) { throw "Ollama installation was not completed." }
& $ollama --version
if (-not $InstallOnly) {
    Write-Host "Download the Qwen2.5 7B instruct model. This uses several GB of disk and network."
    & $ollama pull qwen2.5:7b
}
