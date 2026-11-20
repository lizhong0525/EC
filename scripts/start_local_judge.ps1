param(
    [string]$RuntimeRoot = 'D:\EcoCompanionRuntime\ollama\v0.40.8',
    [string]$Models = 'D:\EcoCompanionRuntime\models',
    [Parameter(Mandatory = $true)][string]$Output
)

$ErrorActionPreference = 'Stop'
$archive = Join-Path $RuntimeRoot 'verified-download\ollama-windows-amd64.zip'
$executable = Join-Path $RuntimeRoot 'runtime\ollama.exe'
$expectedArchive = '04d1d43ac62c2deb0393307bb49fed7caf6b192de41bdd497aaa9488089d5552'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw 'Extract the verified official archive into runtime before starting.'
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedArchive) {
    throw 'Official archive SHA256 differs. Do not start this runtime.'
}
if (Test-Path -LiteralPath $Output) {
    throw 'Use a new runtime log directory; do not overwrite previous records.'
}
if (Get-NetTCPConnection -LocalPort 11434 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 11434 is in use. Inspect the existing service before starting another.'
}

New-Item -ItemType Directory -Path $Output | Out-Null
$Output = (Resolve-Path -LiteralPath $Output).Path
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = $Models
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
$env:OLLAMA_API_KEY = $null
$env:HTTP_PROXY = $null
$env:HTTPS_PROXY = $null
$env:ALL_PROXY = $null
$env:NO_PROXY = '*'
$process = Start-Process -FilePath $executable -ArgumentList 'serve' `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $Output 'server.stdout.log') `
    -RedirectStandardError (Join-Path $Output 'server.stderr.log')

$version = $null
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    if ($process.HasExited) { throw 'Local runtime exited; retain logs for inspection.' }
    try {
        $version = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' `
            -NoProxy -TimeoutSec 2
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $version) { throw 'Runtime startup not verified; retain process and logs for inspection.' }
if ($version.version -ne '0.40.8') { throw 'Server version differs from the pinned runtime.' }
$signature = Get-AuthenticodeSignature -LiteralPath $executable
$manifest = [ordered]@{
    version = 'local-judge-runtime-v1.0.0'
    created_at_utc = [DateTime]::UtcNow.ToString('o')
    process_id = $process.Id
    executable = $executable
    executable_sha256 = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
    official_archive_sha256 = $expectedArchive
    authenticode_status = [string]$signature.Status
    runtime_version = $version.version
    model_storage = $Models
    configuration = [ordered]@{
        OLLAMA_HOST = '127.0.0.1:11434'
        OLLAMA_NO_CLOUD = '1'
        OLLAMA_NUM_PARALLEL = '1'
        OLLAMA_MAX_LOADED_MODELS = '1'
        OLLAMA_FLASH_ATTENTION = '1'
        OLLAMA_KV_CACHE_TYPE = 'q8_0'
    }
    machine_wide_settings_changed = $false
    model_inference_calls = 0
    scope = 'RUNTIME_STARTUP_ONLY_NOT_MODEL_OR_CLINICAL_VALIDATION'
}
$manifest | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath (Join-Path $Output 'runtime.json') -Encoding utf8
$manifest | ConvertTo-Json -Depth 5
