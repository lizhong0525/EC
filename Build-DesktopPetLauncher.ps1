param([switch]$DesktopShortcut)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$compiler = Join-Path $env:WINDIR "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
if (-not (Test-Path -LiteralPath $compiler)) {
    throw "The Windows .NET Framework C# compiler is required."
}
$source = Join-Path $root "web/desktop/launcher/Program.cs"
$output = Join-Path $root "EcoCompanion.exe"
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /reference:System.Windows.Forms.dll "/out:$output" $source
if ($LASTEXITCODE -ne 0) { throw "Launcher build failed." }

if ($DesktopShortcut) {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktop "EcoCompanion.lnk"
    if (Test-Path -LiteralPath $shortcutPath) {
        throw "Desktop shortcut already exists; it has not been overwritten."
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $output
    $shortcut.WorkingDirectory = $root
    $shortcut.Description = "EcoCompanion desktop companion"
    $shortcut.Save()
}
Write-Output "Built: $output"
