$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildRoot = Join-Path $ProjectRoot "build"
$DistRoot = Join-Path $ProjectRoot "dist"
$AppDist = Join-Path $DistRoot "DataAgent"

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
if (Test-Path -LiteralPath $AppDist) {
    Remove-Item -LiteralPath $AppDist -Recurse -Force
}

python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name DataAgent `
    --exclude-module webview `
    --exclude-module pythonnet `
    --exclude-module clr `
    --exclude-module clr_loader `
    --exclude-module PyQt6 `
    --exclude-module PyQt5 `
    --exclude-module qtpy `
    --exclude-module pygame `
    --exclude-module cefpython3 `
    --exclude-module gi `
    --add-data "$ProjectRoot\static;static" `
    "$ProjectRoot\desktop.py"

Copy-Item -LiteralPath "$ProjectRoot\engines" -Destination "$AppDist\engines" -Recurse

$UvPath = (Get-Command uv -ErrorAction Stop).Source
Copy-Item -LiteralPath $UvPath -Destination "$AppDist\uv.exe"
Copy-Item -LiteralPath "$ProjectRoot\README.md" -Destination "$AppDist\README.md"

$ZipPath = Join-Path $DistRoot "DataAgent-Windows-x64.zip"
if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
Compress-Archive -Path "$AppDist\*" -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host "EXE: $AppDist\DataAgent.exe"
Write-Host "ZIP: $ZipPath"
