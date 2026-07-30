$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m app.server --open

