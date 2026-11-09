# Raphael Windows development agent: manual, bounded, non-deploying.
param(
  [Parameter(Mandatory=$true)][string]$Task,
  [switch]$Execute,
  [ValidateRange(60,3600)][int]$MaxSeconds = 1800
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
$python = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "py" }
$arguments = @()
if ($python -eq "py") { $arguments += "-3.12" }
$arguments += @("-m","scripts.windows_agent","--task",$Task,"--max-seconds","$MaxSeconds")
if ($Execute) { $arguments += "--execute" }
& $python @arguments
exit $LASTEXITCODE
