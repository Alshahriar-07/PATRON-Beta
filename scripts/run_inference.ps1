# Run one schema-validated inference.
#
# Required: -Checkpoint (path to best.pt) and -Input (96-candle JSON).
# Optional: -Warmup (parquet with >=384 prior 15m bars), -Output (JSON path).
#
#   .\scripts\run_inference.ps1 -Checkpoint experiments\patron_v2_20261114-203859\best.pt `
#       -Input examples\sample_input.json -Warmup data\clean\BTCUSDT\15m.parquet
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [Parameter(Mandatory = $true)][string]$Input,
    [string]$Warmup = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "No virtual environment found. Run scripts/install.ps1 first."
    exit 2
}

$args = @("--checkpoint", $Checkpoint, "--input", $Input)
if ($Warmup) { $args += @("--warmup", $Warmup) }
if ($Output) { $args += @("--output", $Output) }

& $venvPython -m patron predict @args
exit $LASTEXITCODE
