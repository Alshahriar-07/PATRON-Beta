# Train a PATRON candidate. Arguments are passed to `python -m patron train`.
# Examples:
#   .\scripts\run_training.ps1 --dry-run
#   .\scripts\run_training.ps1 --run-name e0_control_s42 --epochs 6 --train-stride 2 --seed 42
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "No virtual environment found. Run scripts/install.ps1 first."
    exit 2
}
if (-not $TrainArgs -or $TrainArgs.Count -eq 0) {
    Write-Error "usage: .\scripts\run_training.ps1 [patron train options] (e.g. --epochs 6 --train-stride 2)"
    exit 2
}

& $venvPython -m patron train @TrainArgs
exit $LASTEXITCODE
