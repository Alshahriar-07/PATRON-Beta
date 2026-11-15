# PATRON installer for Windows 10/11.
# Creates .venv, installs dependencies, verifies PyTorch and runs the health check.
# No administrator rights, no CUDA, no system-wide changes.
[CmdletBinding()]
param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Resolve-Python {
    if ($Python) { return $Python }
    foreach ($candidate in @("py -3.12", "python")) {
        $parts = $candidate.Split(" ")
        if (Get-Command $parts[0] -ErrorAction SilentlyContinue) {
            return $candidate
        }
    }
    return $null
}

$pythonCmd = Resolve-Python
if (-not $pythonCmd) {
    Write-Error "No Python interpreter found. Install Python 3.12 from python.org and re-run."
    exit 1
}

$version = & cmd /c "$pythonCmd -c `"import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')`""
if ($version.Trim() -ne "3.12") {
    Write-Error "PATRON requires Python 3.12, found $($version.Trim()) ($pythonCmd)."
    exit 1
}

if (-not (Test-Path ".venv")) {
    Write-Host "==> creating virtual environment"
    & cmd /c "$pythonCmd -m venv .venv"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$venvPython = ".\.venv\Scripts\python.exe"

Write-Host "==> upgrading pip"
& $venvPython -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> installing requirements (CPU PyTorch, a few minutes on first run)"
& $venvPython -m pip install --quiet -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> installing patron (editable, no dependency resolution)"
& $venvPython -m pip install --quiet -e . --no-deps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> verifying PyTorch"
& $venvPython -c "import torch; x = torch.ones(2,2) @ torch.ones(2,2); assert float(x.sum()) == 8.0; print(f'    torch {torch.__version__} ok (threads={torch.get_num_threads()})')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> health check"
& $venvPython -m patron health
$status = $LASTEXITCODE
if ($status -ne 0) {
    Write-Error "Health check reported FAIL. Fix the entries above, then re-run: .\.venv\Scripts\python.exe -m patron health"
    exit $status
}

Write-Host ""
Write-Host "Installed. Next:"
Write-Host "  .\.venv\Scripts\python.exe -m pytest"
Write-Host "  .\.venv\Scripts\python.exe -m patron health --json"
