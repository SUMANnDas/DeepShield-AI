$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvRoot = Join-Path $projectRoot '.venv'
$pythonPath = Join-Path $venvRoot 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    Write-Error "DeepShield's existing virtual environment was not found at '$pythonPath'. Create or restore the project .venv first; this launcher does not install packages."
    exit 1
}

Set-Location -LiteralPath $projectRoot
$previousVirtualEnv = $env:VIRTUAL_ENV
$previousPath = $env:PATH
$previousBytecodeSetting = $env:PYTHONDONTWRITEBYTECODE
$exitCode = 0

try {
    $env:VIRTUAL_ENV = $venvRoot
    $env:PATH = (Join-Path $venvRoot 'Scripts') + [IO.Path]::PathSeparator + $previousPath
    $env:PYTHONDONTWRITEBYTECODE = '1'

    & $pythonPath -c 'import backend.app.main'
    if ($LASTEXITCODE -ne 0) {
        throw 'The backend preflight import failed. Check the existing .venv dependencies and the error above.'
    }

    Write-Host 'Starting DeepShield AI from the existing project .venv.' -ForegroundColor Cyan
    Write-Host 'Application: http://127.0.0.1:8000' -ForegroundColor Green
    Write-Host 'API docs:    http://127.0.0.1:8000/docs' -ForegroundColor Green
    Write-Host 'Stop the server with Ctrl+C.'

    & $pythonPath -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
    if ($LASTEXITCODE -ne 0) {
        throw "The backend exited with code $LASTEXITCODE. Check whether port 8000 is already in use."
    }
} catch {
    $exitCode = 1
    [Console]::Error.WriteLine("DeepShield startup failed: $($_.Exception.Message)")
} finally {
    if ($null -eq $previousVirtualEnv) {
        Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
    } else {
        $env:VIRTUAL_ENV = $previousVirtualEnv
    }
    $env:PATH = $previousPath
    if ($null -eq $previousBytecodeSetting) {
        Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONDONTWRITEBYTECODE = $previousBytecodeSetting
    }
}

if ($exitCode -ne 0) { exit $exitCode }
