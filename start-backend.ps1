$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$envFile = Join-Path $projectRoot ".env"
$pythonPath = Join-Path $projectRoot ".venv-1\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "Missing .env file at $envFile. Create it from .env.example and configure the database values."
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "The project virtual environment was not found at $pythonPath. Create .venv-1 and install dependencies with: .venv-1\Scripts\python.exe -m pip install -r requirements.txt"
}

foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        $name = $Matches[1]
        $value = $Matches[2]
        if ($value.Length -ge 2 -and
            (($value.StartsWith('"') -and $value.EndsWith('"')) -or
             ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -TimeoutSec 2
    if ($health.status -eq "ok") {
        Write-Host "CarWorld API is already running at http://127.0.0.1:8000"
        exit 0
    }
} catch {
    # No healthy API is listening; start the local development server below.
}

Write-Host "Starting CarWorld API at http://127.0.0.1:8000"
Write-Host "Press Ctrl+C to stop the backend."
& $pythonPath -m uvicorn --app-dir (Join-Path $projectRoot "backend") app.main:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
