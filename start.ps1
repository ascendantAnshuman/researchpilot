<#
.SYNOPSIS
    Start ResearchPilot locally: FastAPI backend (port 8000) + Next.js website (port 3000).

.DESCRIPTION
    First run: creates .env from .env.example (you then add your free GEMINI_API_KEY), creates the Python
    virtual environment and installs the backend and frontend dependencies. Every run: opens the backend and
    the website in two new PowerShell windows and opens http://localhost:3000 once the backend is healthy.
    Close the two windows (or press Ctrl+C in them) to stop ResearchPilot.

.PARAMETER Prod
    Build and serve the optimised production website instead of the development server.

.EXAMPLE
    .\start.ps1
    .\start.ps1 -Prod
#>
param([switch]$Prod)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

function Step($message) { Write-Host "==> $message" -ForegroundColor Cyan }

# 1. Configuration --------------------------------------------------------------------------------
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $root ".env.example") $envFile
    $bytes = New-Object byte[] 48
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $secret = [Convert]::ToBase64String($bytes)
    $lines = (Get-Content $envFile) -replace '^JWT_SECRET_KEY=.*$', "JWT_SECRET_KEY=$secret"
    [System.IO.File]::WriteAllLines($envFile, [string[]]$lines, (New-Object System.Text.UTF8Encoding $false))
    Write-Host ""
    Write-Host "Created .env. Open it and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)," -ForegroundColor Yellow
    Write-Host "then run .\start.ps1 again." -ForegroundColor Yellow
    exit 1
}
$keyLine = Select-String -Path $envFile -Pattern '^(GEMINI_API_KEY|ANTHROPIC_API_KEY)=\S+' -Quiet
if (-not $keyLine) {
    Write-Host "No API key in .env - set GEMINI_API_KEY (free: https://aistudio.google.com/apikey) and run again." -ForegroundColor Yellow
    exit 1
}

# 2. Backend dependencies (first run only) --------------------------------------------------------
if (-not (Test-Path $python)) {
    Step "Creating the Python 3.11 virtual environment (first run only)"
    & py -3.11 -m venv (Join-Path $backend ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Python 3.11 is required: https://www.python.org/downloads/release/python-3119/" }
    Step "Installing backend packages (5-10 minutes the first time)"
    & $python -m pip install --upgrade pip
    & $python -m pip install -r (Join-Path $backend "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed - see the messages above." }
}

# 3. Frontend dependencies (first run only) -------------------------------------------------------
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js 18.18+ is required: https://nodejs.org (LTS installer), then open a new terminal."
}
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Step "Installing website packages (first run only)"
    Push-Location $frontend
    try { & npm install; if ($LASTEXITCODE -ne 0) { throw "npm install failed." } } finally { Pop-Location }
}

# 4. Start both servers in their own windows ------------------------------------------------------
Step "Starting the backend on http://localhost:8000"
Start-Process powershell -WorkingDirectory $backend -ArgumentList @(
    "-NoExit", "-Command",
    "`$Host.UI.RawUI.WindowTitle = 'ResearchPilot backend'; & '$python' -m uvicorn app.main:app --port 8000"
)

if ($Prod) {
    Step "Building the production website (about a minute)"
    Push-Location $frontend
    try { & npm run build; if ($LASTEXITCODE -ne 0) { throw "npm run build failed." } } finally { Pop-Location }
    $webCommand = "npm start"
} else {
    $webCommand = "npm run dev"
}
Step "Starting the website on http://localhost:3000"
Start-Process powershell -WorkingDirectory $frontend -ArgumentList @(
    "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle = 'ResearchPilot website'; $webCommand"
)

# 5. Open the browser once the backend answers -----------------------------------------------------
Step "Waiting for the backend to start (loading the embedding model can take ~30 s)"
$healthy = $false
for ($i = 0; $i -lt 90 -and -not $healthy; $i++) {
    Start-Sleep -Seconds 2
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 3
        $healthy = $response.StatusCode -eq 200
    } catch { }
}
if ($healthy) {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:3000"
    Write-Host ""
    Write-Host "ResearchPilot is running:  website http://localhost:3000   API docs http://localhost:8000/docs" -ForegroundColor Green
    Write-Host "Stop it by closing the 'ResearchPilot backend' and 'ResearchPilot website' windows." -ForegroundColor Green
} else {
    Write-Host "The backend did not answer within 3 minutes - check the 'ResearchPilot backend' window for errors." -ForegroundColor Red
    exit 1
}
