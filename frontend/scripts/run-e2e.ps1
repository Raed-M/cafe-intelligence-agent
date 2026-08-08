$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$frontendRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $frontendRoot
$apiPort = 18000
$nextPort = 13000
$apiProcess = $null
$nextProcess = $null
$testExitCode = 1

function Assert-PortFree {
    param([int]$Port)

    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
    if ($listener) {
        $owners = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        throw "E2E port $Port is already in use by process(es): $owners"
    }
}

function Wait-ForHttp {
    param(
        [string]$Url,
        [System.Diagnostics.Process]$Process,
        [int]$TimeoutSeconds = 120
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        $Process.Refresh()
        if ($Process.HasExited) {
            throw "Process $($Process.Id) exited with code $($Process.ExitCode) before $Url became ready."
        }

        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 400) {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }

    throw "Timed out after $TimeoutSeconds seconds waiting for $Url (process $($Process.Id))."
}

function Stop-OwnedProcess {
    param([System.Diagnostics.Process]$Process)

    if ($null -eq $Process) {
        return
    }

    try {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue
            if (-not $Process.WaitForExit(5000)) {
                Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
                [void]$Process.WaitForExit(5000)
            }
        }
    }
    finally {
        $Process.Dispose()
    }
}

try {
    Assert-PortFree -Port $apiPort
    Assert-PortFree -Port $nextPort

    $evidenceRoot = Join-Path $projectRoot "outputs\test_evidence"
    New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null

    $e2ePassword = "Waddehha-E2E-Only-2026!"
    $env:WADDEHHA_ENV = "development"
    $env:WADDEHHA_API_DB = Join-Path $evidenceRoot "e2e_api.sqlite3"
    $env:WADDEHHA_DEV_SEED_USERS = "1"
    $env:WADDEHHA_DEV_OWNER_PASSWORD = $e2ePassword
    $env:WADDEHHA_DEV_MANAGER_PASSWORD = $e2ePassword
    $env:WADDEHHA_DEV_EMPLOYEE_PASSWORD = $e2ePassword
    $env:WADDEHHA_RUNS_ENABLED = "0"
    # Browser tests assert against a fixed provider/model configuration, so the
    # API must not pick up whatever happens to be in the developer's .env.
    $env:WADDEHHA_SKIP_DOTENV = "1"
    $env:API_ORIGIN = "http://127.0.0.1:$apiPort"

    $uvicornPath = Join-Path $projectRoot ".venv\Scripts\uvicorn.exe"
    $nodePath = (Get-Command node.exe -ErrorAction Stop).Source
    # Keep this path relative to WorkingDirectory. Start-Process flattens ArgumentList,
    # so an absolute project path containing spaces would be split before Node receives it.
    $nextCli = "node_modules\next\dist\bin\next"
    $playwrightCli = Join-Path $frontendRoot "node_modules\@playwright\test\cli.js"

    $apiProcess = Start-Process -FilePath $uvicornPath `
        -ArgumentList @("api.main:app", "--host", "127.0.0.1", "--port", "$apiPort") `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    Write-Host "Started FastAPI E2E server (PID $($apiProcess.Id), port $apiPort)."
    Wait-ForHttp -Url "http://127.0.0.1:$apiPort/api/health" -Process $apiProcess

    $nextProcess = Start-Process -FilePath $nodePath `
        -ArgumentList @($nextCli, "dev", "--hostname", "127.0.0.1", "--port", "$nextPort") `
        -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru
    Write-Host "Started Next.js E2E server (PID $($nextProcess.Id), port $nextPort)."
    Wait-ForHttp -Url "http://127.0.0.1:$nextPort/login" -Process $nextProcess

    $playwrightConfig = Join-Path $frontendRoot "playwright.config.ts"
    & $nodePath $playwrightCli test --config $playwrightConfig
    $testExitCode = $LASTEXITCODE
}
catch {
    Write-Host "E2E harness failed: $($_.Exception.Message)" -ForegroundColor Red
    $testExitCode = 1
}
finally {
    Stop-OwnedProcess -Process $nextProcess
    Stop-OwnedProcess -Process $apiProcess
}

exit $testExitCode
