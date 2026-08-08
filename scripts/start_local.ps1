[CmdletBinding()]
param(
    [ValidatePattern('^[A-Za-z0-9.:-]+$')]
    [string]$ApiHost = '127.0.0.1',

    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8000,

    [ValidatePattern('^[A-Za-z0-9.:-]+$')]
    [string]$FrontendHost = '127.0.0.1',

    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 3000,

    [switch]$SeedDevelopmentUsers,
    [switch]$IncludeTestEvidence,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $projectRoot 'frontend'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nextScript = Join-Path $frontendRoot 'node_modules\next\dist\bin\next'
$apiProcess = $null
$frontendProcess = $null

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python virtual environment not found at '$pythonPath'. Install project dependencies first."
}
if (-not (Test-Path -LiteralPath $nextScript -PathType Leaf)) {
    throw "Frontend dependencies not found under 'frontend\node_modules'. Install them first."
}
$nodeCommand = Get-Command node.exe -ErrorAction SilentlyContinue
$nodePath = if ($null -ne $nodeCommand) { $nodeCommand.Source } else { $null }
if ($null -eq $nodePath) {
    # A standard Node installer puts node.exe here but only updates the PATH of
    # shells started afterwards, so an existing terminal (or a non-login shell)
    # sees an installed Node as missing. Check the default locations before
    # telling the user to install something they already have.
    foreach ($candidate in @(
        (Join-Path $env:ProgramFiles 'nodejs\node.exe'),
        (Join-Path ${env:ProgramFiles(x86)} 'nodejs\node.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\nodejs\node.exe')
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $nodePath = $candidate
            break
        }
    }
}
if ($null -eq $nodePath) {
    throw 'node.exe was not found on PATH or in the default install locations. Install Node.js before starting the web app.'
}
Write-Host "Using Node from '$nodePath'."

if ($SeedDevelopmentUsers) {
    if ($env:WADDEHHA_ENV -and $env:WADDEHHA_ENV.Trim().ToLowerInvariant() -ne 'development') {
        throw '-SeedDevelopmentUsers is only valid when WADDEHHA_ENV is development.'
    }
    $requiredSeedVariables = @(
        'WADDEHHA_DEV_OWNER_PASSWORD',
        'WADDEHHA_DEV_MANAGER_PASSWORD',
        'WADDEHHA_DEV_EMPLOYEE_PASSWORD'
    )
    $missingSeedVariables = @(
        $requiredSeedVariables | Where-Object {
            [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($_, 'Process'))
        }
    )
    if ($missingSeedVariables.Count -gt 0) {
        throw ('Development seed passwords are missing from the process environment: ' +
            ($missingSeedVariables -join ', '))
    }
    $env:WADDEHHA_ENV = 'development'
    $env:WADDEHHA_DEV_SEED_USERS = '1'
}
else {
    $env:WADDEHHA_DEV_SEED_USERS = '0'
}

if ($IncludeTestEvidence) {
    if ($env:WADDEHHA_ENV -and $env:WADDEHHA_ENV.Trim().ToLowerInvariant() -ne 'development') {
        throw '-IncludeTestEvidence is only valid when WADDEHHA_ENV is development.'
    }
    $env:WADDEHHA_ENV = 'development'
    $env:WADDEHHA_INCLUDE_TEST_EVIDENCE = '1'
}
else {
    $env:WADDEHHA_INCLUDE_TEST_EVIDENCE = '0'
}

if ([string]::IsNullOrWhiteSpace($env:WADDEHHA_ENV)) {
    $env:WADDEHHA_ENV = 'development'
}
if ([string]::IsNullOrWhiteSpace($env:WADDEHHA_API_DB)) {
    $env:WADDEHHA_API_DB = Join-Path $projectRoot 'db\api.sqlite3'
}

$apiOriginHost = $ApiHost
if ($ApiHost.Contains(':') -and -not $ApiHost.StartsWith('[')) {
    $apiOriginHost = "[$ApiHost]"
}
$env:API_ORIGIN = "http://${apiOriginHost}:$ApiPort"

if ($ValidateOnly) {
    Write-Host 'Local launcher validation passed. No processes were started.'
    exit 0
}

try {
    $apiProcess = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList @('-m', 'uvicorn', 'api.main:app', '--host', $ApiHost, '--port', $ApiPort) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    $frontendProcess = Start-Process `
        -FilePath $nodePath `
        -ArgumentList @('node_modules\next\dist\bin\next', 'dev', '--hostname', $FrontendHost, '--port', $FrontendPort) `
        -WorkingDirectory $frontendRoot `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Waddehha API: $($env:API_ORIGIN)"
    Write-Host "Waddehha web app: http://${FrontendHost}:$FrontendPort"
    Write-Host 'Press Ctrl+C to stop both child processes.'

    while ($true) {
        $apiProcess.Refresh()
        $frontendProcess.Refresh()
        if ($apiProcess.HasExited) {
            throw "The API process exited unexpectedly with code $($apiProcess.ExitCode)."
        }
        if ($frontendProcess.HasExited) {
            throw "The frontend process exited unexpectedly with code $($frontendProcess.ExitCode)."
        }
        Start-Sleep -Milliseconds 500
    }
}
finally {
    foreach ($childProcess in @($frontendProcess, $apiProcess)) {
        if ($null -eq $childProcess) {
            continue
        }
        try {
            $childProcess.Refresh()
            if (-not $childProcess.HasExited) {
                Stop-Process -Id $childProcess.Id -ErrorAction SilentlyContinue
                $null = $childProcess.WaitForExit(5000)
            }
        }
        catch {
            Write-Warning "Could not stop child process $($childProcess.Id): $($_.Exception.Message)"
        }
        finally {
            $childProcess.Dispose()
        }
    }
}
