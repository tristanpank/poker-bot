Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-OsName {
    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
        return "Windows"
    }

    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Linux)) {
        return "Linux"
    }

    if ([System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::OSX)) {
        return "OSX"
    }

    return "Unknown"
}

function Get-ComposeFiles {
    $files = @("docker-compose.yml")
    $osName = Get-OsName

    if ($osName -eq "Windows" -and (Test-Path "docker-compose.windows.yml")) {
        $files += "docker-compose.windows.yml"
    } elseif ($osName -eq "Linux" -and (Test-Path "docker-compose.linux.yml")) {
        $files += "docker-compose.linux.yml"
    }

    return $files
}

function Invoke-DockerCompose {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $composeFiles = Get-ComposeFiles
    $fileArgs = @()
    foreach ($file in $composeFiles) {
        $fileArgs += @("-f", $file)
    }

    & docker compose @fileArgs @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed: docker compose $($fileArgs + $Arguments -join ' ')"
    }
}

function Stop-ManagedTunnelProcesses {
    param(
        [string]$StatePath
    )

    if (-not (Test-Path $StatePath)) {
        return
    }

    try {
        $state = Get-Content $StatePath -Raw | ConvertFrom-Json
    } catch {
        return
    }

    foreach ($name in @("backend", "frontend")) {
        $entry = $state.$name
        if (-not $entry) {
            continue
        }

        $processId = $entry.pid
        if (-not $processId) {
            continue
        }

        try {
            $process = Get-Process -Id $processId -ErrorAction Stop
            if ($process.ProcessName -like "cloudflared*") {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
                Write-Host "Stopped $name cloudflared process ($processId)." -ForegroundColor Green
            }
        } catch {
            # Ignore stale PIDs.
        }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$statePath = Join-Path (Join-Path (Join-Path $repoRoot ".local") "phone-tunnels") "state.json"

Stop-ManagedTunnelProcesses -StatePath $statePath
Invoke-DockerCompose -Arguments @("stop", "frontend", "backend", "redis")

Write-Host "Stopped phone tunnel services." -ForegroundColor Green
