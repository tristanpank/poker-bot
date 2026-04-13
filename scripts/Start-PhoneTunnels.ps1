param(
    [string]$FrontendLocalUrl,
    [string]$BackendLocalUrl,
    [string]$ShortPath,
    [string]$ShortIoApiKey,
    [string]$ShortIoDomain,
    [string]$ShortIoLinkId
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-PreferredValue {
    param(
        [AllowNull()]
        [string]$ExplicitValue,
        [AllowNull()]
        [object]$ConfigValue,
        [AllowNull()]
        [string]$EnvironmentValue,
        [AllowNull()]
        [string]$DefaultValue
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitValue)) {
        return $ExplicitValue
    }

    if ($null -ne $ConfigValue) {
        $configString = [string]$ConfigValue
        if (-not [string]::IsNullOrWhiteSpace($configString)) {
            return $configString
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($EnvironmentValue)) {
        return $EnvironmentValue
    }

    return $DefaultValue
}

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

function Get-CloudflaredCommand {
    $command = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "cloudflared is not installed or not on PATH. Install it first with: winget install Cloudflare.cloudflared"
    }

    return $command.Source
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
            }
        } catch {
            # Ignore stale PIDs.
        }
    }
}

function Start-QuickTunnel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$LocalUrl,
        [Parameter(Mandatory = $true)]
        [string]$CloudflaredPath,
        [Parameter(Mandatory = $true)]
        [string]$LogDir
    )

    $runStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $stdoutPath = Join-Path $LogDir "$Name-$runStamp.log"
    $stderrPath = Join-Path $LogDir "$Name-$runStamp.err.log"

    $process = Start-Process `
        -FilePath $CloudflaredPath `
        -ArgumentList @("tunnel", "--url", $LocalUrl, "--no-autoupdate") `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -PassThru

    $timeout = [DateTime]::UtcNow.AddSeconds(60)
    $urlPattern = "https://[-a-z0-9]+\.trycloudflare\.com"

    while ([DateTime]::UtcNow -lt $timeout) {
        Start-Sleep -Milliseconds 500

        foreach ($path in @($stdoutPath, $stderrPath)) {
            if (-not (Test-Path $path)) {
                continue
            }

            $content = Get-Content $path -Raw
            if ([string]::IsNullOrWhiteSpace($content)) {
                continue
            }

            $match = [regex]::Match($content, $urlPattern)
            if ($match.Success) {
                return [pscustomobject]@{
                    name       = $Name
                    url        = $match.Value
                    pid        = $process.Id
                    stdoutPath = $stdoutPath
                    stderrPath = $stderrPath
                }
            }
        }

        if ($process.HasExited) {
            throw "cloudflared exited before publishing the $Name tunnel URL. Check $stderrPath"
        }
    }

    throw "Timed out waiting for the $Name tunnel URL. Check $stderrPath"
}

function Get-ShortIoHeaders {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ApiKey
    )

    return @{
        "authorization" = $ApiKey
        "accept"        = "application/json"
    }
}

function Resolve-ShortIoId {
    param(
        [string]$ConfiguredLinkId,
        [psobject]$State
    )

    if ($ConfiguredLinkId) {
        return $ConfiguredLinkId
    }

    if ($State -and $State.shortUrl -and $State.shortUrl.linkId) {
        return [string]$State.shortUrl.linkId
    }

    return $null
}

function Set-ShortIoShortUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ApiKey,
        [Parameter(Mandatory = $true)]
        [string]$LongUrl,
        [string]$Domain,
        [string]$Path,
        [string]$LinkId
    )

    $headers = Get-ShortIoHeaders -ApiKey $ApiKey

    if ($LinkId) {
        $updateBody = @{
            originalURL     = $LongUrl
            allowDuplicates = $false
        }
        if ($Domain) {
            $updateBody.domain = $Domain
        }
        if ($Path) {
            $updateBody.path = $Path
        }

        $updated = Invoke-RestMethod `
            -Method Post `
            -Uri "https://api.short.io/links/$([uri]::EscapeDataString($LinkId))" `
            -Headers $headers `
            -ContentType "application/json" `
            -Body ($updateBody | ConvertTo-Json)

        return [pscustomobject]@{
            linkId   = [string]($(if ($updated.idString) { $updated.idString } else { $updated.id }))
            shortUrl = $(if ($updated.secureShortURL) { $updated.secureShortURL } else { $updated.shortURL })
        }
    }

    if (-not $Domain) {
        throw "SHORTIO_DOMAIN is required the first time so the script can create the permanent short URL."
    }

    $createBody = @{
        domain          = $Domain
        originalURL     = $LongUrl
        allowDuplicates = $false
    }
    if ($Path) {
        $createBody.path = $Path
    }

    $created = Invoke-RestMethod `
        -Method Post `
        -Uri "https://api.short.io/links" `
        -Headers $headers `
        -ContentType "application/json" `
        -Body ($createBody | ConvertTo-Json)

    return [pscustomobject]@{
        linkId   = [string]($(if ($created.idString) { $created.idString } else { $created.id }))
        shortUrl = $(if ($created.secureShortURL) { $created.secureShortURL } else { $created.shortURL })
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$stateDir = Join-Path (Join-Path $repoRoot ".local") "phone-tunnels"
$null = New-Item -ItemType Directory -Force -Path $stateDir
$configPath = Join-Path $stateDir "config.psd1"
$statePath = Join-Path $stateDir "state.json"

$config = @{}
if (Test-Path $configPath) {
    $config = Import-PowerShellDataFile -Path $configPath
}

$FrontendLocalUrl = Get-PreferredValue -ExplicitValue $FrontendLocalUrl -ConfigValue $config.FrontendLocalUrl -EnvironmentValue $env:FRONTEND_LOCAL_URL -DefaultValue "http://localhost:3000"
$BackendLocalUrl = Get-PreferredValue -ExplicitValue $BackendLocalUrl -ConfigValue $config.BackendLocalUrl -EnvironmentValue $env:BACKEND_LOCAL_URL -DefaultValue "http://localhost:8000"
$ShortPath = Get-PreferredValue -ExplicitValue $ShortPath -ConfigValue $config.ShortIoPath -EnvironmentValue $env:SHORTIO_PATH -DefaultValue "poker"
$ShortIoApiKey = Get-PreferredValue -ExplicitValue $ShortIoApiKey -ConfigValue $config.ShortIoApiKey -EnvironmentValue $env:SHORTIO_API_KEY -DefaultValue $null
$ShortIoDomain = Get-PreferredValue -ExplicitValue $ShortIoDomain -ConfigValue $config.ShortIoDomain -EnvironmentValue $env:SHORTIO_DOMAIN -DefaultValue $null
$ShortIoLinkId = Get-PreferredValue -ExplicitValue $ShortIoLinkId -ConfigValue $config.ShortIoLinkId -EnvironmentValue $env:SHORTIO_LINK_ID -DefaultValue $null

$existingState = $null
if (Test-Path $statePath) {
    try {
        $existingState = Get-Content $statePath -Raw | ConvertFrom-Json
    } catch {
        $existingState = $null
    }
}

$cloudflaredPath = Get-CloudflaredCommand
Stop-ManagedTunnelProcesses -StatePath $statePath

Write-Host "Starting backend service..." -ForegroundColor Cyan
Invoke-DockerCompose -Arguments @("up", "-d", "backend")

Write-Host "Opening backend Quick Tunnel..." -ForegroundColor Cyan
$backendTunnel = Start-QuickTunnel `
    -Name "backend" `
    -LocalUrl $BackendLocalUrl `
    -CloudflaredPath $cloudflaredPath `
    -LogDir $stateDir

$env:NEXT_PUBLIC_BACKEND_URL = $backendTunnel.url

Write-Host "Starting frontend service with NEXT_PUBLIC_BACKEND_URL=$($backendTunnel.url)" -ForegroundColor Cyan
Invoke-DockerCompose -Arguments @("up", "-d", "--force-recreate", "frontend")

Write-Host "Opening frontend Quick Tunnel..." -ForegroundColor Cyan
$frontendTunnel = Start-QuickTunnel `
    -Name "frontend" `
    -LocalUrl $FrontendLocalUrl `
    -CloudflaredPath $cloudflaredPath `
    -LogDir $stateDir

$shortUrl = $null
$shortUrlState = $null
$resolvedShortIoLinkId = Resolve-ShortIoId -ConfiguredLinkId $ShortIoLinkId -State $existingState
$resolvedShortIoDomain = $(if ($ShortIoDomain) { $ShortIoDomain } elseif ($existingState -and $existingState.shortUrl -and $existingState.shortUrl.domain) { [string]$existingState.shortUrl.domain } else { $null })
$resolvedShortPath = $(if ($ShortPath) { $ShortPath } elseif ($existingState -and $existingState.shortUrl -and $existingState.shortUrl.path) { [string]$existingState.shortUrl.path } else { $null })
if ($ShortIoApiKey) {
    Write-Host "Updating Short.io short URL..." -ForegroundColor Cyan
    $shortUrlState = Set-ShortIoShortUrl `
        -ApiKey $ShortIoApiKey `
        -LongUrl $frontendTunnel.url `
        -Domain $resolvedShortIoDomain `
        -Path $resolvedShortPath `
        -LinkId $resolvedShortIoLinkId
    $shortUrl = $shortUrlState.shortUrl
}

$state = [pscustomobject]@{
    updatedAt = (Get-Date).ToString("o")
    backend   = $backendTunnel
    frontend  = $frontendTunnel
    shortUrl  = $(if ($shortUrlState) {
        [pscustomobject]@{
            provider = "short.io"
            linkId   = $shortUrlState.linkId
            url      = $shortUrlState.shortUrl
            path     = $resolvedShortPath
            domain   = $resolvedShortIoDomain
        }
    } else {
        $existingState.shortUrl
    })
}
$state | ConvertTo-Json -Depth 6 | Set-Content -Path $statePath

Write-Host ""
Write-Host "Backend tunnel:  $($backendTunnel.url)" -ForegroundColor Green
Write-Host "Frontend tunnel: $($frontendTunnel.url)" -ForegroundColor Green
if ($shortUrl) {
    Write-Host "Short URL:       $shortUrl" -ForegroundColor Green
} else {
    Write-Host "Short URL:       not updated (set SHORTIO_API_KEY and SHORTIO_DOMAIN for the first run)" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Tunnel state saved to $statePath" -ForegroundColor DarkGray
