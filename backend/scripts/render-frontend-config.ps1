$ErrorActionPreference = "Stop"

$apiBaseUrl = $env:FRONTEND_API_BASE_URL
if ([string]::IsNullOrWhiteSpace($apiBaseUrl)) {
    throw "Set FRONTEND_API_BASE_URL to the public API origin before rendering frontend configuration."
}

$parsedUrl = $null
if (-not [Uri]::TryCreate($apiBaseUrl, [UriKind]::Absolute, [ref]$parsedUrl) -or
    $parsedUrl.Scheme -notin @("http", "https") -or
    $parsedUrl.UserInfo -or
    $parsedUrl.AbsolutePath -ne "/" -or
    $parsedUrl.Query -or
    $parsedUrl.Fragment) {
    throw "FRONTEND_API_BASE_URL must be an HTTP(S) origin without credentials, path, query, or fragment."
}

if ($env:FRONTEND_ENV -eq "production" -and
    ($parsedUrl.Scheme -ne "https" -or $parsedUrl.IsLoopback -or $parsedUrl.Host.EndsWith(".localhost"))) {
    throw "Production frontend configuration must use the configured public HTTPS backend origin."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$configDirectory = Join-Path $projectRoot "apps"
$configPath = Join-Path $configDirectory "carworld-config.js"
$publicOrigin = $parsedUrl.GetLeftPart([UriPartial]::Authority)
$content = "/* Generated from FRONTEND_API_BASE_URL; contains no secret. */`nwindow.CARWORLD_API_BASE_URL = '$publicOrigin';`n"
Set-Content -LiteralPath $configPath -Value $content -Encoding utf8
Write-Host "Wrote public frontend API origin to $configPath"
