param(
    [string]$RepoRoot = "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo",
    [string]$EnvFile = "C:\forex-algo-bot\config\.env",
    [ValidateSet("trend", "smoke")]
    [string]$Strategy = "trend",
    [int]$MaxCycles = 0,
    [double]$MaxSeconds = 0,
    [switch]$ResetState
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path

Push-Location $RepoRoot
try {
    $arguments = @(
        "-m",
        "fxbot.integration.live_cli",
        "--env-file",
        $EnvFile,
        "--strategy",
        $Strategy
    )

    if ($MaxCycles -gt 0) {
        $arguments += @("--max-cycles", [string]$MaxCycles)
    }
    if ($MaxSeconds -gt 0) {
        $arguments += @("--max-seconds", [string]$MaxSeconds)
    }
    if ($ResetState) {
        $arguments += "--reset-state"
    }

    & python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Paper live-feed service exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
