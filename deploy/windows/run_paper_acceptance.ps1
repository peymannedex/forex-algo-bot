param(
    [string]$RepoRoot = "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo",
    [string]$EnvFile = "C:\forex-algo-bot\config\.env",
    [string]$ReplayCsv = "C:\forex-algo-bot\data\paper_replay.csv",
    [int]$M5Bars = 360,
    [switch]$KeepState
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path

if (-not (Test-Path $EnvFile)) {
    throw "Paper environment file not found: $EnvFile"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReplayCsv) |
    Out-Null

Push-Location $RepoRoot
try {
    python scripts/generate_paper_replay.py `
        --output $ReplayCsv `
        --m5-bars $M5Bars
    if ($LASTEXITCODE -ne 0) {
        throw "Replay generation failed."
    }

    $Arguments = @(
        "-m",
        "fxbot.integration.cli",
        "--env-file",
        $EnvFile,
        "--replay-csv",
        $ReplayCsv,
        "--strategy",
        "smoke"
    )
    if (-not $KeepState) {
        $Arguments += "--reset-state"
    }

    python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Paper integration acceptance replay failed."
    }
}
finally {
    Pop-Location
}
