param(
    [string]$RepoRoot = "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo",
    [string]$EnvFile = "",
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $RepoRoot).Path

if ($EnvFile) {
    $EnvFile = (Resolve-Path $EnvFile).Path

    foreach ($RawLine in Get-Content -LiteralPath $EnvFile) {
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith("#")) {
            continue
        }

        $Parts = $Line.Split("=", 2)
        if ($Parts.Count -ne 2) {
            throw "Invalid environment line: $RawLine"
        }

        $Name = $Parts[0].Trim()
        $Value = $Parts[1].Trim().Trim('"').Trim("'")
        if (-not $Name) {
            throw "Environment variable name cannot be empty."
        }

        Set-Item -Path "Env:$Name" -Value $Value
    }
}

Push-Location $RepoRoot
try {
    & $PythonCommand -m fxbot.production.bootstrap
    if ($LASTEXITCODE -ne 0) {
        throw "Production preflight failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
