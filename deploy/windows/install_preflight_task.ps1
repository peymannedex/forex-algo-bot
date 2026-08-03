param(
    [string]$TaskName = "ForexAlgoBot-Preflight",
    [string]$RepoRoot = "C:\Users\peyma\Desktop\forex-algo-bot\temp_repo",
    [string]$EnvFile = "C:\forex-algo-bot\config\.env",
    [string]$RunScript = "C:\forex-algo-bot\deploy\windows\run_preflight.ps1"
)

$ErrorActionPreference = "Stop"

$PowerShell = (Get-Command powershell.exe).Source
$Arguments = @(
    "-NoProfile"
    "-ExecutionPolicy", "Bypass"
    "-File", "`"$RunScript`""
    "-RepoRoot", "`"$RepoRoot`""
    "-EnvFile", "`"$EnvFile`""
) -join " "

$Action = New-ScheduledTaskAction `
    -Execute $PowerShell `
    -Argument $Arguments

$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel Highest

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Force

Write-Host "Installed scheduled task: $TaskName" -ForegroundColor Green
Write-Host "This task runs the readiness preflight only; it does not start live trading."
