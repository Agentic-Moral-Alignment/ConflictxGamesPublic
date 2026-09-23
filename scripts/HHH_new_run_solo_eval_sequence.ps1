param(
    [int] $RecoveryProcessId = 39356,
    [int] $FilterProcessId = 35816,
    [int] $PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$scenarioDir = Join-Path $repoRoot "data/HHH_new_solo_20260826_01/scenarios"
$soloDataDir = Join-Path $repoRoot "data/HHH_new_solo_20260826_01"
$resultDir = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"

if (-not (Test-Path -LiteralPath $gitBash)) {
    throw "Git Bash was not found at $gitBash"
}
New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

function Write-Status([string] $Message) {
    Write-Output "$(Get-Date -Format s) $Message"
}

function Wait-ForProcess([int] $ProcessId, [string] $Label) {
    if ($ProcessId -le 0) { return }
    while (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue) {
        Write-Status "Waiting for $Label PID $ProcessId."
        Start-Sleep -Seconds $PollSeconds
    }
}

function Get-ScenarioCounts {
    $counts = @{}
    Get-ChildItem -LiteralPath $scenarioDir -File -Filter "*.json" |
        ForEach-Object {
            $parsed = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
            $counts[$_.Name] = @($parsed.PSObject.Properties).Count
        }
    return $counts
}

Wait-ForProcess $RecoveryProcessId "solo recovery"
$counts = Get-ScenarioCounts
$shortCounts = ($counts.GetEnumerator() | Sort-Object Name |
    ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "
if ($counts.Count -ne 3 -or @($counts.Values | Where-Object { $_ -lt 50 }).Count -gt 0) {
    throw "Solo recovery ended before every pair reached at least 50: $shortCounts"
}
Write-Status "Solo recovery complete: $shortCounts"

Wait-ForProcess $FilterProcessId "initial solo filter"
Write-Status "Running resumable filter top-up for any recovered scenarios."
& $gitBash $pipeline "solo-filter"
if ($LASTEXITCODE -ne 0) {
    throw "Solo filter top-up failed with exit code $LASTEXITCODE"
}

$filteredCsv = Join-Path $soloDataDir "claude-sonnet-5.csv"
if (-not (Test-Path -LiteralPath $filteredCsv)) {
    throw "Solo filtering did not create $filteredCsv"
}
$expectedRows = ($counts.Values | Measure-Object -Sum).Sum
$actualRows = @(Import-Csv -LiteralPath $filteredCsv).Count
if ($actualRows -ne $expectedRows) {
    throw "Filtered CSV has $actualRows rows; expected $expectedRows from the scenario JSON files."
}
Write-Status "Solo filter corpus complete with $actualRows judged rows."
$keptGroups = @(
    Import-Csv -LiteralPath $filteredCsv |
        Where-Object { $_.keep_scenario -eq "True" } |
        Group-Object { (@($_.value1, $_.value2) | Sort-Object) -join "|" }
)
if ($keptGroups.Count -ne 3 -or @($keptGroups | Where-Object { $_.Count -lt 50 }).Count -gt 0) {
    $keptSummary = ($keptGroups | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ", "
    throw "Solo kept quotas are incomplete; refusing to evaluate: $keptSummary"
}
Write-Status "All three solo value pairs have at least 50 kept scenarios."

$priorIndex = $env:GATEWAY_KEY_INDEX
try {
    $env:GATEWAY_KEY_INDEX = "2"
    $mcqStart = @{
        FilePath = $gitBash
        ArgumentList = @($pipeline, "solo-eval-mcq")
        WorkingDirectory = $repoRoot
        RedirectStandardOutput = Join-Path $resultDir "gemma-4-31B-it_mcq_balanced50_eval.stdout.log"
        RedirectStandardError = Join-Path $resultDir "gemma-4-31B-it_mcq_balanced50_eval.stderr.log"
        WindowStyle = "Hidden"
        PassThru = $true
    }
    $mcq = Start-Process @mcqStart

    $env:GATEWAY_KEY_INDEX = "3"
    $interactiveStart = @{
        FilePath = $gitBash
        ArgumentList = @($pipeline, "solo-eval-interactive")
        WorkingDirectory = $repoRoot
        RedirectStandardOutput = Join-Path $resultDir "gemma-4-31B-it_interactive_balanced50_eval.stdout.log"
        RedirectStandardError = Join-Path $resultDir "gemma-4-31B-it_interactive_balanced50_eval.stderr.log"
        WindowStyle = "Hidden"
        PassThru = $true
    }
    $interactive = Start-Process @interactiveStart
}
finally {
    if ($null -eq $priorIndex) {
        Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue
    }
    else {
        $env:GATEWAY_KEY_INDEX = $priorIndex
    }
}

Write-Status "Started Gemma solo MCQ PID $($mcq.Id) on gateway key 2."
Write-Status "Started Gemma solo interactive PID $($interactive.Id) on gateway key 3."
while (-not ($mcq.HasExited -and $interactive.HasExited)) {
    Start-Sleep -Seconds $PollSeconds
    $mcq.Refresh()
    $interactive.Refresh()
}
$mcq.WaitForExit()
$interactive.WaitForExit()
if ($mcq.ExitCode -ne 0 -or $interactive.ExitCode -ne 0) {
    throw "Gemma solo eval failure: MCQ=$($mcq.ExitCode), interactive=$($interactive.ExitCode)"
}

Write-Status "Gemma solo evaluations complete; refreshing the shared solo analysis."
& $gitBash $pipeline "solo-analyze"
if ($LASTEXITCODE -ne 0) {
    throw "Solo analysis refresh failed with exit code $LASTEXITCODE"
}
Write-Status "Gemma solo evaluation and analysis sequence complete."
