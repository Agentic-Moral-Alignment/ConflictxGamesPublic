param(
    [string] $GameProcessIds = "41048,3752,48772",
    [int] $PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gameResultDir = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01"
$gameCoveragePath = Join-Path $repoRoot "data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/games/pair_coverage.json"
$soloDataDir = Join-Path $repoRoot "data/HHH_new_solo_20260826_01"
$soloScenarioDir = Join-Path $soloDataDir "scenarios"
$soloResultDir = Join-Path $repoRoot "results/HHH_new_solo_20260826_01"
$pipeline = Join-Path $repoRoot "scripts/HHH_new_pipeline.sh"
$gitBash = "C:\Program Files\Git\bin\bash.exe"

if (-not (Test-Path -LiteralPath $gitBash)) {
    throw "Git Bash was not found at $gitBash"
}
New-Item -ItemType Directory -Path $soloResultDir -Force | Out-Null

$gameOutputs = @(
    "gemma-4-31B-it_mcq.csv",
    "gemma-4-31B-it_open_user-gpt-5.6-luna_judge-gpt-5.6-luna.csv",
    "gemma-4-31B-it_play_scripted_r2_rounds5_mem1_fallback-gpt-5.6-luna.csv"
) | ForEach-Object { Join-Path $gameResultDir $_ }

$gamePids = @(
    $GameProcessIds.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ } |
        ForEach-Object { [int] $_ }
)

function Write-QueueStatus([string] $Message) {
    Write-Output "$(Get-Date -Format s) $Message"
}

function Get-SoloCounts {
    $counts = @{}
    Get-ChildItem -LiteralPath $soloScenarioDir -Filter "*.json" |
        ForEach-Object {
            $parsed = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
            $counts[$_.Name] = @($parsed.PSObject.Properties).Count
        }
    return $counts
}

Write-QueueStatus "Waiting for game evaluation PIDs: $($gamePids -join ', ')"
while (@($gamePids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }).Count -gt 0) {
    Start-Sleep -Seconds $PollSeconds
}

$missingGameOutputs = @($gameOutputs | Where-Object {
    -not (Test-Path -LiteralPath $_) -or (Get-Item -LiteralPath $_).Length -eq 0
})
if ($missingGameOutputs.Count -gt 0) {
    throw "Game evaluation process ended without output: $($missingGameOutputs -join ', ')"
}
Write-QueueStatus "All three game evaluation outputs are complete."

# The running game evals loaded the corpus at process start. Generation may
# accept additional games while they run, so wait for the final quota and
# then rerun all three modes once; their fixed output files make this a
# resumable top-up that skips scenario IDs already evaluated.
while ($true) {
    $coverage = Get-Content -LiteralPath $gameCoveragePath -Raw | ConvertFrom-Json
    $coverageStatus = ($coverage.pairs | ForEach-Object {
        "$($_.pair)=$($_.kept)/$($_.target)"
    }) -join ", "
    if ($coverage.complete) {
        Write-QueueStatus "Game generation quota is complete: $coverageStatus"
        break
    }
    Write-QueueStatus "Waiting for final game-generation quota: $coverageStatus"
    Start-Sleep -Seconds $PollSeconds
}

Write-QueueStatus "Starting resumable game-evaluation top-ups on the final corpus."
$priorIndex = $env:GATEWAY_KEY_INDEX
try {
    $topUpJobs = @()
    foreach ($topUp in @(
        @{ Command = "game-eval-mcq"; Key = "1"; Label = "mcq" },
        @{ Command = "game-eval-open"; Key = "2"; Label = "open" },
        @{ Command = "game-eval-play"; Key = "3"; Label = "play" }
    )) {
        $env:GATEWAY_KEY_INDEX = $topUp.Key
        $start = @{
            FilePath = $gitBash
            ArgumentList = @($pipeline, $topUp.Command)
            WorkingDirectory = $repoRoot
            RedirectStandardOutput = Join-Path $gameResultDir "gemma-4-31B-it_$($topUp.Label)_topup.stdout.log"
            RedirectStandardError = Join-Path $gameResultDir "gemma-4-31B-it_$($topUp.Label)_topup.stderr.log"
            WindowStyle = "Hidden"
            PassThru = $true
        }
        $topUpJobs += Start-Process @start
    }
}
finally {
    if ($null -eq $priorIndex) {
        Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue
    }
    else {
        $env:GATEWAY_KEY_INDEX = $priorIndex
    }
}

while (@($topUpJobs | Where-Object {
    Get-Process -Id $_.Id -ErrorAction SilentlyContinue
}).Count -gt 0) {
    Start-Sleep -Seconds $PollSeconds
}
Write-QueueStatus "Final-corpus game-evaluation top-ups are complete."

while ($true) {
    $soloCounts = Get-SoloCounts
    $shortCounts = ($soloCounts.GetEnumerator() | Sort-Object Name |
        ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ", "
    if ($soloCounts.Count -eq 3 -and @($soloCounts.Values | Where-Object { $_ -lt 50 }).Count -eq 0) {
        Write-QueueStatus "Solo generation is complete: $shortCounts"
        break
    }
    if (-not (Get-Process -Id 12720 -ErrorAction SilentlyContinue)) {
        throw "Solo generation stopped before 50/50/50: $shortCounts"
    }
    Write-QueueStatus "Still waiting for solo 50/50/50: $shortCounts"
    Start-Sleep -Seconds $PollSeconds
}

Write-QueueStatus "Starting resumable solo filtering with GPT-5.6 Luna."
& $gitBash $pipeline "solo-filter"
if ($LASTEXITCODE -ne 0) {
    throw "Solo filtering failed with exit code $LASTEXITCODE"
}

$filteredCsv = Join-Path $soloDataDir "claude-sonnet-5.csv"
if (-not (Test-Path -LiteralPath $filteredCsv) -or (Get-Item -LiteralPath $filteredCsv).Length -eq 0) {
    throw "Solo filtering completed without creating $filteredCsv"
}

$priorIndex = $env:GATEWAY_KEY_INDEX
try {
    $env:GATEWAY_KEY_INDEX = "2"
    $mcqStart = @{
        FilePath = $gitBash
        ArgumentList = @($pipeline, "solo-eval-mcq")
        WorkingDirectory = $repoRoot
        RedirectStandardOutput = Join-Path $soloResultDir "gemma-4-31B-it_mcq_eval.stdout.log"
        RedirectStandardError = Join-Path $soloResultDir "gemma-4-31B-it_mcq_eval.stderr.log"
        WindowStyle = "Hidden"
        PassThru = $true
    }
    $mcq = Start-Process @mcqStart

    $env:GATEWAY_KEY_INDEX = "3"
    $interactiveStart = @{
        FilePath = $gitBash
        ArgumentList = @($pipeline, "solo-eval-interactive")
        WorkingDirectory = $repoRoot
        RedirectStandardOutput = Join-Path $soloResultDir "gemma-4-31B-it_interactive_eval.stdout.log"
        RedirectStandardError = Join-Path $soloResultDir "gemma-4-31B-it_interactive_eval.stderr.log"
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

Write-QueueStatus "Started solo MCQ PID $($mcq.Id) on gateway key 2."
Write-QueueStatus "Started solo interactive PID $($interactive.Id) on gateway key 3."
