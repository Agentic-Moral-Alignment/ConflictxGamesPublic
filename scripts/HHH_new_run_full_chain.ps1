param(
    [int] $PollSeconds = 60,
    [switch] $SkipGame,
    [switch] $SkipSolo
)

# Detached driver for the 2026-08-27 resume.
#
# Runs in ONE process so it survives the shell that launched it, and so the
# three gateway keys are never double-booked. Unlike
# HHH_new_run_game_eval_sequence.ps1 (which finishes every mode of one model
# before starting the next), this schedules by KEY, so no key sits idle while
# Qwen's ~11-hour play run occupies key 3:
#
#   WAVE 1   key 3  qwen-play (adopts the already-running process if present)
#            key 1  gptoss-mcq
#            key 2  gptoss-open
#   WAVE 2   key 3  gptoss-play
#            keys 1-2  solo baseline + steering, once the top-up has exited and
#                      all 3 value pairs are at their 50-kept quota
#
# Gemma (all modes) and Qwen mcq/open are already complete; they are only
# re-verified by case count here, never re-run.
#
# Launch detached -- NOT from an interactive background shell, whose job object
# kills the orchestrator when that shell exits:
#   schtasks /Run /TN "HHH_full_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash   = "C:\Program Files\Git\bin\bash.exe"
$pipeline  = "scripts/HHH_new_pipeline.sh"
$ps        = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$gameDir   = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$soloDir   = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"
$soloCsv   = Join-Path $repoRoot "data/HHH_new_solo_20260826_01/claude-sonnet-5.csv"
$chainLog  = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$soloLog   = Join-Path $soloDir "solo_extended_sequence.stdout.log"
$expected  = 150

New-Item -ItemType Directory -Path $gameDir -Force | Out-Null
New-Item -ItemType Directory -Path $soloDir -Force | Out-Null

function Write-Chain([string] $Message) {
    # Add-Content fails outright if anything else holds the file -- Dropbox
    # sync touches this folder constantly, and PowerShell's own Get-Content
    # opens with FileShare.Read, which denies writers. Append through a
    # FileStream that shares read+write, and retry briefly on a transient lock,
    # so a reader can never cost the chain a log line.
    $line = "$(Get-Date -Format s) [chain] $Message"
    for ($attempt = 0; $attempt -lt 25; $attempt++) {
        try {
            $stream = [System.IO.File]::Open($chainLog, [System.IO.FileMode]::Append,
                                             [System.IO.FileAccess]::Write,
                                             [System.IO.FileShare]::ReadWrite)
            try {
                $writer = New-Object System.IO.StreamWriter($stream, [System.Text.UTF8Encoding]::new($false))
                $writer.WriteLine($line)
                $writer.Flush()
                $writer.Dispose()
            }
            finally { $stream.Dispose() }
            return
        }
        catch { Start-Sleep -Milliseconds 200 }
    }
}

function Get-ResultCaseCount([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    try {
        return @(Import-Csv -LiteralPath $Path |
            ForEach-Object { "$($_.scenario_id)|$($_.seat)" } |
            Select-Object -Unique).Count
    }
    catch { return 0 }
}

# Any live evaluate_games process for this config fragment, or $null.
function Get-LiveEval([string] $ConfigMatch) {
    return @(
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match 'conflictxgame\.games\.evaluate_games' -and $_.CommandLine -match $ConfigMatch }
    ) | Select-Object -First 1
}

function Start-Eval($Run) {
    $prior = $env:GATEWAY_KEY_INDEX
    try {
        $env:GATEWAY_KEY_INDEX = [string]$Run.Key
        $startArgs = @{
            FilePath = $gitBash
            ArgumentList = @($pipeline, $Run.Command)
            WorkingDirectory = $repoRoot
            RedirectStandardOutput = (Join-Path $gameDir "$($Run.Log).stdout.log")
            RedirectStandardError = (Join-Path $gameDir "$($Run.Log).stderr.log")
            WindowStyle = "Hidden"
            PassThru = $true
        }
        $p = Start-Process @startArgs
        Write-Chain "launched $($Run.Command) PID $($p.Id) on key $($Run.Key)."
        return $p
    }
    finally {
        if ($null -eq $prior) { Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue }
        else { $env:GATEWAY_KEY_INDEX = $prior }
    }
}

# Wait for an already-running eval (adopted from a previous run) to finish.
function Wait-LiveEval([string] $ConfigMatch, [string] $Label) {
    $waited = 0
    while ($true) {
        $live = Get-LiveEval $ConfigMatch
        if ($null -eq $live) { break }
        if (($waited % 1800) -eq 0) {
            Write-Chain "waiting on adopted $Label (PID $($live.ProcessId)), $([int]($waited / 60)) min elapsed."
        }
        Start-Sleep -Seconds $PollSeconds
        $waited += $PollSeconds
    }
}

# Run one wave of jobs in parallel, then verify each by unique scenario|seat count.
function Invoke-Wave([string] $Label, $Runs) {
    $names = ($Runs | ForEach-Object { "$($_.Command)(k$($_.Key))" }) -join ", "
    Write-Chain "WAVE '$Label' start: $names"
    $jobs = @()
    foreach ($run in $Runs) {
        $done = Get-ResultCaseCount (Join-Path $gameDir $run.Output)
        if ($done -ge $expected) {
            Write-Chain "$($run.Command) already complete ($done/$expected cases); skipping."
            continue
        }
        $live = Get-LiveEval $run.Match
        if ($null -ne $live) {
            Write-Chain "$($run.Command) already running as PID $($live.ProcessId); adopting it instead of relaunching."
            $jobs += [pscustomobject]@{ Run = $run; Process = $null; Adopted = $true }
            continue
        }
        $jobs += [pscustomobject]@{ Run = $run; Process = (Start-Eval $run); Adopted = $false }
    }

    foreach ($job in $jobs) {
        if ($job.Adopted) {
            Wait-LiveEval $job.Run.Match $job.Run.Command
        }
        else {
            $job.Process.WaitForExit()
            # ExitCode reads back blank on PS 5.1 for fast-exiting Start-Process
            # jobs even after WaitForExit(); log it, never throw on it. The case
            # count below is the real pass/fail signal.
            $code = $job.Process.ExitCode
            if ($code -isnot [int] -or $code -ne 0) {
                Write-Chain "$($job.Run.Command) reported exit code '$code' (unreliable when a job exits fast); verifying by case count instead."
            }
        }
        $n = Get-ResultCaseCount (Join-Path $gameDir $job.Run.Output)
        if ($n -lt $expected -and $job.Adopted) {
            Write-Chain "$($job.Run.Command) adopted process ended with only $n/$expected cases; relaunching once."
            $p = Start-Eval $job.Run
            $p.WaitForExit()
            $n = Get-ResultCaseCount (Join-Path $gameDir $job.Run.Output)
        }
        if ($n -lt $expected) {
            Write-Chain "FAIL $($job.Run.Command): $n/$expected cases in $($job.Run.Output)"
        }
        else {
            Write-Chain "OK   $($job.Run.Command): $n/$expected cases."
        }
    }
    Write-Chain "WAVE '$Label' done."
}

Write-Chain "=== chain start (PID $PID) ==="

$QWEN_PLAY = @{ Command = "game-eval-qwen-play"; Key = 3; Log = "qwen3.5-122b_play_balanced50";
                Match = "eval-play_focal-qwen3\.5-122b";
                Output = "qwen3.5-122b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv" }
$OSS_MCQ   = @{ Command = "game-eval-gptoss-mcq"; Key = 1; Log = "gpt-oss-120b_mcq_balanced50";
                Match = "eval-mcq_focal-gpt-oss-120b";
                Output = "gpt-oss-120b_mcq_balanced50.csv" }
$OSS_OPEN  = @{ Command = "game-eval-gptoss-open"; Key = 2; Log = "gpt-oss-120b_open_balanced50";
                Match = "eval-open_focal-gpt-oss-120b";
                Output = "gpt-oss-120b_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv" }
$OSS_PLAY  = @{ Command = "game-eval-gptoss-play"; Key = 3; Log = "gpt-oss-120b_play_balanced50";
                Match = "eval-play_focal-gpt-oss-120b";
                Output = "gpt-oss-120b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv" }

$ossPlayJob = $null

if (-not $SkipGame) {
    Invoke-Wave "1 - qwen play + gptoss mcq/open" @($QWEN_PLAY, $OSS_MCQ, $OSS_OPEN)
    # gptoss play takes key 3 on its own; keys 1-2 go to the solo evals below.
    if ((Get-ResultCaseCount (Join-Path $gameDir $OSS_PLAY.Output)) -lt $expected) {
        $live = Get-LiveEval $OSS_PLAY.Match
        if ($null -eq $live) {
            $ossPlayJob = Start-Eval $OSS_PLAY
            Write-Chain "WAVE 2: gptoss-play running on key 3 while the solo evals take keys 1-2."
        }
    }
}

if (-not $SkipSolo) {
    Write-Chain "SOLO gate: waiting for the top-up controller to exit."
    $waited = 0
    while ($true) {
        $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                     Where-Object { $_.CommandLine -match 'conflictxgame\.solo\.topup_scenarios' })
        if ($running.Count -eq 0) { break }
        if (($waited % 1800) -eq 0) {
            Write-Chain "SOLO gate: top-up PID $($running[0].ProcessId) still running ($([int]($waited / 60)) min)."
        }
        Start-Sleep -Seconds $PollSeconds
        $waited += $PollSeconds
    }

    $groups = @(Import-Csv -LiteralPath $soloCsv |
                Where-Object { $_.keep_scenario -eq "True" } |
                Group-Object { (@($_.value1, $_.value2) | Sort-Object) -join "|" })
    $summary = ($groups | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ", "
    Write-Chain "SOLO gate: kept quotas = $summary"

    if ($groups.Count -ne 3 -or @($groups | Where-Object { $_.Count -lt 50 }).Count -gt 0) {
        Write-Chain "SOLO SKIPPED: a pair is under its 50-kept quota. Re-run 'bash scripts/HHH_new_pipeline.sh solo-topup-to-50', then re-run this chain with -SkipGame."
    }
    else {
        Write-Chain "SOLO start: baseline + steering for all 3 models on keys 1-2."
        & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repoRoot "scripts\HHH_new_run_solo_extended_eval_sequence.ps1") *>> $soloLog
        Write-Chain "SOLO returned (native exit code '$LASTEXITCODE'); see solo_extended_sequence.stdout.log."
    }
}

if ($null -ne $ossPlayJob) {
    Write-Chain "waiting on gptoss-play to finish."
    $ossPlayJob.WaitForExit()
    $n = Get-ResultCaseCount (Join-Path $gameDir $OSS_PLAY.Output)
    if ($n -lt $expected) {
        Write-Chain "FAIL game-eval-gptoss-play: $n/$expected cases."
    }
    else {
        Write-Chain "OK   game-eval-gptoss-play: $n/$expected cases."
    }
}

Write-Chain "=== chain complete ==="
