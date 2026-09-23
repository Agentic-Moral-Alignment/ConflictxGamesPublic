param(
    [int] $PollSeconds = 60,
    [int] $Key = 1,
    # Wait for this task to stop before starting. The steering sweep uses the
    # same key, so the two must not overlap.
    [string] $WaitForTask = 'HHH_game_steer_chain',
    [switch] $IncludeQwen,
    # Run Qwen alongside its own play evaluation instead of after it.
    # They share the qwen3.5-122b backend, so both get slower -- but with the
    # other keys idle that is still better total throughput than serialising.
    [switch] $NoWaitForPlay
)

# Open-ended REPEATED game runs (rounds: 5, memory_k: 1), each model with and
# without the HHH steering prompt, so the steering effect can be read round by
# round rather than only at a single exchange.
#
# This uses the repeated-round capability: `--mode open` with
# `--rounds >= 2` routes to evaluate_open_repeated, which judges every round
# independently and gives the target a genuine multi-turn message array capped
# at memory_k turns, while the simulated user always sees full history.
#
# Qwen runs last and only with -IncludeQwen, because its runs hit the same
# backend model as its play evaluation and would otherwise contend with it.
#
# Launch detached:  schtasks /Run /TN "HHH_game_repeated_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150     # unique scenario|seat cases; each carries 5 rounds of rows

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [repeat] $Message"
    for ($attempt = 0; $attempt -lt 25; $attempt++) {
        try {
            $stream = [System.IO.File]::Open($chainLog, [System.IO.FileMode]::Append,
                                             [System.IO.FileAccess]::Write,
                                             [System.IO.FileShare]::ReadWrite)
            try {
                $writer = New-Object System.IO.StreamWriter($stream, [System.Text.UTF8Encoding]::new($false))
                $writer.WriteLine($line); $writer.Flush(); $writer.Dispose()
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

function Get-LiveEval([string] $ConfigMatch) {
    return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match 'conflictxgame\.games\.evaluate_games' -and
                            $_.CommandLine -match $ConfigMatch }) | Select-Object -First 1
}

function Invoke-Run($Run) {
    $outPath = Join-Path $gameDir $Run.Output
    $done = Get-ResultCaseCount $outPath
    if ($done -ge $expected) {
        Write-Chain "$($Run.Command) already complete ($done/$expected cases); skipping."
        return
    }
    $live = Get-LiveEval $Run.Match
    if ($null -ne $live) {
        Write-Chain "$($Run.Command) already running as PID $($live.ProcessId); waiting on it."
        while ($null -ne (Get-LiveEval $Run.Match)) { Start-Sleep -Seconds $PollSeconds }
    }
    else {
        $prior = $env:GATEWAY_KEY_INDEX
        try {
            $env:GATEWAY_KEY_INDEX = [string]$Key
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
            Write-Chain "launched $($Run.Command) PID $($p.Id) on key $Key."
            $p.WaitForExit()
        }
        finally {
            if ($null -eq $prior) { Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue }
            else { $env:GATEWAY_KEY_INDEX = $prior }
        }
    }
    $n = Get-ResultCaseCount $outPath
    if ($n -lt $expected) { Write-Chain "FAIL $($Run.Command): $n/$expected cases in $($Run.Output)" }
    else { Write-Chain "OK   $($Run.Command): $n/$expected cases." }
}

$OPEN = '_open_balanced50'
$TAIL = '_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv'
$REP  = '_repeated_rounds5_mem1'

$FAST = @(
    @{ Command = 'game-eval-open-repeated'; Log = 'gemma-4-31B-it_open_repeated';
       Match = 'eval-open-repeated_focal-gemma';
       Output = 'gemma-4-31B-it' + $OPEN + $REP + $TAIL },
    @{ Command = 'game-eval-open-repeated-steer'; Log = 'gemma-4-31B-it_open_repeated_steer';
       Match = 'eval-open-repeated-steer_focal-gemma';
       Output = 'gemma-4-31B-it' + $OPEN + '_steer-HHH' + $REP + $TAIL },
    @{ Command = 'game-eval-gptoss-open-repeated'; Log = 'gpt-oss-120b_open_repeated';
       Match = 'eval-open-repeated_focal-gpt-oss';
       Output = 'gpt-oss-120b' + $OPEN + $REP + $TAIL },
    @{ Command = 'game-eval-gptoss-open-repeated-steer'; Log = 'gpt-oss-120b_open_repeated_steer';
       Match = 'eval-open-repeated-steer_focal-gpt-oss';
       Output = 'gpt-oss-120b' + $OPEN + '_steer-HHH' + $REP + $TAIL }
)
$QWEN = @(
    @{ Command = 'game-eval-qwen-open-repeated'; Log = 'qwen3.5-122b_open_repeated';
       Match = 'eval-open-repeated_focal-qwen';
       Output = 'qwen3.5-122b' + $OPEN + $REP + $TAIL },
    @{ Command = 'game-eval-qwen-open-repeated-steer'; Log = 'qwen3.5-122b_open_repeated_steer';
       Match = 'eval-open-repeated-steer_focal-qwen';
       Output = 'qwen3.5-122b' + $OPEN + '_steer-HHH' + $REP + $TAIL }
)

function Wait-ForTask([string] $Name, [string] $Why) {
    if (-not $Name) { return }
    $waited = 0
    while ($true) {
        $state = (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue).State
        if ($null -eq $state -or $state -ne 'Running') { break }
        if (($waited % 1800) -eq 0) {
            Write-Chain "waiting for $Name $Why ($([int]($waited / 60)) min)."
        }
        Start-Sleep -Seconds $PollSeconds
        $waited += $PollSeconds
    }
    Write-Chain "$Name is clear."
}

Write-Chain "=== open-repeated sweep start (PID $PID, key $Key) ==="

# The fast models do NOT wait on the steering sweep. That sweep spends most of
# its life idle, blocked on Qwen's play run, and waiting on it would have parked
# Gemma and GPT-OSS behind hours of unrelated Qwen work with key $Key empty.
# Only the Qwen phase below needs to serialise against it.
foreach ($run in $FAST) { Invoke-Run $run }
Write-Chain "fast models done (Gemma, GPT-OSS): open-repeated with and without steering."

if ($IncludeQwen) {
    Wait-ForTask $WaitForTask 'to release its Qwen phase'
    if ($NoWaitForPlay) {
        Write-Chain 'running Qwen alongside its play evaluation by request; both will be slower.'
    }
    else {
    Write-Chain "waiting for Qwen's play run to clear before its repeated runs (same backend model)."
        $waited = 0
        while ($null -ne (Get-LiveEval 'eval-play_focal-qwen3\.5-122b')) {
            if (($waited % 1800) -eq 0) { Write-Chain "still waiting on Qwen play ($([int]($waited / 60)) min)." }
            Start-Sleep -Seconds $PollSeconds
            $waited += $PollSeconds
        }
    }
    foreach ($run in $QWEN) { Invoke-Run $run }
}
else {
    Write-Chain "Qwen repeated runs skipped (-IncludeQwen not set)."
}

Write-Chain "=== open-repeated sweep complete ==="
