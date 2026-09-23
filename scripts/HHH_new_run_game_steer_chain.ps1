param(
    [int] $PollSeconds = 60,
    # gateway key for the fast models. Key 1 is free once GPT-OSS play ends.
    [int] $Key = 1,
    # Run Qwen's steering runs too. They hit the same backend model as Qwen's
    # play run, so this waits for that to clear first rather than tripling the
    # load on one model.
    [switch] $IncludeQwen,
    # Run Qwen alongside its own play evaluation instead of after it.
    # They share the qwen3.5-122b backend, so both get slower -- but with the
    # other keys idle that is still better total throughput than serialising.
    [switch] $NoWaitForPlay
)

# Prompt-steering sweep for the GAME domain: the mcq and open-ended modes with
# --steer-prompt value_sets/steering/HHH.txt, so the steering effect can be read
# against the unsteered baselines already on disk.
#
# Play mode is deliberately NOT included -- it is the expensive one (Qwen's
# unsteered play run alone takes ~8 h) and was not asked for. Add it later with
# game-eval-{,qwen-,gptoss-}play-steer if the mcq/open result justifies it.
#
# Order is cheapest-first: Gemma and GPT-OSS both finish their two modes in
# minutes, so their answers land long before Qwen's.
#
# Launch detached:  schtasks /Run /TN "HHH_game_steer_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [steer] $Message"
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

function Invoke-Steer($Run) {
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
            # ExitCode is unreliable for fast Start-Process jobs on PS 5.1; the
            # case count below is the real pass/fail signal.
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

$FAST = @(
    @{ Command = 'game-eval-mcq-steer'; Log = 'gemma-4-31B-it_mcq_steer';
       Match = 'eval-mcq-steer_focal-gemma'; Output = 'gemma-4-31B-it_mcq_balanced50_steer-HHH.csv' },
    @{ Command = 'game-eval-open-steer'; Log = 'gemma-4-31B-it_open_steer';
       Match = 'eval-open-steer_focal-gemma';
       Output = 'gemma-4-31B-it_open_balanced50_steer-HHH_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' },
    @{ Command = 'game-eval-gptoss-mcq-steer'; Log = 'gpt-oss-120b_mcq_steer';
       Match = 'eval-mcq-steer_focal-gpt-oss'; Output = 'gpt-oss-120b_mcq_balanced50_steer-HHH.csv' },
    @{ Command = 'game-eval-gptoss-open-steer'; Log = 'gpt-oss-120b_open_steer';
       Match = 'eval-open-steer_focal-gpt-oss';
       Output = 'gpt-oss-120b_open_balanced50_steer-HHH_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' }
)
$QWEN = @(
    @{ Command = 'game-eval-qwen-mcq-steer'; Log = 'qwen3.5-122b_mcq_steer';
       Match = 'eval-mcq-steer_focal-qwen'; Output = 'qwen3.5-122b_mcq_balanced50_steer-HHH.csv' },
    @{ Command = 'game-eval-qwen-open-steer'; Log = 'qwen3.5-122b_open_steer';
       Match = 'eval-open-steer_focal-qwen';
       Output = 'qwen3.5-122b_open_balanced50_steer-HHH_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' }
)

Write-Chain "=== game steering sweep start (PID $PID, key $Key) ==="
foreach ($run in $FAST) { Invoke-Steer $run }
Write-Chain "fast models done (Gemma, GPT-OSS): mcq-steer and open-steer."

if ($IncludeQwen) {
    if ($NoWaitForPlay) {
        Write-Chain 'running Qwen alongside its play evaluation by request; both will be slower.'
    }
    else {
    Write-Chain "waiting for Qwen's play run to clear before its steering runs (same backend model)."
        $waited = 0
        while ($null -ne (Get-LiveEval 'eval-play_focal-qwen3\.5-122b')) {
            if (($waited % 1800) -eq 0) { Write-Chain "still waiting on Qwen play ($([int]($waited / 60)) min)." }
            Start-Sleep -Seconds $PollSeconds
            $waited += $PollSeconds
        }
    }
    foreach ($run in $QWEN) { Invoke-Steer $run }
}
else {
    Write-Chain "Qwen steering skipped (-IncludeQwen not set)."
}

Write-Chain "=== game steering sweep complete ==="
