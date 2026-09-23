param(
    [int] $PollSeconds = 60,
    [int] $Key = 2,
    # Qwen's online play run waits for its existing play evaluation to clear:
    # both hit qwen3.5-122b, and that run is already the slowest thing here.
    [switch] $IncludeQwen
)

# Play mode under the ONLINE framing: the repeated-game prompt no longer states
# the horizon or the elapsed round count, and shows only what memory_k allows,
# labelled relative to now ("Previous round") instead of by absolute index.
#
# This is a different measurement from the play CSVs already on disk, which were
# produced under a prompt that said "This interaction repeats for 5 rounds" and
# numbered the history absolutely. The outputs carry `_online` so the two never
# mix, and neither resume-dedup nor the analysis can confuse them.
#
# Order is cheapest-first: GPT-OSS ran the old play mode in 29 minutes against
# Gemma's ~2 h and Qwen's 13 h, so it gives the first read on the new framing.
#
# Launch detached:  schtasks /Run /TN "HHH_play_online_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [online] $Message"
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

$TAIL = '_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv'
$FAST = @(
    @{ Command = 'game-eval-gptoss-play-online'; Log = 'gpt-oss-120b_play_online';
       Match = 'eval-play-online_focal-gpt-oss';
       Output = 'gpt-oss-120b_play_balanced50_online' + $TAIL },
    @{ Command = 'game-eval-play-online'; Log = 'gemma-4-31B-it_play_online';
       Match = 'eval-play-online_focal-gemma';
       Output = 'gemma-4-31B-it_play_balanced50_online' + $TAIL }
)
$QWEN = @(
    @{ Command = 'game-eval-qwen-play-online'; Log = 'qwen3.5-122b_play_online';
       Match = 'eval-play-online_focal-qwen';
       Output = 'qwen3.5-122b_play_balanced50_online' + $TAIL }
)

Write-Chain "=== play ONLINE sweep start (PID $PID, key $Key) ==="
foreach ($run in $FAST) { Invoke-Run $run }
Write-Chain "fast models done (GPT-OSS, Gemma) under the online framing."

if ($IncludeQwen) {
    Write-Chain "waiting for Qwen's existing play run to clear before its online run."
    $waited = 0
    while ($null -ne (Get-LiveEval 'eval-play_focal-qwen3\.5-122b')) {
        if (($waited % 1800) -eq 0) { Write-Chain "still waiting on Qwen play ($([int]($waited / 60)) min)." }
        Start-Sleep -Seconds $PollSeconds
        $waited += $PollSeconds
    }
    foreach ($run in $QWEN) { Invoke-Run $run }
}
else {
    Write-Chain "Qwen online play skipped (-IncludeQwen not set)."
}

Write-Chain "=== play ONLINE sweep complete ==="
