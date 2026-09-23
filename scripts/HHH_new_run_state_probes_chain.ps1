param(
    [int] $PollSeconds = 60,
    [int] $Key = 2,
    # Qwen's online play run waits for its existing play evaluation to clear:
    # both hit qwen3.5-122b, and that run is already the slowest thing here.
    [switch] $IncludeQwen
)

# MCQ evaluated once per distinct mem-1 memory state: a fresh start plus every
# possible previous-round outcome, which for these 2x2 games is exactly five.
#
# Under the online framing the model is told neither the horizon nor how many
# rounds have passed, so with memory_k=1 that finite set IS its whole decision
# context. Enumerating it beats sampling trajectories through it: play visits
# states at rates set by the scripted opponent and the model's own choices (one
# opponent reached 10 decisions against 750 for the others), whereas this gives
# every state the same n and needs no opponent at all. 150 cases x 5 states =
# 750 rows per model, against a 13-hour play run.
#
# Launch detached:  schtasks /Run /TN "HHH_state_probes_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [states] $Message"
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

$FAST = @(
    @{ Command = 'game-eval-gptoss-mcq-states'; Log = 'gpt-oss-120b_mcq_states';
       Match = 'eval-mcq-states_focal-gpt-oss';
       Output = 'gpt-oss-120b_mcq_balanced50_states.csv' },
    @{ Command = 'game-eval-mcq-states'; Log = 'gemma-4-31B-it_mcq_states';
       Match = 'eval-mcq-states_focal-gemma';
       Output = 'gemma-4-31B-it_mcq_balanced50_states.csv' }
)
$QWEN = @()

Write-Chain "=== state-probe sweep start (PID $PID, key $Key) ==="
foreach ($run in $FAST) { Invoke-Run $run }
Write-Chain 'Gemma and GPT-OSS done: mcq over all five mem-1 states.'

Write-Chain 'Qwen not included (stopped by request).'

Write-Chain "=== state-probe sweep complete ==="
