param(
    [int] $PollSeconds = 60,
    [int] $Key = 2,
    # Qwen's online play run waits for its existing play evaluation to clear:
    # both hit qwen3.5-122b, and that run is already the slowest thing here.
    [switch] $IncludeQwen
)

# Visibility arms for mcq and open: the system prompt states whether this
# decision is hidden from or revealed to the other participant, with no value
# content. Both arms are run, not just `unobserved` -- the unsteered baseline
# says nothing about visibility, so unobserved-vs-baseline could be the extra
# paragraph rather than the invisibility. `observed` is the matched control.
#
# Play is excluded: it reveals each round's choices by design, so an unobserved
# claim would contradict the mechanic the model is shown. That arm needs a
# structure-text change, not just a system prompt.
#
# Launch detached:  schtasks /Run /TN "HHH_visibility_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [vis] $Message"
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
    @{ Command = 'game-eval-mcq-unobserved'; Log = 'gemma-4-31B-it_mcq_unobserved';
       Match = 'eval-mcq-unobserved_focal-gemma';
       Output = 'gemma-4-31B-it_mcq_balanced50_unobserved.csv' },
    @{ Command = 'game-eval-open-unobserved'; Log = 'gemma-4-31B-it_open_unobserved';
       Match = 'eval-open-unobserved_focal-gemma';
       Output = 'gemma-4-31B-it_open_balanced50_unobserved_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' },
    @{ Command = 'game-eval-mcq-observed'; Log = 'gemma-4-31B-it_mcq_observed';
       Match = 'eval-mcq-observed_focal-gemma';
       Output = 'gemma-4-31B-it_mcq_balanced50_observed.csv' },
    @{ Command = 'game-eval-open-observed'; Log = 'gemma-4-31B-it_open_observed';
       Match = 'eval-open-observed_focal-gemma';
       Output = 'gemma-4-31B-it_open_balanced50_observed_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' },
    @{ Command = 'game-eval-gptoss-mcq-unobserved'; Log = 'gpt-oss-120b_mcq_unobserved';
       Match = 'eval-mcq-unobserved_focal-gpt-oss';
       Output = 'gpt-oss-120b_mcq_balanced50_unobserved.csv' },
    @{ Command = 'game-eval-gptoss-open-unobserved'; Log = 'gpt-oss-120b_open_unobserved';
       Match = 'eval-open-unobserved_focal-gpt-oss';
       Output = 'gpt-oss-120b_open_balanced50_unobserved_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' },
    @{ Command = 'game-eval-gptoss-mcq-observed'; Log = 'gpt-oss-120b_mcq_observed';
       Match = 'eval-mcq-observed_focal-gpt-oss';
       Output = 'gpt-oss-120b_mcq_balanced50_observed.csv' },
    @{ Command = 'game-eval-gptoss-open-observed'; Log = 'gpt-oss-120b_open_observed';
       Match = 'eval-open-observed_focal-gpt-oss';
       Output = 'gpt-oss-120b_open_balanced50_observed_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv' }
)
$QWEN = @()

Write-Chain "=== visibility sweep start (PID $PID, key $Key) ==="
foreach ($run in $FAST) { Invoke-Run $run }
Write-Chain 'Gemma and GPT-OSS done: mcq and open, observed and unobserved.'

Write-Chain 'Qwen visibility arms not included in this sweep.'

Write-Chain "=== visibility sweep complete ==="
