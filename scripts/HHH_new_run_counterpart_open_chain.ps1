param(
    [int] $PollSeconds = 60,
    [int] $Key = 2,
    [switch] $SkipGate
)

# Counterpart-disclosed open-ended sweep: 3 models x 9 arms.
#
# The baseline open-ended openings hide the game -- the other decision-maker is
# named in 2 of the 150 cached openings and never called an AI -- so the target
# answers what is effectively a single-agent request. These arms rerun the open
# pipeline with the simulated user told who the counterpart is and asked to name
# it (--open-disclose-counterpart), writing to their own result files and their
# own opening-message cache.
#
# Order matters. Gemma's baseline arm runs first on the free institutional gateway
# and writes the shared cache; the gate then checks the openings actually name
# the counterpart before any paid OpenRouter model runs. States arms (750 rows,
# 150 cases) precede the one-shot arms (150 rows) per model.
#
# Launch detached:
#   Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\HHH_new_run_counterpart_open_chain.ps1'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_counterpart_chain.log"
$cache    = "data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/games/open_prompts_balanced50_counterpart_paid-gpt-5.6-luna.json"
$expected = 150

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [counterpart] $Message"
    Write-Output $line
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

function Get-ResultRowCount([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    try { return @(Import-Csv -LiteralPath $Path).Count } catch { return 0 }
}

function Get-LiveEval([string] $ConfigMatch) {
    return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match 'conflictxgame\.games\.evaluate_games' -and
                            $_.CommandLine -match $ConfigMatch }) | Select-Object -First 1
}

function Invoke-Run($Run) {
    $outPath = Join-Path $gameDir $Run.Output
    $doneCases = Get-ResultCaseCount $outPath
    $doneRows  = Get-ResultRowCount $outPath
    if ($doneCases -ge $expected -and $doneRows -ge $Run.MinRows) {
        Write-Chain "$($Run.Command) already complete ($doneCases cases / $doneRows rows); skipping."
        return $true
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
            Write-Chain "launched $($Run.Command) PID $($p.Id)."
            $p.WaitForExit()
        }
        finally {
            if ($null -eq $prior) { Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue }
            else { $env:GATEWAY_KEY_INDEX = $prior }
        }
    }
    # Verify by rows/cases, never by Process.ExitCode: PowerShell 5.1 reads it
    # back blank for a fast-exiting process even when the run succeeded.
    $n = Get-ResultCaseCount $outPath
    $r = Get-ResultRowCount $outPath
    if ($n -lt $expected -or $r -lt $Run.MinRows) {
        Write-Chain "FAIL $($Run.Command): $n/$expected cases, $r/$($Run.MinRows) rows in $($Run.Output)"
        return $false
    }
    Write-Chain "OK   $($Run.Command): $n cases / $r rows."
    return $true
}

# Arms in run order: memory-state arms (750 rows) first, then one-shot (150).
$ARMS = @(
    @{ Suffix = '-states';                   Token = 'states';                   MinRows = 750 },
    @{ Suffix = '-states-stakes-payoff';     Token = 'states-stakes-payoff';     MinRows = 750 },
    @{ Suffix = '-states-stakes-alignment';  Token = 'states-stakes-alignment';  MinRows = 750 },
    @{ Suffix = '-states-stakes-usefulness'; Token = 'states-stakes-usefulness'; MinRows = 750 },
    @{ Suffix = '';                          Token = '';                         MinRows = 150 },
    @{ Suffix = '-stakes-payoff';            Token = 'stakes-payoff';            MinRows = 150 },
    @{ Suffix = '-stakes-alignment';         Token = 'stakes-alignment';         MinRows = 150 },
    @{ Suffix = '-stakes-usefulness';        Token = 'stakes-usefulness';        MinRows = 150 },
    @{ Suffix = '-steer';                    Token = 'steer-HHH';                MinRows = 150 }
)
$JUDGE_TAIL = '_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv'

function New-RunList([string] $TargetPrefix, [string] $Slug) {
    $runs = @()
    foreach ($arm in $ARMS) {
        $token = ''
        if ($arm.Token) { $token = '_' + $arm.Token }
        $logArm = $arm.Suffix -replace '-', '_'
        $runs += @{
            Command = $TargetPrefix + $arm.Suffix
            Log     = $Slug + '_open_counterpart' + $logArm
            # Anchored with the trailing _focal- so '-counterpart_focal' cannot
            # match '-counterpart-states_focal'.
            Match   = 'eval-open-counterpart' + $arm.Suffix + '_focal-' + $Slug
            Output  = $Slug + '_open_balanced50_counterpart' + $token + $JUDGE_TAIL
            MinRows = $arm.MinRows
        }
    }
    return $runs
}

$GEMMA    = New-RunList 'game-eval-open-counterpart'          'gemma-4-31B-it'
$MAVERICK = New-RunList 'game-eval-maverick-open-counterpart' 'llama-4-maverick'
$QWEN     = New-RunList 'game-eval-qwen3235-open-counterpart' 'qwen3-235b-a22b'

Write-Chain "=== counterpart-disclosed open sweep start (PID $PID, key $Key) ==="

# The baseline Gemma arm writes the shared opening cache on a free focal model.
$seed = $GEMMA | Where-Object { $_.Command -eq 'game-eval-open-counterpart' }
if (-not (Invoke-Run $seed)) {
    Write-Chain 'ABORT: cache-seeding arm did not complete.'
    exit 1
}

if (-not $SkipGate) {
    Write-Chain 'Gate: checking that the new openings name the counterpart.'
    $gateCmd = "cd '" + $repoRoot + "' && PYTHONPATH=src python scripts/check_counterpart_openings.py --cache '" + $cache + "' --min-rate 0.8333 --show 3"
    $gate = & $gitBash -lc $gateCmd 2>&1
    $gate | ForEach-Object { Write-Chain "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Chain 'ABORT: opening-message gate failed; no paid model was run.'
        exit 1
    }
}

foreach ($group in @(
    @{ Name = 'Gemma 4 31B';      Runs = $GEMMA },
    @{ Name = 'Llama 4 Maverick'; Runs = $MAVERICK },
    @{ Name = 'Qwen 3 235B A22B'; Runs = $QWEN })) {
    Write-Chain "--- $($group.Name) ---"
    foreach ($run in $group.Runs) { [void](Invoke-Run $run) }
}

Write-Chain '=== counterpart-disclosed open sweep complete ==='
