param(
    # Gateway worker: pull arms off the shared queue using this gateway key.
    [int] $KeyWorker = 0,
    # OpenRouter worker: run just this model's nine arms, sequentially.
    [string] $Model = '',
    [int] $PollSeconds = 60,
    [switch] $SkipGate
)

# Counterpart-disclosed open-ended sweep across all ten paper models.
#
# Parent mode (no -KeyWorker / -Model): verifies the shared opening cache
# passes the gate, then launches
#   * three gateway workers, one per gateway key, which pull arms greedily
#     off a shared 27-arm queue (3 models x 9 arms). A key never sits idle
#     because a worker takes the next unclaimed arm regardless of which model
#     it belongs to; claiming is an atomic create-new of a marker file, so two
#     workers cannot take the same arm. The evaluation seed stays 0 throughout
#     -- the keys are parallelism, not a different sample.
#   * seven OpenRouter workers, one per model, each with
#     CONFLICTXGAME_MAX_WORKERS=1. openrouter_env.sh defaults to 3 and
#     documents HTTP 402 (in_flight_budget_exhausted) when too much is in
#     flight; the parallelism here comes from models, not workers per model.
#
# Arms per model: the four memory-state arms (750 rows) then the five one-shot
# arms (150 rows). The gateway queue is ordered states-first so the long arms
# start earliest and the keys drain evenly. Completed arms are skipped, so this
# is safe to re-run and safe to point at a partially finished sweep.

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash   = "C:\Program Files\Git\bin\bash.exe"
$pipeline  = "scripts/HHH_new_pipeline.sh"
$gameDir   = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog  = Join-Path $repoRoot "results/HHH_new_counterpart_chain.log"
$claimDir  = Join-Path $repoRoot "results/.counterpart_claims"
$cache     = "data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/games/open_prompts_balanced50_counterpart_paid-gpt-5.6-luna.json"
$expected  = 150

$GATEWAY = @(
    @{ Slug = 'gemma-4-31B-it'; Prefix = 'game-eval-open-counterpart' },
    @{ Slug = 'gemma-3-27b';    Prefix = 'game-eval-gemma3-open-counterpart' },
    @{ Slug = 'gpt-oss-120b';   Prefix = 'game-eval-gptoss-open-counterpart' }
)
$OPENROUTER = @(
    @{ Slug = 'llama-4-maverick'; Prefix = 'game-eval-maverick-open-counterpart' },
    @{ Slug = 'qwen3-235b-a22b';  Prefix = 'game-eval-qwen3235-open-counterpart' },
    @{ Slug = 'llama-3.3-70b';    Prefix = 'game-eval-llama33-open-counterpart' },
    @{ Slug = 'qwen3-32b';        Prefix = 'game-eval-qwen332-open-counterpart' },
    @{ Slug = 'gemma-3-12b';      Prefix = 'game-eval-gemma312-open-counterpart' },
    @{ Slug = 'gemma-4-26b-a4b';  Prefix = 'game-eval-gemma426-open-counterpart' },
    @{ Slug = 'gpt-oss-20b';      Prefix = 'game-eval-oss20-open-counterpart' }
)
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

function Write-Chain([string] $Message) {
    $tag = if ($KeyWorker) { "[key$KeyWorker]" } elseif ($Model) { "[$Model]" } else { '[all]' }
    $line = "$(Get-Date -Format s) $tag $Message"
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

function New-Arm($Slug, $Prefix, $Arm) {
    $token = ''
    if ($Arm.Token) { $token = '_' + $Arm.Token }
    return [pscustomobject]@{
        Slug    = $Slug
        Command = $Prefix + $Arm.Suffix
        Log     = $Slug + '_open_counterpart' + ($Arm.Suffix -replace '-', '_')
        Match   = 'eval-open-counterpart' + $Arm.Suffix + '_focal-' + $Slug
        Output  = $Slug + '_open_balanced50_counterpart' + $token + $JUDGE_TAIL
        MinRows = $Arm.MinRows
    }
}

function Test-ArmComplete($Arm) {
    $p = Join-Path $gameDir $Arm.Output
    return ((Get-ResultCaseCount $p) -ge $expected -and (Get-ResultRowCount $p) -ge $Arm.MinRows)
}

function Invoke-Arm($Arm) {
    $outPath = Join-Path $gameDir $Arm.Output
    $live = Get-LiveEval $Arm.Match
    if ($null -ne $live) {
        Write-Chain "$($Arm.Command) already running as PID $($live.ProcessId); waiting."
        while ($null -ne (Get-LiveEval $Arm.Match)) { Start-Sleep -Seconds $PollSeconds }
    }
    else {
        $startArgs = @{
            FilePath = $gitBash
            ArgumentList = @($pipeline, $Arm.Command)
            WorkingDirectory = $repoRoot
            RedirectStandardOutput = (Join-Path $gameDir "$($Arm.Log).stdout.log")
            RedirectStandardError = (Join-Path $gameDir "$($Arm.Log).stderr.log")
            WindowStyle = "Hidden"
            PassThru = $true
        }
        $p = Start-Process @startArgs
        Write-Chain "run $($Arm.Command) (PID $($p.Id))."
        $p.WaitForExit()
    }
    # Verify by rows/cases, never Process.ExitCode: PowerShell 5.1 reads it back
    # blank for a fast-exiting process even on success.
    $n = Get-ResultCaseCount $outPath
    $r = Get-ResultRowCount $outPath
    if ($n -lt $expected -or $r -lt $Arm.MinRows) {
        Write-Chain "FAIL $($Arm.Command): $n/$expected cases, $r/$($Arm.MinRows) rows."
        $err = Join-Path $gameDir "$($Arm.Log).stderr.log"
        if (Test-Path -LiteralPath $err) {
            $hit = Select-String -LiteralPath $err -Pattern '402|in_flight_budget|Unknown command|Traceback|rate.?limit' |
                   Select-Object -First 1
            if ($hit) { Write-Chain "  stderr hint: $($hit.Line.Trim())" }
        }
        return $false
    }
    Write-Chain "OK   $($Arm.Command): $n cases / $r rows."
    return $true
}

# ------------------------------------------------- gateway queue worker --
if ($KeyWorker -gt 0) {
    $env:GATEWAY_KEY_INDEX = [string]$KeyWorker
    Write-Chain "worker start on gateway key $KeyWorker."
    # States arms first so the long jobs begin earliest; models interleaved so
    # one slow model does not leave the other keys with only short arms.
    $queue = @()
    foreach ($arm in $ARMS) {
        foreach ($m in $GATEWAY) { $queue += New-Arm $m.Slug $m.Prefix $arm }
    }
    New-Item -ItemType Directory -Path $claimDir -Force | Out-Null
    $did = 0
    while ($true) {
        $tookOne = $false
        foreach ($a in $queue) {
            if (Test-ArmComplete $a) { continue }
            $claim = Join-Path $claimDir ($a.Command + '.claim')
            try { New-Item -ItemType File -Path $claim -ErrorAction Stop | Out-Null }
            catch { continue }          # another key already owns this arm
            $tookOne = $true; $did++
            [void](Invoke-Arm $a)
        }
        if (-not $tookOne) { break }     # nothing left to claim
    }
    Write-Chain "worker done on key $KeyWorker ($did arms taken)."
    exit 0
}

# ------------------------------------------------ OpenRouter model worker --
if ($Model) {
    $spec = $OPENROUTER | Where-Object { $_.Slug -eq $Model }
    if (-not $spec) { Write-Chain "unknown OpenRouter model '$Model'"; exit 2 }
    $env:CONFLICTXGAME_MAX_WORKERS = '1'
    Write-Chain 'worker start (OpenRouter, 1 in-flight request).'
    foreach ($arm in $ARMS) {
        $a = New-Arm $spec.Slug $spec.Prefix $arm
        if (Test-ArmComplete $a) { Write-Chain "skip $($a.Command) (already complete)."; continue }
        [void](Invoke-Arm $a)
    }
    Write-Chain 'worker done.'
    exit 0
}

# ----------------------------------------------------------------- parent --
Write-Chain "=== all-model counterpart sweep start (PID $PID) ==="

if (-not $SkipGate) {
    $gateCmd = "cd '" + $repoRoot + "' && PYTHONPATH=src python scripts/check_counterpart_openings.py --cache '" + $cache + "' --min-rate 0.8333"
    $gate = & $gitBash -lc $gateCmd 2>&1
    $gate | ForEach-Object { Write-Chain "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Chain 'ABORT: opening cache has not passed the gate; run the seeding arm first.'
        exit 1
    }
}

$self = Join-Path 'scripts' 'HHH_new_run_counterpart_all_models.ps1'
$jobs = @()
foreach ($k in 1, 2, 3) {
    $p = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-File', $self, '-KeyWorker', [string]$k) `
            -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
    $jobs += $p
    Write-Chain "launched gateway queue worker on key $k (PID $($p.Id))."
    Start-Sleep -Seconds 2
}
foreach ($m in $OPENROUTER) {
    $p = Start-Process -FilePath 'powershell.exe' `
            -ArgumentList @('-NoProfile', '-File', $self, '-Model', $m.Slug) `
            -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru
    $jobs += $p
    Write-Chain "launched OpenRouter worker $($m.Slug) (PID $($p.Id))."
    Start-Sleep -Seconds 2
}

foreach ($j in $jobs) { $j.WaitForExit() }
Write-Chain '=== all-model counterpart sweep complete ==='
