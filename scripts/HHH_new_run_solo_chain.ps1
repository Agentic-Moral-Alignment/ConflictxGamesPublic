param(
    [int] $PollSeconds = 60,
    # Skip the "wait for the top-up, then require every pair at 50 kept" gates
    # and evaluate the corpus as it stands right now.
    [switch] $AllowUnderQuota,
    [int] $KeyA = 1,
    [int] $KeyB = 2
)

# Detached solo driver for the 2026-08-27 resume, split out of
# HHH_new_run_full_chain.ps1 so the solo evaluations do NOT have to wait for
# Qwen's ~11-hour play run to clear.
#
# The game chain occupies key 3 for that whole stretch but releases keys 1-2 as
# soon as the GPT-OSS mcq/open jobs finish (minutes, not hours). The solo
# evaluations only need keys 1-2, so they can run alongside the game play runs
# instead of behind them. This script therefore gates on three things and
# nothing else:
#
#   1. the solo top-up controller has exited,
#   2. all 3 value pairs are at their 50-kept quota,
#   3. no key-1/key-2 game job (mcq or open) is still running.
#
# Launch detached:  schtasks /Run /TN "HHH_solo_chain"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$ps       = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$soloDir  = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"
$soloCsv  = Join-Path $repoRoot "data/HHH_new_solo_20260826_01/claude-sonnet-5.csv"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$soloLog  = Join-Path $soloDir "solo_extended_sequence.stdout.log"

New-Item -ItemType Directory -Path $soloDir -Force | Out-Null

function Write-Chain([string] $Message) {
    # Shared log with the game chain; append with read+write sharing and retry,
    # so neither a reader nor Dropbox sync can cost a line. See the same note in
    # HHH_new_run_full_chain.ps1.
    $line = "$(Get-Date -Format s) [solo] $Message"
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

function Wait-For([string] $Label, [scriptblock] $StillBusy) {
    $waited = 0
    while ($true) {
        $busy = & $StillBusy
        if (-not $busy) { break }
        if (($waited % 1800) -eq 0) {
            Write-Chain "gate: still waiting on $Label ($([int]($waited / 60)) min)."
        }
        Start-Sleep -Seconds $PollSeconds
        $waited += $PollSeconds
    }
    Write-Chain "gate: $Label cleared."
}

Write-Chain "=== solo chain start (PID $PID) ==="

# 1. the top-up controller must have exited (skipped under -AllowUnderQuota)
if (-not $AllowUnderQuota) {
Wait-For "the solo top-up controller" {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'conflictxgame\.solo\.topup_scenarios' }).Count -gt 0
}
}
else {
    Write-Chain 'gate: top-up wait SKIPPED (-AllowUnderQuota); evaluating the corpus as it stands.'
}

# 2. every value pair must be at quota
if (-not (Test-Path -LiteralPath $soloCsv)) {
    Write-Chain "ABORT: solo corpus CSV missing at $soloCsv."
    exit 2
}
$groups = @(Import-Csv -LiteralPath $soloCsv |
            Where-Object { $_.keep_scenario -eq "True" } |
            Group-Object { (@($_.value1, $_.value2) | Sort-Object) -join "|" })
$summary = ($groups | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ", "
Write-Chain "kept quotas: $summary"
if ($groups.Count -ne 3 -or @($groups | Where-Object { $_.Count -lt 50 }).Count -gt 0) {
    if (-not $AllowUnderQuota) {
        Write-Chain "SKIPPED: a value pair is under its 50-kept quota. Re-run 'bash scripts/HHH_new_pipeline.sh solo-topup-to-50', then 'schtasks /Run /TN HHH_solo_chain'."
        exit 2
    }
    Write-Chain "UNDER QUOTA, proceeding by request: $summary. A second pass is required after the top-up finishes."
}

# 3. keys 1-2 must be free (game mcq/open jobs are the only other users)
Wait-For "game mcq/open jobs to release keys 1-2" {
    @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -match 'conflictxgame\.games\.evaluate_games' -and
                     $_.CommandLine -match 'eval-(mcq|open)_focal-' }).Count -gt 0
}

Write-Chain "start: solo baseline + steering for all 3 models on keys $KeyA/$KeyB."
$seqArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
             (Join-Path $repoRoot "scripts\HHH_new_run_solo_extended_eval_sequence.ps1"),
             '-KeyA', $KeyA, '-KeyB', $KeyB)
if ($AllowUnderQuota) { $seqArgs += '-AllowUnderQuota' }
# PS 5.1's *>> creates a NEW file as UTF-16, which reads as binary to grep/tail.
# Merge every stream into stdout and append explicitly as UTF-8 instead.
& $ps @seqArgs 2>&1 | Out-File -LiteralPath $soloLog -Encoding utf8 -Append
Write-Chain "returned (native exit code '$LASTEXITCODE'); see solo_extended_sequence.stdout.log. Row counts are the real signal."
Write-Chain "=== solo chain complete ==="
