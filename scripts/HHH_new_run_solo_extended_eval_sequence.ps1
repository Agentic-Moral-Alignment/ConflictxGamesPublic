param(
    [int] $PollSeconds = 30,
    # Evaluate the corpus as it stands instead of refusing when a value pair is
    # still short of its 50-kept quota. The per-pair selection is a stable
    # sha256(seed|scenario_id) ranking and the evaluators resume by scenario_id,
    # so a short pair can be topped up by re-running this script later -- but see
    # the warning it logs about rows that fall out of the canonical top 50.
    [switch] $AllowUnderQuota,
    # Which gateway keys the two concurrent jobs use. Both can point at the
    # same key when the others are busy with long play runs.
    [int] $KeyA = 1,
    [int] $KeyB = 2
)

# Runs after the solo scenario top-up (topup_scenarios.py) has already exited
# and every value pair has reached its kept quota -- the caller is expected to
# have waited for that separately (this script does not poll any PID itself).
# Per model: baseline balanced-50 (mcq + interactive, never run before this
# sequence) then the prompt-steering variant of the same two modes
# (--steer-prompt value_sets/steering/HHH.txt), each verified by row count
# before moving on.

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$soloCsv = Join-Path $repoRoot "data/HHH_new_solo_20260826_01/claude-sonnet-5.csv"
$resultDir = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"
New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $gitBash)) {
    throw "Git Bash was not found at $gitBash"
}

function Write-Status([string] $Message) {
    Write-Output "$(Get-Date -Format s) $Message"
}

function Invoke-CommandGroup([array] $Runs) {
    $priorIndex = $env:GATEWAY_KEY_INDEX
    $jobs = @()
    try {
        foreach ($run in $Runs) {
            $env:GATEWAY_KEY_INDEX = [string]$run.Key
            $start = @{
                FilePath = $gitBash
                ArgumentList = @($pipeline, $run.Command)
                WorkingDirectory = $repoRoot
                RedirectStandardOutput = $run.Stdout
                RedirectStandardError = $run.Stderr
                WindowStyle = "Hidden"
                PassThru = $true
            }
            $process = Start-Process @start
            $jobs += [pscustomobject]@{ Run = $run; Process = $process }
            Write-Status "Started $($run.Command) PID $($process.Id) on gateway key $($run.Key)."
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

    foreach ($job in $jobs) {
        $job.Process.WaitForExit()
        # Process.ExitCode is not trustworthy for a fast-exiting process on
        # PowerShell 5.1 (reproducibly reads back blank/null even after
        # WaitForExit(), confirmed against a job that actually succeeded) --
        # the caller's Assert-CsvRows on the actual output is the real
        # correctness signal, so a bad code here is only logged.
        $exitCode = $job.Process.ExitCode
        if ($exitCode -isnot [int] -or $exitCode -ne 0) {
            Write-Status "$($job.Run.Command) reported exit code '$exitCode' (unreliable when a job exits fast); verifying by row count instead."
        }
        else {
            Write-Status "$($job.Run.Command) complete."
        }
    }
}

function Assert-CsvRows([string] $Path, [int] $MinimumRows) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing expected result $Path"
    }
    $rows = @(Import-Csv -LiteralPath $Path).Count
    if ($rows -lt $MinimumRows) {
        throw "$Path has $rows rows; expected at least $MinimumRows"
    }
}

# Fail fast rather than evaluate a moving/incomplete corpus.
if (-not (Test-Path -LiteralPath $soloCsv)) {
    throw "Solo corpus CSV not found at $soloCsv; has the top-up run yet?"
}
$keptGroups = @(
    Import-Csv -LiteralPath $soloCsv |
        Where-Object { $_.keep_scenario -eq "True" } |
        Group-Object { (@($_.value1, $_.value2) | Sort-Object) -join "|" }
)
$summary = ($keptGroups | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Count)" }) -join ", "

# evaluate_models selects exactly pair_limit scenarios per pair and raises if any
# pair has fewer, so the row count per mode is always 3 * pair_limit. Read it from
# the config rather than hardcoding, so lowering pair_limit to run under quota does
# not leave these assertions expecting rows that will never exist.
$mcqConfigPath = Join-Path $repoRoot "configs/solo/HHH_new_solo_20260826_01_eval-mcq_focal-gemma-4-31B-it.yaml"
$pairLimitLine = Select-String -LiteralPath $mcqConfigPath -Pattern '^\s*pair_limit:\s*(\d+)' | Select-Object -First 1
if ($null -eq $pairLimitLine) { throw "Could not read pair_limit from $mcqConfigPath" }
$pairLimit = [int]$pairLimitLine.Matches[0].Groups[1].Value
$expectedRows = 3 * $pairLimit

$short = @($keptGroups | Where-Object { $_.Count -lt $pairLimit })
if ($keptGroups.Count -ne 3 -or $short.Count -gt 0) {
    if (-not $AllowUnderQuota) {
        throw "Solo kept quotas are incomplete for pair_limit=$pairLimit; refusing to evaluate: $summary"
    }
    Write-Status "WARNING: evaluating UNDER QUOTA by request (-AllowUnderQuota): $summary"
}
if ($pairLimit -lt 50) {
    Write-Status "WARNING: pair_limit is $pairLimit, not the full 50. A short pair may later gain scenarios that outrank ones evaluated now, leaving rows outside the canonical top-50 selection. Restore pair_limit to 50 after the top-up, re-run to top these results up in place, then prune to the canonical set before analysis."
}
if ($keptGroups.Count -eq 3 -and $short.Count -eq 0) {
    Write-Status "Solo corpus verified: all 3 value pairs have at least $pairLimit kept scenarios."
}
Write-Status "Expecting $expectedRows rows per mode (3 pairs x pair_limit $pairLimit). Corpus: $summary."

function Run-Model([string] $Label, [string] $ShortName, [string] $McqCmd, [string] $InteractiveCmd,
                    [string] $McqSteerCmd, [string] $InteractiveSteerCmd) {
    Write-Status "Starting $Label solo baseline (mcq + interactive)."
    Invoke-CommandGroup @(
        @{Command=$McqCmd; Key=$KeyA;
          Stdout=(Join-Path $resultDir "$ShortName`_mcq_baseline.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_mcq_baseline.stderr.log")},
        @{Command=$InteractiveCmd; Key=$KeyB;
          Stdout=(Join-Path $resultDir "$ShortName`_interactive_baseline.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_interactive_baseline.stderr.log")}
    )
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_mcq_balanced50.csv") $expectedRows
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_interactive_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") $expectedRows
    Write-Status "$Label solo baseline verified ($expectedRows rows each)."

    Write-Status "Starting $Label solo steering (mcq + interactive, HHH.txt)."
    Invoke-CommandGroup @(
        @{Command=$McqSteerCmd; Key=$KeyA;
          Stdout=(Join-Path $resultDir "$ShortName`_mcq_steer.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_mcq_steer.stderr.log")},
        @{Command=$InteractiveSteerCmd; Key=$KeyB;
          Stdout=(Join-Path $resultDir "$ShortName`_interactive_steer.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_interactive_steer.stderr.log")}
    )
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_mcq_balanced50_steer-HHH.csv") $expectedRows
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_interactive_balanced50_steer-HHH_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") $expectedRows
    Write-Status "$Label solo steering verified ($expectedRows rows each)."

    # solo-analyze is a post-hoc reporting step over CSVs that are already on
    # disk, so a failure here must not abandon the remaining models' data
    # collection -- it can be re-run at any time. Warn loudly and continue.
    & $gitBash $pipeline "solo-analyze"
    if ($LASTEXITCODE -ne 0) {
        Write-Status "WARNING: shared solo analysis failed after $Label (exit $LASTEXITCODE). Results CSVs are unaffected; re-run 'bash scripts/HHH_new_pipeline.sh solo-analyze' once fixed. Continuing with the remaining models."
    }
}

Run-Model "Gemma 4 31B" "gemma-4-31B-it" "solo-eval-mcq" "solo-eval-interactive" `
    "solo-eval-mcq-steer" "solo-eval-interactive-steer"
Run-Model "GPT-OSS 120B" "gpt-oss-120b" "solo-eval-gptoss-mcq" "solo-eval-gptoss-interactive" `
    "solo-eval-gptoss-mcq-steer" "solo-eval-gptoss-interactive-steer"
# Qwen 3.5 122B runs LAST: on the institutional gateway it is roughly an order of
# magnitude slower per request than the other two, so putting it first would
# hold the faster models behind it for hours.
Run-Model "Qwen 3.5 122B" "qwen3.5-122b" "solo-eval-qwen-mcq" "solo-eval-qwen-interactive" `
    "solo-eval-qwen-mcq-steer" "solo-eval-qwen-interactive-steer"

Write-Status "Solo baseline and steering evaluations are complete for all 3 models."
