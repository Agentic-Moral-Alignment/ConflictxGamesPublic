param(
    [int] $PollSeconds = 30
)

# Runs after the balanced-50 game-eval sequence (mcq/open/play for all 3
# models) has already finished -- the caller is expected to have waited for
# that separately (this script does not poll any PID itself; it just
# verifies the baseline result files are actually there before proceeding).
# Per model: the prompt-steering variant of mcq/open/play
# (--steer-prompt value_sets/steering/HHH.txt), then the repeated,
# memory_k=1 variant of open mode (5 rounds, judged every round), each
# verified by row count before moving on.

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$resultDir = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"

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

function Assert-BaselineComplete([string] $ShortName) {
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_mcq_balanced50.csv") 150
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv") 3000
}

# Fail fast rather than build on top of an incomplete/moving baseline.
foreach ($short in @("gemma-4-31B-it", "qwen3.5-122b", "gpt-oss-120b")) {
    Assert-BaselineComplete $short
}
Write-Status "Balanced-50 baseline (mcq/open/play, all 3 models) verified complete."

function Run-Model([string] $Label, [string] $ShortName,
                    [string] $McqSteerCmd, [string] $OpenSteerCmd, [string] $PlaySteerCmd,
                    [string] $OpenRepeatedCmd) {
    Write-Status "Starting $Label game steering (mcq + open + play, HHH.txt)."
    Invoke-CommandGroup @(
        @{Command=$McqSteerCmd; Key=1;
          Stdout=(Join-Path $resultDir "$ShortName`_mcq_steer.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_mcq_steer.stderr.log")},
        @{Command=$OpenSteerCmd; Key=2;
          Stdout=(Join-Path $resultDir "$ShortName`_open_steer.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_open_steer.stderr.log")},
        @{Command=$PlaySteerCmd; Key=3;
          Stdout=(Join-Path $resultDir "$ShortName`_play_steer.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_play_steer.stderr.log")}
    )
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_mcq_balanced50_steer-HHH.csv") 150
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_open_balanced50_steer-HHH_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_play_balanced50_steer-HHH_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv") 3000
    Write-Status "$Label game steering verified."

    Write-Status "Starting $Label repeated open-ended (5 rounds, memory_k=1)."
    Invoke-CommandGroup @(
        @{Command=$OpenRepeatedCmd; Key=1;
          Stdout=(Join-Path $resultDir "$ShortName`_open_repeated.stdout.log");
          Stderr=(Join-Path $resultDir "$ShortName`_open_repeated.stderr.log")}
    )
    Assert-CsvRows (Join-Path $resultDir "$ShortName`_open_balanced50_repeated_rounds5_mem1_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 750
    Write-Status "$Label repeated open-ended verified (>=750 rows: 150 cases x 5 rounds)."
}

Run-Model "Gemma 4 31B" "gemma-4-31B-it" `
    "game-eval-mcq-steer" "game-eval-open-steer" "game-eval-play-steer" "game-eval-open-repeated"
Run-Model "Qwen 3.5 122B" "qwen3.5-122b" `
    "game-eval-qwen-mcq-steer" "game-eval-qwen-open-steer" "game-eval-qwen-play-steer" "game-eval-qwen-open-repeated"
Run-Model "GPT-OSS 120B" "gpt-oss-120b" `
    "game-eval-gptoss-mcq-steer" "game-eval-gptoss-open-steer" "game-eval-gptoss-play-steer" "game-eval-gptoss-open-repeated"

Write-Status "Game steering and repeated open-ended evaluations are complete for all 3 models."
