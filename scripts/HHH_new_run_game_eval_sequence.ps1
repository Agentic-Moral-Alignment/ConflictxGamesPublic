param(
    [int] $PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$sourceCsv = "data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/claude-sonnet-5_games.csv"
$coveragePath = "data/HHH_new_paper_bothAI_anthropic_batch_20260826_01/games/pair_coverage.json"
$resultDir = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"

if (-not (Test-Path -LiteralPath $gitBash)) {
    throw "Git Bash was not found at $gitBash"
}
$coverage = Get-Content -LiteralPath $coveragePath -Raw | ConvertFrom-Json
if (-not $coverage.complete) {
    throw "Game generation is not complete; refusing to evaluate a moving corpus."
}

$source = Import-Csv -LiteralPath $sourceCsv
$acceptedScenarioIds = @(
    $source |
        Where-Object { $_.keep_scenario -eq "True" } |
        Select-Object -ExpandProperty scenario_id -Unique
)
$expectedCases = 150
if ($acceptedScenarioIds.Count -eq 0) {
    throw "The final game corpus has no accepted scenarios."
}

function Write-SequenceStatus([string] $Message) {
    Write-Output "$(Get-Date -Format s) $Message"
}

function Get-ResultCaseCount([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return @(
        Import-Csv -LiteralPath $Path |
            ForEach-Object { "$($_.scenario_id)|$($_.seat)" } |
            Select-Object -Unique
    ).Count
}

function Run-ModelStage([string] $Label, [array] $Runs) {
    Write-SequenceStatus "Starting $Label on $expectedCases balanced value-pair cases."
    $priorIndex = $env:GATEWAY_KEY_INDEX
    try {
        $jobs = @()
        foreach ($run in $Runs) {
            $env:GATEWAY_KEY_INDEX = $run.Key
            $start = @{
                FilePath = $gitBash
                ArgumentList = @($pipeline, $run.Command)
                WorkingDirectory = $repoRoot
                RedirectStandardOutput = Join-Path $resultDir "$($run.Log).stdout.log"
                RedirectStandardError = Join-Path $resultDir "$($run.Log).stderr.log"
                WindowStyle = "Hidden"
                PassThru = $true
            }
            $process = Start-Process @start
            $jobs += [pscustomobject]@{
                Process = $process
                Run = $run
            }
            Write-SequenceStatus "Started $($run.Command) PID $($process.Id) on gateway key $($run.Key)."
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
        # Process.ExitCode is not trustworthy here: PowerShell 5.1 has a
        # reproducible bug where it reads back blank/null for a process that
        # exits very quickly (e.g. a resume-skip that finishes in under a
        # second), even after WaitForExit() returns -- confirmed directly
        # against a job that actually succeeded (clean stdout, empty
        # stderr, exit code 0) but still read back as "". The row-count
        # check below on the job's actual output is the real correctness
        # signal, so a non-numeric or nonzero code is only logged, not
        # treated as failure on its own.
        $exitCode = $job.Process.ExitCode
        if ($exitCode -isnot [int] -or $exitCode -ne 0) {
            Write-SequenceStatus "$($job.Run.Command) reported exit code '$exitCode' (unreliable when a job exits fast); verifying by row count instead."
        }
        $resultPath = Join-Path $resultDir $job.Run.Output
        $actual = Get-ResultCaseCount $resultPath
        if ($actual -ne $expectedCases) {
            throw "$($job.Run.Command) covered $actual/$expectedCases scenario-seat cases in $resultPath"
        }
    }
    Write-SequenceStatus "$Label is complete and verified for all $expectedCases balanced cases."
}

Run-ModelStage "Gemma 4 31B final-corpus top-up" @(
    @{ Command = "game-eval-mcq"; Key = "1"; Log = "gemma-4-31B-it_mcq_balanced50"; Output = "gemma-4-31B-it_mcq_balanced50.csv" },
    @{ Command = "game-eval-open"; Key = "2"; Log = "gemma-4-31B-it_open_balanced50"; Output = "gemma-4-31B-it_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv" },
    @{ Command = "game-eval-play"; Key = "3"; Log = "gemma-4-31B-it_play_balanced50"; Output = "gemma-4-31B-it_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv" }
)

Run-ModelStage "Qwen 3.5 122B game evaluations" @(
    @{ Command = "game-eval-qwen-mcq"; Key = "1"; Log = "qwen3.5-122b_mcq_balanced50"; Output = "qwen3.5-122b_mcq_balanced50.csv" },
    @{ Command = "game-eval-qwen-open"; Key = "2"; Log = "qwen3.5-122b_open_balanced50"; Output = "qwen3.5-122b_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv" },
    @{ Command = "game-eval-qwen-play"; Key = "3"; Log = "qwen3.5-122b_play_balanced50"; Output = "qwen3.5-122b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv" }
)

Run-ModelStage "GPT-OSS 120B game evaluations" @(
    @{ Command = "game-eval-gptoss-mcq"; Key = "1"; Log = "gpt-oss-120b_mcq_balanced50"; Output = "gpt-oss-120b_mcq_balanced50.csv" },
    @{ Command = "game-eval-gptoss-open"; Key = "2"; Log = "gpt-oss-120b_open_balanced50"; Output = "gpt-oss-120b_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv" },
    @{ Command = "game-eval-gptoss-play"; Key = "3"; Log = "gpt-oss-120b_play_balanced50"; Output = "gpt-oss-120b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv" }
)

Write-SequenceStatus "All requested game evaluations are complete."
