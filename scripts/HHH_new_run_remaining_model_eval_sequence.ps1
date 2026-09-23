param(
    [int] $GemmaSoloSequenceProcessId = 45252,
    [int] $PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameResultDir = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$soloResultDir = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"
New-Item -ItemType Directory -Path $gameResultDir -Force | Out-Null
New-Item -ItemType Directory -Path $soloResultDir -Force | Out-Null

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
        if ($job.Process.ExitCode -ne 0) {
            throw "$($job.Run.Command) failed with exit code $($job.Process.ExitCode); inspect $($job.Run.Stderr)"
        }
        Write-Status "$($job.Run.Command) complete."
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

while (Get-Process -Id $GemmaSoloSequenceProcessId -ErrorAction SilentlyContinue) {
    Write-Status "Waiting for priority Gemma solo sequence PID $GemmaSoloSequenceProcessId."
    Start-Sleep -Seconds $PollSeconds
}
$gemmaSequenceLog = Join-Path $soloResultDir "solo_eval_sequence.stdout.log"
$gemmaComplete = Select-String -LiteralPath $gemmaSequenceLog -SimpleMatch `
    "Gemma solo evaluation and analysis sequence complete." -Quiet
if (-not $gemmaComplete) {
    # Older PowerShell process objects can lose ExitCode after HasExited and
    # make the parent report a false failure. Advance only if both complete
    # result files prove that all retained scenarios landed.
    Assert-CsvRows (Join-Path $soloResultDir "gemma-4-31B-it_mcq_balanced50.csv") 150
    Assert-CsvRows (Join-Path $soloResultDir "gemma-4-31B-it_interactive_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
    Write-Status "Gemma result files are complete despite a missing sequence marker; refreshing shared solo analysis."
    & $gitBash $pipeline "solo-analyze"
    if ($LASTEXITCODE -ne 0) { throw "Shared solo analysis failed after Gemma recovery." }
}

Write-Status "Starting Qwen game evaluations."
Invoke-CommandGroup @(
    @{Command="game-eval-qwen-mcq"; Key=1;
      Stdout=(Join-Path $gameResultDir "qwen3.5-122b_mcq_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "qwen3.5-122b_mcq_eval.stderr.log")},
    @{Command="game-eval-qwen-open"; Key=2;
      Stdout=(Join-Path $gameResultDir "qwen3.5-122b_open_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "qwen3.5-122b_open_eval.stderr.log")},
    @{Command="game-eval-qwen-play"; Key=3;
      Stdout=(Join-Path $gameResultDir "qwen3.5-122b_play_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "qwen3.5-122b_play_eval.stderr.log")}
)
Assert-CsvRows (Join-Path $gameResultDir "qwen3.5-122b_mcq_balanced50.csv") 150
Assert-CsvRows (Join-Path $gameResultDir "qwen3.5-122b_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
Assert-CsvRows (Join-Path $gameResultDir "qwen3.5-122b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv") 3000
& $gitBash $pipeline "game-figures-qwen"
if ($LASTEXITCODE -ne 0) { throw "Qwen game figure generation failed." }

Write-Status "Qwen games complete; starting Qwen solo evaluations before GPT-OSS."
Invoke-CommandGroup @(
    @{Command="solo-eval-qwen-mcq"; Key=2;
      Stdout=(Join-Path $soloResultDir "qwen3.5-122b_mcq_eval.stdout.log");
      Stderr=(Join-Path $soloResultDir "qwen3.5-122b_mcq_eval.stderr.log")},
    @{Command="solo-eval-qwen-interactive"; Key=3;
      Stdout=(Join-Path $soloResultDir "qwen3.5-122b_interactive_eval.stdout.log");
      Stderr=(Join-Path $soloResultDir "qwen3.5-122b_interactive_eval.stderr.log")}
)
Assert-CsvRows (Join-Path $soloResultDir "qwen3.5-122b_mcq_balanced50.csv") 150
Assert-CsvRows (Join-Path $soloResultDir "qwen3.5-122b_interactive_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
& $gitBash $pipeline "solo-analyze"
if ($LASTEXITCODE -ne 0) { throw "Shared solo analysis failed after Qwen." }

Write-Status "Starting GPT-OSS game evaluations."
Invoke-CommandGroup @(
    @{Command="game-eval-gptoss-mcq"; Key=1;
      Stdout=(Join-Path $gameResultDir "gpt-oss-120b_mcq_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "gpt-oss-120b_mcq_eval.stderr.log")},
    @{Command="game-eval-gptoss-open"; Key=2;
      Stdout=(Join-Path $gameResultDir "gpt-oss-120b_open_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "gpt-oss-120b_open_eval.stderr.log")},
    @{Command="game-eval-gptoss-play"; Key=3;
      Stdout=(Join-Path $gameResultDir "gpt-oss-120b_play_eval.stdout.log");
      Stderr=(Join-Path $gameResultDir "gpt-oss-120b_play_eval.stderr.log")}
)
Assert-CsvRows (Join-Path $gameResultDir "gpt-oss-120b_mcq_balanced50.csv") 150
Assert-CsvRows (Join-Path $gameResultDir "gpt-oss-120b_open_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
Assert-CsvRows (Join-Path $gameResultDir "gpt-oss-120b_play_balanced50_scripted_r2_rounds5_mem1_fallback-paid-gpt-5.6-luna.csv") 3000
& $gitBash $pipeline "game-figures-gptoss"
if ($LASTEXITCODE -ne 0) { throw "GPT-OSS game figure generation failed." }

Write-Status "GPT-OSS games complete; starting GPT-OSS solo evaluations."
Invoke-CommandGroup @(
    @{Command="solo-eval-gptoss-mcq"; Key=2;
      Stdout=(Join-Path $soloResultDir "gpt-oss-120b_mcq_eval.stdout.log");
      Stderr=(Join-Path $soloResultDir "gpt-oss-120b_mcq_eval.stderr.log")},
    @{Command="solo-eval-gptoss-interactive"; Key=3;
      Stdout=(Join-Path $soloResultDir "gpt-oss-120b_interactive_eval.stdout.log");
      Stderr=(Join-Path $soloResultDir "gpt-oss-120b_interactive_eval.stderr.log")}
)
Assert-CsvRows (Join-Path $soloResultDir "gpt-oss-120b_mcq_balanced50.csv") 150
Assert-CsvRows (Join-Path $soloResultDir "gpt-oss-120b_interactive_balanced50_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv") 150
& $gitBash $pipeline "solo-analyze"
if ($LASTEXITCODE -ne 0) { throw "Shared solo analysis failed after GPT-OSS." }
& $gitBash $pipeline "comparison-figures"
if ($LASTEXITCODE -ne 0) { throw "Cross-domain/model comparison figures failed." }

Write-Status "All requested model evaluation sequences are complete."
