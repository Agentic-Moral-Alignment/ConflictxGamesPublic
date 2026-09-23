param(
    [int] $PollSeconds = 60,
    [int] $Key = 2,
    # Which phases this instance runs, as a string of digits ("1234"). Two
    # instances on different keys can share the programme: the adopt-if-running
    # and skip-if-complete checks make an overlap safe, but splitting the phases
    # keeps them off each other's work. A digit string rather than an int array
    # because Task Scheduler drops the commas out of an -Argument list, which
    # silently turns "1,2,3,4" into the single value 1234 and skips everything.
    [string] $Phases = '12345678',
    # Which focal model this instance runs: 'gemma', 'gptoss', or '' for both.
    # Splitting the programme by PHASE puts whatever phases are live on every
    # key at once, and when those phases happen to be the same model all three
    # keys queue against one backend -- which is slower than a single key, not
    # faster. Splitting by MODEL instead gives each key its own backend. With
    # two focal models that means two non-contending lanes; a third key can only
    # help by taking a third model.
    [string] $Model = ''
)

$MODEL_SLUG = @{ 'gemma' = 'gemma-4-31B-it'; 'gptoss' = 'gpt-oss-120b' }
if ($Model -and -not $MODEL_SLUG.ContainsKey($Model)) {
    throw "Unknown -Model '$Model'; expected one of: $($MODEL_SLUG.Keys -join ', ')"
}

# The state-probe programme, in the order it was asked for.
#
#   PHASE 1  states, no stake        - the 5 mem-1 memory states on their own
#   PHASE 2  states x 3 stakes       - the same states under payoff / alignment /
#                                      usefulness pressure
#   PHASE 3  open x 3 stakes         - the same three stakes in open-ended mode
#   PHASE 4  states x visibility     - added: the same states with the decision
#                                      stated as private or as revealed, which
#                                      tests whether the previous round matters
#                                      because the counterparty saw it
#
# Every phase runs Gemma and GPT-OSS only. Each job is skipped if its output is
# already complete and adopted if it is already running, so the chain is safe to
# restart at any point.
#
# Launch detached:  schtasks /Run /TN "HHH_states_program"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$gameDir  = Join-Path $repoRoot "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
$expected = 150     # unique scenario|seat cases; state runs carry 5 rows each

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [prog] $Message"
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

function Get-LiveEval([string] $ConfigMatch, $Solo = $false) {
    # $Solo is deliberately untyped: game runs carry no Solo key, and passing the
    # resulting $null to a [bool] parameter throws in PowerShell 5.1. The throw
    # left $live null, so the adoption check silently failed and a second copy of
    # an already-running job was launched against the same output file.
    $module = if ($Solo) { 'conflictxgame\.solo\.evaluate_models' } else { 'conflictxgame\.games\.evaluate_games' }
    return @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match $module -and
                            $_.CommandLine -match $ConfigMatch }) | Select-Object -First 1
}

$soloDir = Join-Path $repoRoot "results/HHH_new_solo_20260826_01_balanced50"

function Invoke-Run($Run) {
    # PowerShell 5.1 does not accept `if` as an expression in argument position,
    # so this must be a plain statement; inlining it made every path empty and
    # every run report 0/150 without ever launching.
    $dir = $gameDir
    if ($Run.Solo) { $dir = $soloDir }
    $outPath = Join-Path $dir $Run.Output
    $done = Get-ResultCaseCount $outPath
    if ($done -ge $expected) {
        Write-Chain "$($Run.Command) already complete ($done/$expected cases); skipping."
        return 'complete'
    }
    $live = Get-LiveEval $Run.Match $Run.Solo
    if ($null -ne $live) {
        # Move on rather than block. Waiting here is what left a key idle: a lane
        # that reached a job another lane was already running sat on it for the
        # rest of that job instead of taking the next piece of outstanding work.
        # The caller revisits skipped jobs on a later pass, so nothing is lost.
        Write-Chain "$($Run.Command) already running as PID $($live.ProcessId); skipping for now."
        return 'busy'
    }
    else {
        # Claim the job before launching. Get-LiveEval alone is a check-then-act
        # race: three lanes starting together all saw "not running" and all three
        # launched the same job against the same output file. CreateNew is
        # atomic, so exactly one lane wins the claim and the others move on.
        $claim = $outPath + '.claim'
        $claimStream = $null
        try {
            $claimStream = [System.IO.File]::Open(
                $claim, [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        }
        catch {
            # A claim already exists. If nothing is actually running it is stale
            # -- a lane killed mid-run -- so clear it and let the next pass retry.
            if ($null -eq (Get-LiveEval $Run.Match $Run.Solo)) {
                $age = (Get-Date) - (Get-Item -LiteralPath $claim -EA SilentlyContinue).LastWriteTime
                if ($age.TotalMinutes -gt 5) {
                    Write-Chain "$($Run.Command) claim is stale ($([int]$age.TotalMinutes)m, no process); clearing."
                    Remove-Item -LiteralPath $claim -Force -EA SilentlyContinue
                }
            }
            Write-Chain "$($Run.Command) claimed by another lane; skipping for now."
            return 'busy'
        }
        $prior = $env:GATEWAY_KEY_INDEX
        try {
            $env:GATEWAY_KEY_INDEX = [string]$Key
            $p = Start-Process -FilePath $gitBash -ArgumentList @($pipeline, $Run.Command) `
                -WorkingDirectory $repoRoot `
                -RedirectStandardOutput (Join-Path $dir "$($Run.Log).stdout.log") `
                -RedirectStandardError  (Join-Path $dir "$($Run.Log).stderr.log") `
                -WindowStyle Hidden -PassThru
            Write-Chain "launched $($Run.Command) PID $($p.Id) on key $Key."
            $p.WaitForExit()
        }
        finally {
            if ($null -eq $prior) { Remove-Item Env:GATEWAY_KEY_INDEX -ErrorAction SilentlyContinue }
            else { $env:GATEWAY_KEY_INDEX = $prior }
            # Release the claim whatever happened, including on a throw: a claim
            # left behind would block that job on every later pass, in this lane
            # and every other one.
            if ($null -ne $claimStream) { $claimStream.Dispose() }
            Remove-Item -LiteralPath $claim -Force -ErrorAction SilentlyContinue
        }
    }
    $n = Get-ResultCaseCount $outPath
    if ($n -lt $expected) {
        Write-Chain "FAIL $($Run.Command): $n/$expected cases in $($Run.Output)"
        return 'failed'
    }
    Write-Chain "OK   $($Run.Command): $n/$expected cases."
    return 'ran'
}

$JT = '_user-paid-gpt-5.6-luna_judge-paid-gpt-5.6-luna.csv'

function Run-Spec($cmd, $log, $match, $out) {
    return @{ Command = $cmd; Log = $log; Match = $match; Output = $out }
}

# GPT-OSS first in every phase: it runs the same work in a fraction of the time,
# so each phase yields a readable result long before Gemma finishes it.
$PHASE1 = @(
    (Run-Spec 'game-eval-gptoss-mcq-states' 'gpt-oss-120b_mcq_states' 'eval-mcq-states_focal-gpt-oss' 'gpt-oss-120b_mcq_balanced50_states.csv'),
    (Run-Spec 'game-eval-mcq-states' 'gemma-4-31B-it_mcq_states' 'eval-mcq-states_focal-gemma' 'gemma-4-31B-it_mcq_balanced50_states.csv'),
    (Run-Spec 'game-eval-gptoss-mcq-states-steer' 'gpt-oss-120b_mcq_states_steer' 'eval-mcq-states-steer_focal-gpt-oss' 'gpt-oss-120b_mcq_balanced50_states-steer.csv'),
    (Run-Spec 'game-eval-mcq-states-steer' 'gemma-4-31B-it_mcq_states_steer' 'eval-mcq-states-steer_focal-gemma' 'gemma-4-31B-it_mcq_balanced50_states-steer.csv')
)
$PHASE2 = @()
foreach ($arm in @('payoff', 'alignment', 'usefulness')) {
    $PHASE2 += (Run-Spec "game-eval-gptoss-mcq-states-stakes-$arm" "gpt-oss-120b_mcq_states_stakes_$arm" "eval-mcq-states-stakes-$arm`_focal-gpt-oss" "gpt-oss-120b_mcq_balanced50_states_stakes-$arm.csv")
    $PHASE2 += (Run-Spec "game-eval-mcq-states-stakes-$arm" "gemma-4-31B-it_mcq_states_stakes_$arm" "eval-mcq-states-stakes-$arm`_focal-gemma" "gemma-4-31B-it_mcq_balanced50_states_stakes-$arm.csv")
}
# Open now carries the same five memory states as mcq, so phase 3 runs
# open-states rather than plain open: the no-stake baseline first, then each
# stake. mcq-states and open-states differ only in response format, which is
# what makes the pair comparable.
$PHASE3 = @(
    (Run-Spec 'game-eval-gptoss-open-states' 'gpt-oss-120b_open_states' 'eval-open-states_focal-gpt-oss' "gpt-oss-120b_open_balanced50_states$JT"),
    (Run-Spec 'game-eval-open-states' 'gemma-4-31B-it_open_states' 'eval-open-states_focal-gemma' "gemma-4-31B-it_open_balanced50_states$JT"),
    (Run-Spec 'game-eval-gptoss-open-states-steer' 'gpt-oss-120b_open_states_steer' 'eval-open-states-steer_focal-gpt-oss' "gpt-oss-120b_open_balanced50_states-steer$JT"),
    (Run-Spec 'game-eval-open-states-steer' 'gemma-4-31B-it_open_states_steer' 'eval-open-states-steer_focal-gemma' "gemma-4-31B-it_open_balanced50_states-steer$JT")
)
foreach ($arm in @('payoff', 'alignment', 'usefulness')) {
    $PHASE3 += (Run-Spec "game-eval-gptoss-open-states-stakes-$arm" "gpt-oss-120b_open_states_stakes_$arm" "eval-open-states-stakes-$arm`_focal-gpt-oss" "gpt-oss-120b_open_balanced50_states-stakes-$arm$JT")
    $PHASE3 += (Run-Spec "game-eval-open-states-stakes-$arm" "gemma-4-31B-it_open_states_stakes_$arm" "eval-open-states-stakes-$arm`_focal-gemma" "gemma-4-31B-it_open_balanced50_states-stakes-$arm$JT")
}
$PHASE4 = @()
foreach ($arm in @('unobserved', 'observed')) {
    $PHASE4 += (Run-Spec "game-eval-gptoss-mcq-states-$arm" "gpt-oss-120b_mcq_states_$arm" "eval-mcq-states-$arm`_focal-gpt-oss" "gpt-oss-120b_mcq_balanced50_states_$arm.csv")
    $PHASE4 += (Run-Spec "game-eval-mcq-states-$arm" "gemma-4-31B-it_mcq_states_$arm" "eval-mcq-states-$arm`_focal-gemma" "gemma-4-31B-it_mcq_balanced50_states_$arm.csv")
}

$SOLO = @()
foreach ($arm in @('alignment', 'usefulness')) {
    foreach ($m in @(
        @{ s='gpt-oss-120b'; c='gptoss-' },
        @{ s='gemma-4-31B-it'; c='' }
    )) {
        $SOLO += @{ Command = "solo-eval-$($m.c)mcq-stakes-$arm"; Log = "$($m.s)_mcq_stakes_$arm";
                    Match = "eval-mcq-stakes-$arm`_focal-$($m.s)"; Solo = $true;
                    Output = "$($m.s)_mcq_balanced50_stakes-$arm.csv" }
        $SOLO += @{ Command = "solo-eval-$($m.c)interactive-stakes-$arm"; Log = "$($m.s)_interactive_stakes_$arm";
                    Match = "eval-interactive-stakes-$arm`_focal-$($m.s)"; Solo = $true;
                    Output = "$($m.s)_interactive_balanced50_stakes-$arm$JT" }
    }
}

# Single-state game stakes: the same three criteria without any history at all.
# Paired with phase 2 (the same stakes over the five memory states) this
# separates what the stake does from what the stake plus a remembered round do.
$SINGLE = @()
foreach ($arm in @('payoff', 'alignment', 'usefulness')) {
    $SINGLE += (Run-Spec "game-eval-gptoss-mcq-stakes-$arm" "gpt-oss-120b_mcq_stakes_$arm" "eval-mcq-stakes-$arm`_focal-gpt-oss" "gpt-oss-120b_mcq_balanced50_stakes-$arm.csv")
    $SINGLE += (Run-Spec "game-eval-mcq-stakes-$arm" "gemma-4-31B-it_mcq_stakes_$arm" "eval-mcq-stakes-$arm`_focal-gemma" "gemma-4-31B-it_mcq_balanced50_stakes-$arm.csv")
    $SINGLE += (Run-Spec "game-eval-gptoss-open-stakes-$arm" "gpt-oss-120b_open_stakes_$arm" "eval-open-stakes-$arm`_focal-gpt-oss" "gpt-oss-120b_open_balanced50_stakes-$arm$JT")
    $SINGLE += (Run-Spec "game-eval-open-stakes-$arm" "gemma-4-31B-it_open_stakes_$arm" "eval-open-stakes-$arm`_focal-gemma" "gemma-4-31B-it_open_balanced50_stakes-$arm$JT")
}

# Single-state visibility, moved here from HHH_run_visibility_chain so that the
# -Model filter covers it too. Run as its own chain it picked its own model
# order, which put it on the same backend as a model lane and turned two keys
# into one queue.
$VIS = @()
foreach ($arm in @('unobserved', 'observed')) {
    $VIS += (Run-Spec "game-eval-mcq-$arm" "gemma-4-31B-it_mcq_$arm" "eval-mcq-$arm`_focal-gemma" "gemma-4-31B-it_mcq_balanced50_$arm.csv")
    $VIS += (Run-Spec "game-eval-open-$arm" "gemma-4-31B-it_open_$arm" "eval-open-$arm`_focal-gemma" "gemma-4-31B-it_open_balanced50_$arm$JT")
    $VIS += (Run-Spec "game-eval-gptoss-mcq-$arm" "gpt-oss-120b_mcq_$arm" "eval-mcq-$arm`_focal-gpt-oss" "gpt-oss-120b_mcq_balanced50_$arm.csv")
    $VIS += (Run-Spec "game-eval-gptoss-open-$arm" "gpt-oss-120b_open_$arm" "eval-open-$arm`_focal-gpt-oss" "gpt-oss-120b_open_balanced50_$arm$JT")
}

# Phase 8: the abstract-rival stake, inside the game. The stake prompts were
# never word-identical across domains -- the game names "the other participant",
# solo names "other deployed systems" -- so the solo/game stake contrast
# confounded domain with rival-specificity. These runs use the SOLO prompt file
# in the game, giving three cells of a 2x2 (solo x specific cannot exist), so
# game/abstract vs solo/abstract isolates domain and game/abstract vs
# game/specific isolates the rival. Payoff is absent: it has no solo twin.
$ABSTRACT = @()
foreach ($arm in @('alignment', 'usefulness')) {
    $ABSTRACT += (Run-Spec "game-eval-gptoss-mcq-stakes-abstract-$arm" "gpt-oss-120b_mcq_stakes_abstract_$arm" "eval-mcq-stakes-abstract-$arm`_focal-gpt-oss" "gpt-oss-120b_mcq_balanced50_stakes-abstract-$arm.csv")
    $ABSTRACT += (Run-Spec "game-eval-mcq-stakes-abstract-$arm" "gemma-4-31B-it_mcq_stakes_abstract_$arm" "eval-mcq-stakes-abstract-$arm`_focal-gemma" "gemma-4-31B-it_mcq_balanced50_stakes-abstract-$arm.csv")
    $ABSTRACT += (Run-Spec "game-eval-gptoss-open-stakes-abstract-$arm" "gpt-oss-120b_open_stakes_abstract_$arm" "eval-open-stakes-abstract-$arm`_focal-gpt-oss" "gpt-oss-120b_open_balanced50_stakes-abstract-$arm$JT")
    $ABSTRACT += (Run-Spec "game-eval-open-stakes-abstract-$arm" "gemma-4-31B-it_open_stakes_abstract_$arm" "eval-open-stakes-abstract-$arm`_focal-gemma" "gemma-4-31B-it_open_balanced50_stakes-abstract-$arm$JT")
}

Write-Chain "=== states programme start (PID $PID, key $Key, phases $Phases, model $(if ($Model) { $Model } else { 'both' })) ==="
$plan = @()
foreach ($phase in @(
    @{ N = 1; Name = '1 - states, no stake'; Runs = $PHASE1 },
    @{ N = 2; Name = '2 - states x 3 stakes'; Runs = $PHASE2 },
    @{ N = 3; Name = '3 - open states, then open states x 3 stakes'; Runs = $PHASE3 },
    @{ N = 4; Name = '4 - states x visibility'; Runs = $PHASE4 },
    @{ N = 5; Name = '5 - single-state game x 3 stakes'; Runs = $SINGLE },
    @{ N = 6; Name = '6 - solo x 2 stakes'; Runs = $SOLO },
    @{ N = 7; Name = '7 - single-state visibility'; Runs = $VIS },
    @{ N = 8; Name = '8 - abstract-rival stake in the game'; Runs = $ABSTRACT }
)) {
    if (-not $Phases.Contains([string] $phase.N)) { continue }
    # Every Output begins with the model slug, so that is the model filter.
    $runs = @($phase.Runs | Where-Object {
        (-not $Model) -or $_.Output.StartsWith($MODEL_SLUG[$Model])
    })
    if ($runs.Count -eq 0) { continue }
    $plan += ,@{ Name = $phase.Name; Runs = $runs }
}

# Work-steal rather than march through a fixed list. A lane that reaches a job
# another lane is already running skips it and takes the next outstanding one,
# then comes back for the skipped jobs on a later pass. Without this a lane
# either blocked on someone else's job or ran out of its own phases and left a
# key idle while another lane still had hours of queue. The pass repeats until
# nothing is left incomplete, so skipping never loses work.
$pass = 0
while ($true) {
    $pass++
    $busy = 0
    $worked = 0
    foreach ($phase in $plan) {
        foreach ($run in $phase.Runs) {
            $status = Invoke-Run $run
            if ($status -eq 'busy') { $busy++ }
            elseif ($status -eq 'ran' -or $status -eq 'failed') { $worked++ }
        }
    }
    if ($busy -eq 0) {
        Write-Chain "=== pass $pass finished with nothing outstanding ==="
        break
    }
    if ($worked -eq 0) {
        # Everything left belongs to another lane and is still running. Wait for
        # one of them to land rather than spinning through the plan.
        Write-Chain "pass ${pass}: $busy job(s) held by other lanes; waiting $PollSeconds s."
        Start-Sleep -Seconds $PollSeconds
    }
}
Write-Chain "=== states programme complete ==="
