param(
    [Parameter(Mandatory = $true)] [string] $Command,
    [Parameter(Mandatory = $true)] [int]    $Key,
    [Parameter(Mandatory = $true)] [string] $Log,
    [string] $ResultDir = "results/HHH_new_paper_bothAI_anthropic_batch_20260826_01_balanced50"
)

# Minimal detached runner for a single HHH_new_pipeline.sh eval subcommand on a
# chosen gateway key. Used to start a long job on a key that happens to be
# free, out of the order the wave scripts would otherwise pick.
#
# The wave scripts adopt rather than relaunch anything already running for the
# same config, so starting a job this way does not collide with them.
#
#   powershell -File scripts\HHH_new_run_one_eval.ps1 -Command game-eval-gptoss-play -Key 1 -Log gpt-oss-120b_play_balanced50

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$gitBash  = "C:\Program Files\Git\bin\bash.exe"
$pipeline = "scripts/HHH_new_pipeline.sh"
$dir      = Join-Path $repoRoot $ResultDir
$chainLog = Join-Path $repoRoot "results/HHH_new_full_chain.log"
New-Item -ItemType Directory -Path $dir -Force | Out-Null

function Write-Chain([string] $Message) {
    $line = "$(Get-Date -Format s) [one] $Message"
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

$env:GATEWAY_KEY_INDEX = [string]$Key
Write-Chain "start: $Command on key $Key (PID $PID)."

# PS 5.1 creates a NEW file as UTF-16LE for >> / 1>> / 2>>, which reads as binary
# to grep and tail. Write UTF-8 explicitly instead.
& $gitBash $pipeline $Command 2>&1 | Out-File -LiteralPath (Join-Path $dir "$Log.stdout.log") -Encoding utf8 -Append

Write-Chain "finished: $Command (native exit code '$LASTEXITCODE'); verify by row count."
