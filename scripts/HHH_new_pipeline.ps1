param(
    [Parameter(Position = 0)]
    [string] $Command = "list"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$launcher = Join-Path $repoRoot "scripts/HHH_new_pipeline.sh"

$gitBashCandidates = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files\Git\usr\bin\bash.exe"
)
$gitBash = $gitBashCandidates |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1

if (-not $gitBash) {
    throw "Git Bash was not found. Install Git for Windows or run the .sh launcher in an existing Bash shell."
}

Push-Location $repoRoot
try {
    & $gitBash $launcher $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
