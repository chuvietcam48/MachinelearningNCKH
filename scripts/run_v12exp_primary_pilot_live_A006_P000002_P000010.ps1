param(
    [string]$RepoRoot = "",
    [string]$PythonExe = "",
    [string]$WorkspaceRoot = "$env:LOCALAPPDATA\MLNCKH_GeminiPilotWorkspace"
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if (-not $PythonExe) {
    $candidatePython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidatePython) {
        $PythonExe = $candidatePython
    } else {
        $PythonExe = "python"
    }
}

if ($WorkspaceRoot -match "(?i)(^|[\\/])OneDrive([\\/]|$)") {
    throw "WorkspaceRoot must be outside OneDrive: $WorkspaceRoot"
}

if (-not $env:GEMINI_API_KEY) {
    throw "GEMINI_API_KEY is not set in this PowerShell session."
}

$approvalDir = Join-Path $WorkspaceRoot "approvals"
New-Item -ItemType Directory -Force -Path $approvalDir | Out-Null
$approvalFile = Join-Path $approvalDir "live_approval_V12EXP_PRIMARY_V3_P000002_P000010_A006.json"

$approval = [ordered]@{
    execution_mode = "LIVE"
    campaign_run_id = "V12EXP_PRIMARY_V3"
    batch_start = "P000002"
    batch_end = "P000010"
    attempt_id = "A006"
    ack_live_provider = $true
    approval_id = "LIVE_APPROVAL_V12EXP_PRIMARY_V3_P000002_P000010_A006_OPERATOR"
    max_provider_requests = 9
    max_concurrency = 1
    rate_limit_requests_per_minute = 1
    cost_ceiling_usd = 5
    stop_if_cost_unavailable = $true
    audit_rule = "Audit all 80 pilot outputs before release into expanded semantic corpus."
    operator_note = "A006 continues after A005 repeated evidence length violation; provider schema now includes a simple <=7-token evidence pattern."
}

if (Test-Path -LiteralPath $approvalFile) {
    throw "Approval file already exists; create the next attempt ID instead of reusing A006: $approvalFile"
}

$approvalJson = $approval | ConvertTo-Json -Depth 5
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($approvalFile, $approvalJson, $utf8NoBom)

& $PythonExe (Join-Path $RepoRoot "src\amazon_v5_rebuild\16_v12exp_primary_pilot_live_executor.py") `
    --repo-root $RepoRoot `
    --workspace-root $WorkspaceRoot `
    --approval-file $approvalFile `
    --execute-live

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
