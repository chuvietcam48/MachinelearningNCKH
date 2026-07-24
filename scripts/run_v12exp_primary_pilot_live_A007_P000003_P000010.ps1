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

$repairLedger = Join-Path $RepoRoot "outputs\amazon_v5_rebuild\features_semantic_b3\V12EXP_PRIMARY_V3\human_evidence_repairs\P000002\A006\human_evidence_repair_ledger.json"
if (-not (Test-Path -LiteralPath $repairLedger)) {
    throw "Required P000002 evidence-only repair ledger is missing: $repairLedger"
}
$repair = Get-Content -LiteralPath $repairLedger -Raw | ConvertFrom-Json
if ($repair.status -ne "RESOLVED_BY_DOCUMENTED_HUMAN_EVIDENCE_ONLY_REPAIR" -or
    $repair.validator_result_on_repaired_output -ne "PASS" -or
    $repair.semantic_fields_changed -ne $false -or
    $repair.field_changed -ne "Evidence1 only") {
    throw "P000002 repair ledger is not in the required resolved/pass/evidence-only state."
}

$approvalDir = Join-Path $WorkspaceRoot "approvals"
New-Item -ItemType Directory -Force -Path $approvalDir | Out-Null
$approvalFile = Join-Path $approvalDir "live_approval_V12EXP_PRIMARY_V3_P000003_P000010_A007.json"

$approval = [ordered]@{
    execution_mode = "LIVE"
    campaign_run_id = "V12EXP_PRIMARY_V3"
    batch_start = "P000003"
    batch_end = "P000010"
    attempt_id = "A007"
    ack_live_provider = $true
    approval_id = "LIVE_APPROVAL_V12EXP_PRIMARY_V3_P000003_P000010_A007_OPERATOR"
    max_provider_requests = 8
    max_concurrency = 1
    rate_limit_requests_per_minute = 1
    cost_ceiling_usd = 5
    stop_if_cost_unavailable = $true
    audit_rule = "Audit all 80 pilot outputs before release into expanded semantic corpus."
    operator_note = "A007 resumes remaining pilot batches after P000001/A004 pass and P000002 final documented evidence-only repair pass."
}

if (Test-Path -LiteralPath $approvalFile) {
    throw "Approval file already exists; create the next attempt ID instead of reusing A007: $approvalFile"
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
