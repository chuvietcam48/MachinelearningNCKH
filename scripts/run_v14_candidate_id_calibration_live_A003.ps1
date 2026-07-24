$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$WorkspaceRoot = Join-Path ([System.IO.Path]::GetTempPath()) "MachineLearningNCKH\v13_calibration_workspace"
$Approval = Join-Path $Repo "outputs\amazon_v5_rebuild\annotation\v1_3_calibration_v2\v14_candidate_id_calibration_live_approval_A003.json"

Set-Location $Repo
& $Python "src\amazon_v5_rebuild\22_gate7_11_v14_candidate_id_calibration_runner.py" `
  --execute-live `
  --workspace-root $WorkspaceRoot `
  --approval-file $Approval
