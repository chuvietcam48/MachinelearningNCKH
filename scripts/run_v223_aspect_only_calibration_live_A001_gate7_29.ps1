$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Runner = Join-Path $Repo "src\amazon_v5_rebuild\34_gate7_29_v223_aspect_only_calibration_runner.py"
$Approval = Join-Path $Repo "outputs\amazon_v5_rebuild\annotation\v1_3_calibration_v2\v223_aspect_only_calibration_live_approval_A001_gate7_29.json"
$Workspace = Join-Path $env:LOCALAPPDATA "Temp\MachineLearningNCKH\v223_aspect_only_workspace"

& $Python $Runner --repo-root $Repo --execute --workspace-root $Workspace --approval-file $Approval
