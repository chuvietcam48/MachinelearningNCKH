$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Repo ".venv\Scripts\python.exe"
$Runner = Join-Path $Repo "src\amazon_v5_rebuild\35_gate7_30_v224_aspect_only_calibration_runner.py"
$Approval = Join-Path $Repo "outputs\amazon_v5_rebuild\annotation\v1_3_calibration_v2\v224_aspect_only_calibration_live_approval_A001_gate7_30.json"
$Workspace = Join-Path $env:LOCALAPPDATA "Temp\MachineLearningNCKH\v224_aspect_only_workspace"

& $Python $Runner --repo-root $Repo --execute --workspace-root $Workspace --approval-file $Approval
