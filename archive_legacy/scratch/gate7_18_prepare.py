import hashlib
import json
import pathlib


repo = pathlib.Path(__file__).resolve().parents[1]
out_dir = repo / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
approval_path = out_dir / "v151_candidate_id_calibration_live_approval_A001_gate7_18.json"
ps_path = repo / "scripts" / "run_v151_candidate_id_calibration_live_A001_gate7_18.ps1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path: pathlib.Path, data: bytes) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


approval = {
    "approval_id": "GATE7_18_V151CIDCALIBRATION_A001_OPERATOR_APPROVAL",
    "attempt_id": "A001",
    "execution_mode": "LIVE",
    "campaign_run_id": "V151CIDCALIBRATION_FRESH_V1",
    "policy_version": "semantic_policy_v1_5_1_candidate_id",
    "batch_start": "V14CAL_B001",
    "batch_end": "V14CAL_B012",
    "ack_live_provider": True,
    "max_provider_requests": 12,
    "max_concurrency": 1,
    "no_blind_retry": True,
    "human_repair_allowed": False,
    "scope": "Gate 7.18 v1.5.1 candidate-ID calibration over the same 96-case manifest.",
    "manifest_sha256": "2437e322c64c113bf7bb49416551ddd49dc52dcecb7dbe89ed5c9572d6938c48",
    "human_adjudication_sha256": "f70b57612f30651a2242195452a46c610629da82604ee18206b10729128f30a5",
    "operator_note": "Versioned prompt compliance fix: Polarity must not be Mixed.",
}

ps = '''$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"
$WorkspaceRoot = Join-Path ([System.IO.Path]::GetTempPath()) "MachineLearningNCKH\\v13_calibration_workspace"
$Approval = Join-Path $Repo "outputs\\amazon_v5_rebuild\\annotation\\v1_3_calibration_v2\\v151_candidate_id_calibration_live_approval_A001_gate7_18.json"

Set-Location $Repo
& $Python "src\\amazon_v5_rebuild\\25_gate7_18_v151_candidate_id_calibration_runner.py" `
  --execute-live `
  --workspace-root $WorkspaceRoot `
  --approval-file $Approval
'''

write_new(approval_path, json.dumps(approval, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
write_new(ps_path, ps.encode("utf-8"))
print(
    json.dumps(
        {
            "status": "GATE7_18_PREPARED",
            "approval_sha256": sha256_file(approval_path),
            "script_sha256": sha256_file(ps_path),
            "provider_calls_made": False,
        },
        indent=2,
    )
)
