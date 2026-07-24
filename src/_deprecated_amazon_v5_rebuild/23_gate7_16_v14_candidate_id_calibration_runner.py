#!/usr/bin/env python
"""Gate 7.16 v1.4 candidate-ID calibration runner.

This wrapper reuses the approved Gate 7.11 candidate-ID resolver, schema,
provider projection, and validator implementation. It only swaps in the
Gate 7.16 execution manifest and human adjudication table.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_RUNNER_PATH = Path(__file__).with_name("22_gate7_11_v14_candidate_id_calibration_runner.py")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_candidate_id_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_base_runner()
    runner.POLICY_VERSION = "semantic_policy_v1_4_candidate_id"
    runner.CAMPAIGN_ID = "V14CIDCALIBRATION_FRESH_V1"
    runner.EXPECTED_BATCHES = [f"V14CAL_B{i:03d}" for i in range(1, 13)]
    runner.EXPECTED_ITEM_COUNT = 96
    runner.EXPECTED_BATCH_SIZE = 8
    runner.MANIFEST_PATH = runner.OUT_DIR / "gate7_16_v14_candidate_id_execution_manifest_96.json"
    runner.HUMAN_ADJ_PATH = runner.OUT_DIR / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
    runner.LOCKED_HASHES = {
        "manifest": "2437e322c64c113bf7bb49416551ddd49dc52dcecb7dbe89ed5c9572d6938c48",
        "human_adjudication": "f70b57612f30651a2242195452a46c610629da82604ee18206b10729128f30a5",
        "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
    }
    runner.main()


if __name__ == "__main__":
    main()
