#!/usr/bin/env python
"""Gate 7.18 v1.5.1 candidate-ID calibration runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_RUNNER_PATH = Path(__file__).with_name("24_gate7_17_v15_candidate_id_calibration_runner.py")


def load_v15_runner():
    spec = importlib.util.spec_from_file_location("gate7_17_v15_candidate_id_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load v1.5 runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v15 = load_v15_runner()
runner = v15.runner


def render_prompt_v151(items):
    prompt = v15.render_prompt_v15(items)
    hard_stop = """

FINAL OUTPUT CHECKLIST - REQUIRED:
1. Polarity1 must be exactly "Positive", "Negative", or "NotApplicable".
2. Polarity2 must be exactly "Positive", "Negative", "NotApplicable", or null.
3. NEVER write "Mixed" in Polarity1 or Polarity2.
4. Mixed sentiment is represented ONLY by Review_Mixed_Flag=true plus separate Positive/Negative slots.
5. If one aspect contains both positive and negative sentiment, use two slots with the same Aspect and opposite polarities, never Polarity="Mixed".
6. If you cannot find a separate negative evidence candidate, choose the dominant sentiment and set Review_Mixed_Flag=false.
"""
    return prompt + hard_stop


def main() -> None:
    runner.POLICY_VERSION = "semantic_policy_v1_5_1_candidate_id"
    runner.CAMPAIGN_ID = "V151CIDCALIBRATION_FRESH_V1"
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
    runner.render_prompt = render_prompt_v151
    runner.apply_policy_normalization = v15.normalize_v15
    runner.main()


if __name__ == "__main__":
    main()
