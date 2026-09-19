#!/usr/bin/env python
"""Gate 7.7 static payload-contract materialization and verification.

This script never calls Gemini or any provider. It locks and verifies immutable
provider payloads for the v3 expansion campaigns under the frozen v1.2
taxonomy/prompt/schema contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd


ITEMS_PER_REQUEST = 8
PRIMARY_CAMPAIGN_ID = "V12EXP_PRIMARY_V3"
ENRICH_CAMPAIGN_ID = "V12EXP_ENRICH_V3"
ATTEMPT_ID = "A001"

EXPECTED_CANONICAL_8_ITEM_SCHEMA_SHA256 = "0583cb57466f6cbd454a45d536dec66c7cff14c880b8470fb49eca68dfb9814c"
EXPECTED_PROVIDER_PROJECTION_8_ITEM_SCHEMA_SHA256 = "003371c1939ddf14a5f453889abc193775b6bb87c96f95a65c0f7a67a2c40f37"

LOCKED_HASHES = {
    "primary_review_manifest_sha256": "f1ae749ed253d2900191457a118f4cffbf9ac48e90d63c311982a3de91367527",
    "enrichment_review_manifest_sha256": "e4d255ef9d643843a934214f44fd63fee564a309785342d2a900a360289acc75",
    "primary_batch_manifest_sha256": "d47da37311b1b89e148e8f382d22e0f89ef94ce81d9fbe62a05dc221818cf937",
    "enrichment_batch_manifest_sha256": "ad6cb83f710248903bd0f186ac3136f4cf269c3d9fec881f040db71b44bf5601",
    "gate7_6_v3_cohort_report_sha256": "af3714228eb4bfc5647f78b6387056fd57ae9a7bfe8f051425e476304ee4c38b",
    "episode_annotation_membership_v3_sha256": "d0beb86a720130575e2ede90b1b90df221c8d042c2b640743653cc4123953b25",
    "prompt_template_v1_2_sha256": "f2cf7512c65860bdc6f2cb07b0cb55781262c39df6d8a196671b69b951737860",
    "taxonomy_v1_2_sha256": "8cde93d06f446c9d776b4935786c81d76a7e5a16d4cdde257ac095ea80ca52f9",
    "canonical_schema_file_v1_6_sha256": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
}

REL_PATHS = {
    "prompt_template": Path("outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_2.md"),
    "taxonomy": Path("outputs/amazon_v5_rebuild/annotation/taxonomy_v1_2_frozen.md"),
    "schema": Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json"),
    "primary_review_manifest": Path("outputs/amazon_v5_rebuild/features_semantic_b3/gate7_6_v3_primary_b3_annotation_review_manifest.csv"),
    "enrichment_review_manifest": Path("outputs/amazon_v5_rebuild/features_semantic_b3/gate7_6_v3_targeted_enrichment_review_manifest.csv"),
    "primary_batch_manifest": Path("outputs/amazon_v5_rebuild/features_semantic_b3/gate7_7_v3_primary_batch_manifest.csv"),
    "enrichment_batch_manifest": Path("outputs/amazon_v5_rebuild/features_semantic_b3/gate7_7_v3_enrichment_batch_manifest.csv"),
    "gate7_6_v3_report": Path("outputs/amazon_v5_rebuild/features_semantic_b3/gate7_6_annotation_coverage_design_report_v3.json"),
    "episode_membership_v3": Path("outputs/amazon_v5_rebuild/features_semantic_b3/episode_annotation_membership_v3.parquet"),
    "feature_dir": Path("outputs/amazon_v5_rebuild/features_semantic_b3"),
}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")


def json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_locked_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Locked hash mismatch for {label}: expected {expected}, actual {actual}")


def write_new_or_verify(path: Path, data: bytes) -> str:
    expected_hash = sha256_bytes(data)
    if path.exists():
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Immutable artifact differs: {path} expected {expected_hash}, existing {actual_hash}"
            )
        return actual_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(tmp_path), flags)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Immutable A001 artifacts require no-clobber finalization; never replace an existing final path.
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Immutable artifact differs after concurrent create: {path} "
                    f"expected {expected_hash}, existing {actual_hash}"
                )
            return actual_hash
        except OSError as e:
            raise RuntimeError(
                f"No-clobber hard-link finalization failed for {path}; "
                f"hard links may be unsupported on this filesystem. Refusing overwrite fallback. {e}"
            ) from e
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(f"Post-write hash mismatch for {path}: expected {expected_hash}, actual {actual_hash}")
    return actual_hash


def write_json_new_or_verify(path: Path, payload: Any) -> str:
    return write_new_or_verify(path, canonical_json_bytes(payload))


def build_run_schema(base_schema: dict[str, Any], batch_size: int) -> dict[str, Any]:
    run_schema = copy.deepcopy(base_schema)
    run_schema["properties"]["items"]["minItems"] = batch_size
    run_schema["properties"]["items"]["maxItems"] = batch_size
    return run_schema


def build_provider_response_json_schema(canonical_schema: dict[str, Any], batch_size: int) -> dict[str, Any]:
    provider_schema = copy.deepcopy(canonical_schema)

    def _clean(d: Any) -> None:
        if isinstance(d, dict):
            d.pop("$schema", None)
            d.pop("allOf", None)
            d.pop("if", None)
            d.pop("then", None)
            if "enum" in d and isinstance(d["enum"], list):
                d["enum"] = [x for x in d["enum"] if x is not None]
            for _, v in list(d.items()):
                _clean(v)
        elif isinstance(d, list):
            for i in d:
                _clean(i)

    _clean(provider_schema)
    items_def = provider_schema["properties"]["items"]["items"]

    for es_key in ["Evidence_Source1", "Evidence_Source2"]:
        if es_key in items_def["properties"]:
            items_def["properties"][es_key] = {
                "anyOf": [
                    {"type": "string", "enum": ["summary", "reviewText"]},
                    {"type": "null"},
                ]
            }

    for e_key in ["Evidence1", "Evidence2"]:
        if e_key in items_def["properties"]:
            items_def["properties"][e_key] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ]
            }

    if "Aspect2" in items_def["properties"]:
        items_def["properties"]["Aspect2"] = {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "Product_Condition_Quality",
                        "Product_Performance_Usability",
                        "Delivery_Fulfillment",
                        "Packaging_Presentation",
                        "Customer_Service_Returns",
                        "Price_Value",
                        "Listing_Expectation_Compatibility",
                        "Domain_Experience",
                        "Other_Specific",
                    ],
                },
                {"type": "null"},
            ]
        }

    if "Polarity2" in items_def["properties"]:
        items_def["properties"]["Polarity2"] = {
            "anyOf": [
                {"type": "string", "enum": ["Positive", "Negative", "Neutral", "NotApplicable"]},
                {"type": "null"},
            ]
        }

    provider_schema["properties"]["items"]["minItems"] = batch_size
    provider_schema["properties"]["items"]["maxItems"] = batch_size
    return provider_schema


def render_prompt(prompt_template: str, provider_input: list[dict[str, str]]) -> str:
    prompt = prompt_template.replace("{EXPECTED_BATCH_SIZE}", str(len(provider_input)))
    for item in provider_input:
        prompt += (
            f"\n--- ITEM ---\n"
            f"annotation_item_id: {item['annotation_item_id']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
        )
    return prompt


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def campaign_config(repo: Path, campaign_id: str) -> dict[str, Any]:
    feature_dir = repo / REL_PATHS["feature_dir"]
    if campaign_id == PRIMARY_CAMPAIGN_ID:
        return {
            "campaign_run_id": PRIMARY_CAMPAIGN_ID,
            "review_manifest_path": repo / REL_PATHS["primary_review_manifest"],
            "batch_manifest_path": repo / REL_PATHS["primary_batch_manifest"],
            "review_manifest_hash_key": "primary_review_manifest_sha256",
            "batch_manifest_hash_key": "primary_batch_manifest_sha256",
            "batch_prefix": "P",
            "expected_batch_count": 7745,
            "expected_review_count": 61960,
            "campaign_dir": feature_dir / PRIMARY_CAMPAIGN_ID,
        }
    if campaign_id == ENRICH_CAMPAIGN_ID:
        return {
            "campaign_run_id": ENRICH_CAMPAIGN_ID,
            "review_manifest_path": repo / REL_PATHS["enrichment_review_manifest"],
            "batch_manifest_path": repo / REL_PATHS["enrichment_batch_manifest"],
            "review_manifest_hash_key": "enrichment_review_manifest_sha256",
            "batch_manifest_hash_key": "enrichment_batch_manifest_sha256",
            "batch_prefix": "E",
            "expected_batch_count": 500,
            "expected_review_count": 4000,
            "campaign_dir": feature_dir / ENRICH_CAMPAIGN_ID,
        }
    raise ValueError(f"Unknown campaign {campaign_id}")


def load_and_assert_sources(repo: Path) -> tuple[str, dict[str, Any], dict[str, Any], str, str]:
    verify_locked_hash(repo / REL_PATHS["primary_review_manifest"], LOCKED_HASHES["primary_review_manifest_sha256"], "primary review manifest")
    verify_locked_hash(repo / REL_PATHS["enrichment_review_manifest"], LOCKED_HASHES["enrichment_review_manifest_sha256"], "enrichment review manifest")
    verify_locked_hash(repo / REL_PATHS["primary_batch_manifest"], LOCKED_HASHES["primary_batch_manifest_sha256"], "primary batch manifest")
    verify_locked_hash(repo / REL_PATHS["enrichment_batch_manifest"], LOCKED_HASHES["enrichment_batch_manifest_sha256"], "enrichment batch manifest")
    verify_locked_hash(repo / REL_PATHS["gate7_6_v3_report"], LOCKED_HASHES["gate7_6_v3_cohort_report_sha256"], "Gate 7.6 v3 cohort report")
    verify_locked_hash(repo / REL_PATHS["episode_membership_v3"], LOCKED_HASHES["episode_annotation_membership_v3_sha256"], "episode-review lineage manifest")
    verify_locked_hash(repo / REL_PATHS["prompt_template"], LOCKED_HASHES["prompt_template_v1_2_sha256"], "prompt v1.2")
    verify_locked_hash(repo / REL_PATHS["taxonomy"], LOCKED_HASHES["taxonomy_v1_2_sha256"], "taxonomy v1.2")
    verify_locked_hash(repo / REL_PATHS["schema"], LOCKED_HASHES["canonical_schema_file_v1_6_sha256"], "canonical schema file")

    prompt_template = (repo / REL_PATHS["prompt_template"]).read_text(encoding="utf-8")
    base_schema = json.loads((repo / REL_PATHS["schema"]).read_text(encoding="utf-8"))
    canonical_schema = build_run_schema(base_schema, ITEMS_PER_REQUEST)
    provider_schema = build_provider_response_json_schema(base_schema, ITEMS_PER_REQUEST)
    canonical_hash = sha256_bytes(json.dumps(canonical_schema, sort_keys=True).encode("utf-8"))
    provider_hash = sha256_bytes(json.dumps(provider_schema, sort_keys=True).encode("utf-8"))
    if canonical_hash != EXPECTED_CANONICAL_8_ITEM_SCHEMA_SHA256:
        raise RuntimeError(f"Canonical 8-item schema hash mismatch: {canonical_hash}")
    if provider_hash != EXPECTED_PROVIDER_PROJECTION_8_ITEM_SCHEMA_SHA256:
        raise RuntimeError(f"Provider projection schema hash mismatch: {provider_hash}")
    return prompt_template, canonical_schema, provider_schema, canonical_hash, provider_hash


def build_provider_input_and_provenance(
    group: pd.DataFrame,
    review_lookup: pd.DataFrame,
    campaign_id: str,
    batch_id: str,
    source_manifest_sha256: str,
    batch_input_sha256: str,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    provider_input: list[dict[str, str]] = []
    provenance: list[dict[str, Any]] = []
    for _, batch_row in group.sort_values("item_position").iterrows():
        item_id = str(batch_row["annotation_item_id"])
        review = review_lookup.loc[item_id]
        provider_item = {
            "annotation_item_id": item_id,
            "summary": normalize_text(review.get("summary", "")),
            "reviewText": normalize_text(review.get("reviewText", "")),
        }
        provider_input.append(provider_item)
        provenance.append(
            {
                "campaign_run_id": campaign_id,
                "batch_id": batch_id,
                "attempt_id": ATTEMPT_ID,
                "item_position": int(batch_row["item_position"]),
                "annotation_item_id": item_id,
                "review_timestamp": normalize_text(review.get("review_timestamp", "")),
                "source_manifest_sha256": source_manifest_sha256,
                "batch_input_sha256": batch_input_sha256,
            }
        )
    return provider_input, provenance


def materialize_or_verify_campaign(
    repo: Path,
    campaign_id: str,
    prompt_template: str,
    provider_schema: dict[str, Any],
    canonical_hash: str,
    provider_hash: str,
    verify_only: bool,
) -> dict[str, Any]:
    cfg = campaign_config(repo, campaign_id)
    verify_locked_hash(cfg["review_manifest_path"], LOCKED_HASHES[cfg["review_manifest_hash_key"]], f"{campaign_id} review manifest")
    verify_locked_hash(cfg["batch_manifest_path"], LOCKED_HASHES[cfg["batch_manifest_hash_key"]], f"{campaign_id} batch manifest")

    reviews = pd.read_csv(cfg["review_manifest_path"])
    batch_manifest = pd.read_csv(cfg["batch_manifest_path"])
    if len(reviews) != cfg["expected_review_count"]:
        raise RuntimeError(f"{campaign_id} review count mismatch: {len(reviews)}")
    if batch_manifest["annotation_item_id"].duplicated().any():
        raise RuntimeError(f"{campaign_id} duplicate annotation_item_id in batch manifest")
    if len(batch_manifest) != cfg["expected_review_count"]:
        raise RuntimeError(f"{campaign_id} batch row count mismatch: {len(batch_manifest)}")

    review_lookup = reviews.set_index("annotation_item_id", drop=False)
    campaign_dir = cfg["campaign_dir"]
    manifest_dir = campaign_dir / "manifests"
    batches_dir = campaign_dir / "batches"
    schema_path = manifest_dir / "provider_response_schema_frozen_v1_2.json"
    provider_schema_bytes = json.dumps(provider_schema, sort_keys=True).encode("utf-8")
    schema_hash = write_new_or_verify(schema_path, provider_schema_bytes) if not verify_only else sha256_file(schema_path)
    if schema_hash != provider_hash:
        raise RuntimeError(f"{campaign_id} stored provider schema hash mismatch: {schema_hash}")

    batch_summaries = []
    for batch_id, group in batch_manifest.groupby("batch_id", sort=True):
        group = group.sort_values("item_position")
        if len(group) != ITEMS_PER_REQUEST:
            raise RuntimeError(f"{campaign_id} {batch_id} has {len(group)} items")
        if group["item_position"].tolist() != list(range(1, ITEMS_PER_REQUEST + 1)):
            raise RuntimeError(f"{campaign_id} {batch_id} item positions are not 1..8")

        provisional_provider_input = []
        for _, batch_row in group.iterrows():
            review = review_lookup.loc[str(batch_row["annotation_item_id"])]
            provisional_provider_input.append(
                {
                    "annotation_item_id": str(batch_row["annotation_item_id"]),
                    "summary": normalize_text(review.get("summary", "")),
                    "reviewText": normalize_text(review.get("reviewText", "")),
                }
            )
        provider_input_bytes = canonical_json_bytes(provisional_provider_input)
        batch_input_sha256 = sha256_bytes(provider_input_bytes)
        provider_input, provenance = build_provider_input_and_provenance(
            group,
            review_lookup,
            campaign_id,
            str(batch_id),
            LOCKED_HASHES[cfg["review_manifest_hash_key"]],
            batch_input_sha256,
        )
        rendered_prompt_bytes = render_prompt(prompt_template, provider_input).encode("utf-8")

        attempt_dir = batches_dir / str(batch_id) / ATTEMPT_ID
        provider_input_path = attempt_dir / "provider_input.json"
        provenance_path = attempt_dir / "provenance_snapshot.json"
        prompt_path = attempt_dir / "rendered_prompt.md"
        contract_path = attempt_dir / "payload_contract_static_v3.json"
        legacy_checkpoint_path = attempt_dir / "checkpoint_state.json"
        checkpoint_path = attempt_dir / "checkpoint_state_static_v3.json"

        if verify_only:
            required = [provider_input_path, provenance_path, prompt_path, contract_path, checkpoint_path]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                raise RuntimeError(f"{campaign_id} {batch_id} missing artifacts: {missing[:3]}")
            provider_input_hash = sha256_file(provider_input_path)
            provenance_hash = sha256_file(provenance_path)
            rendered_prompt_hash = sha256_file(prompt_path)
        else:
            provider_input_hash = write_new_or_verify(provider_input_path, provider_input_bytes)
            provenance_hash = write_json_new_or_verify(provenance_path, provenance)
            rendered_prompt_hash = write_new_or_verify(prompt_path, rendered_prompt_bytes)

        if provider_input_hash != batch_input_sha256:
            raise RuntimeError(f"{campaign_id} {batch_id} provider_input hash mismatch")

        contract = {
            "campaign_run_id": campaign_id,
            "batch_id": str(batch_id),
            "attempt_id": ATTEMPT_ID,
            "items_per_request": ITEMS_PER_REQUEST,
            "batch_input_sha256": batch_input_sha256,
            "provider_input_path": str(provider_input_path),
            "provider_input_sha256": provider_input_hash,
            "provenance_snapshot_path": str(provenance_path),
            "provenance_snapshot_sha256": provenance_hash,
            "rendered_prompt_path": str(prompt_path),
            "rendered_prompt_sha256": rendered_prompt_hash,
            "provider_response_schema_path": str(schema_path),
            "provider_response_schema_sha256": schema_hash,
            "expected_hashes": {
                **LOCKED_HASHES,
                "canonical_8_item_schema_sha256": canonical_hash,
                "provider_projection_8_item_schema_sha256": provider_hash,
            },
            "raw_response_path": "raw_provider_response.json",
            "parsed_output_path": "parsed_output.json",
            "derived_validation_output_path": "derived_validation_output.json",
            "no_auto_retry": True,
            "no_overwrite_raw_artifact": True,
            "provider_request_sent": False,
            "failed_batch_retry_policy": "same batch_id with new attempt_id A002/A003; never overwrite A001",
        }
        checkpoint_expected = {
            "campaign_run_id": campaign_id,
            "batch_id": str(batch_id),
            "attempt_id": ATTEMPT_ID,
            "state": "PLANNED_NOT_EXECUTED",
            "provider_request_sent": False,
            "batch_input_sha256": batch_input_sha256,
            "rendered_prompt_sha256": rendered_prompt_hash,
            "terminal_reason": None,
        }
        if verify_only:
            stored_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if stored_contract != contract:
                raise RuntimeError(f"{campaign_id} {batch_id} contract mismatch")
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("state") != "PLANNED_NOT_EXECUTED" or checkpoint.get("provider_request_sent") is not False:
                raise RuntimeError(f"{campaign_id} {batch_id} checkpoint is not planned/not executed")
            if legacy_checkpoint_path.exists():
                legacy_checkpoint = json.loads(legacy_checkpoint_path.read_text(encoding="utf-8"))
                if legacy_checkpoint.get("state") != "PLANNED_NOT_EXECUTED" or legacy_checkpoint.get("provider_request_sent") is not False:
                    raise RuntimeError(f"{campaign_id} {batch_id} legacy checkpoint is not planned/not executed")
        else:
            write_json_new_or_verify(contract_path, contract)
            write_json_new_or_verify(checkpoint_path, checkpoint_expected)

        batch_summaries.append(
            {
                "campaign_run_id": campaign_id,
                "batch_id": str(batch_id),
                "attempt_id": ATTEMPT_ID,
                "item_count": ITEMS_PER_REQUEST,
                "provider_input_sha256": provider_input_hash,
                "batch_input_sha256": batch_input_sha256,
                "rendered_prompt_sha256": rendered_prompt_hash,
                "provider_response_schema_sha256": schema_hash,
                "provider_input_path": str(provider_input_path),
                "provenance_snapshot_path": str(provenance_path),
                "rendered_prompt_path": str(prompt_path),
                "payload_contract_path": str(contract_path),
                "checkpoint_path": str(checkpoint_path),
            }
        )

    if len(batch_summaries) != cfg["expected_batch_count"]:
        raise RuntimeError(f"{campaign_id} batch count mismatch: {len(batch_summaries)}")

    index_path = manifest_dir / "payload_contract_index_static_v3.json"
    index_hash = sha256_bytes(canonical_json_bytes(batch_summaries))
    if verify_only:
        if not index_path.exists():
            raise RuntimeError(f"{campaign_id} missing contract index")
        if sha256_file(index_path) != index_hash:
            raise RuntimeError(f"{campaign_id} contract index hash mismatch")
    else:
        write_json_new_or_verify(index_path, batch_summaries)

    return {
        "campaign_run_id": campaign_id,
        "review_count": int(len(reviews)),
        "batch_count": int(len(batch_summaries)),
        "all_batches_exactly_8": True,
        "provider_schema_path": str(schema_path),
        "provider_schema_sha256": schema_hash,
        "contract_index_path": str(index_path),
        "contract_index_sha256": index_hash,
    }


def run(repo: Path, verify_only: bool) -> dict[str, Any]:
    prompt_template, canonical_schema, provider_schema, canonical_hash, provider_hash = load_and_assert_sources(repo)
    primary = materialize_or_verify_campaign(
        repo,
        PRIMARY_CAMPAIGN_ID,
        prompt_template,
        provider_schema,
        canonical_hash,
        provider_hash,
        verify_only,
    )
    enrichment = materialize_or_verify_campaign(
        repo,
        ENRICH_CAMPAIGN_ID,
        prompt_template,
        provider_schema,
        canonical_hash,
        provider_hash,
        verify_only,
    )
    status = "VERIFY_ONLY_PASS" if verify_only else "PAYLOAD_CONTRACT_STATIC_ARTIFACTS_LOCKED"
    return {
        "gate7_7_static_payload_contract_status": status,
        "verify_only": verify_only,
        "provider_request_sent": False,
        "micro_pilot_live_executed": False,
        "schema_hash_assertions": {
            "canonical_8_item_schema_sha256": canonical_hash,
            "provider_projection_8_item_schema_sha256": provider_hash,
        },
        "locked_hashes": LOCKED_HASHES,
        "campaigns": {"primary": primary, "enrichment": enrichment},
        "verification_checks": {
            "counts_confirmed": True,
            "eight_item_batches_confirmed": True,
            "file_presence_confirmed": True,
            "hash_consistency_confirmed": True,
            "no_provider_request_sent_confirmed": True,
            "checkpoints_planned_not_executed_confirmed": True,
        },
        "still_pending_before_live_execution": {
            "quota_rate_limit": "PENDING",
            "cost_ceiling": "PENDING",
            "checkpoint_resume_implementation_review": "PENDING",
            "initial_pilot_at_scale_audit_rule": "PENDING",
            "periodic_sample_audit_rule": "PENDING",
            "operator_approval_for_live_provider_campaign": "PENDING",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    report = run(repo, verify_only=args.verify_only)
    out_dir = repo / REL_PATHS["feature_dir"]
    report_name = (
        "gate7_7_static_payload_contract_verify_only_report_v3.json"
        if args.verify_only
        else "gate7_7_static_payload_contract_materialization_report_v3.json"
    )
    report_path = out_dir / report_name
    write_json_new_or_verify(report_path, report)
    print(json.dumps({
        "status": report["gate7_7_static_payload_contract_status"],
        "verify_only": report["verify_only"],
        "primary_batches": report["campaigns"]["primary"]["batch_count"],
        "enrichment_batches": report["campaigns"]["enrichment"]["batch_count"],
        "provider_request_sent": False,
    }, indent=2))


if __name__ == "__main__":
    main()
