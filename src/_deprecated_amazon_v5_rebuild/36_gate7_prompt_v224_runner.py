#!/usr/bin/env python
"""Gate 7.6: Mass Annotation Runner (Prompt v2.2.4)

Implements Quota-Aware API Key Rotation (Single Model: gemini-2.5-flash).
Modes:
  --mode qa   : Stratified QA sample (e.g., 5,000 items stratified by rating).
  --mode full : Full Amazon Corpus.
"""

import argparse
import importlib.util
import json
import time
import hashlib
import builtins
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
import os
import sys

_print = builtins.print

def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _print(*args, **kwargs)

# Load v224 functions
def load_v224_runner():
    v224_path = Path(__file__).with_name("35_gate7_30_v224_aspect_only_calibration_runner.py")
    spec = importlib.util.spec_from_file_location("v224", v224_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

v224 = load_v224_runner()
v222 = v224.v222  # The patched v222 base

class RotationManager:
    def __init__(self, config_path: Path):
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            print('Please create it. Example: {"keys": ["KEY1"], "model": "gemini-2.5-flash", "warning_threshold": {"unknown_rate": 5, "polarity_shift": 20}}')
            sys.exit(1)
            
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.keys = self.config.get("keys", [])
        if not self.keys:
            print("No keys found in config.")
            sys.exit(1)
            
        self.model_name = self.config.get("model", "gemini-2.5-flash")
        self.warning_threshold = self.config.get("warning_threshold", {"unknown_rate": 5, "polarity_shift": 20})
        self.active_key_idx = 0
        self.dead_keys = set()
        
        # Configure initial key
        self.switch_key()

    def get_active_key_hash(self):
        key = self.keys[self.active_key_idx]
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def switch_key(self):
        if len(self.dead_keys) >= len(self.keys):
            print("CRITICAL: All API keys have exhausted their daily quota. Halting execution.")
            return False
            
        while self.active_key_idx in self.dead_keys:
            self.active_key_idx = (self.active_key_idx + 1) % len(self.keys)
            
        api_key = self.keys[self.active_key_idx]
        self.client = genai.Client(api_key=api_key)
        print(f"Switched to API Key [Hash: {self.get_active_key_hash()}]")
        return True

    def mark_current_dead(self):
        print(f"Marking API Key [Hash: {self.get_active_key_hash()}] as EXHAUSTED.")
        self.dead_keys.add(self.active_key_idx)
        return self.switch_key()

    def generate(self, prompt: str) -> dict:
        start_time = time.time()
        result = {
            "text": "",
            "latency": 0.0,
            "status": "ERROR",
            "finish_reason": None,
            "usage_metadata": None,
            "error": None
        }
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                    max_output_tokens=8192,
                    http_options=types.HttpOptions(timeout=120000),
                )
            )
            result["latency"] = time.time() - start_time
            result["text"] = response.text
            result["status"] = "SUCCESS"
            
            if hasattr(response, 'candidates') and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, 'finish_reason') and cand.finish_reason is not None:
                    result["finish_reason"] = getattr(cand.finish_reason, 'name', str(cand.finish_reason))
                    
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                um = response.usage_metadata
                result["usage_metadata"] = {
                    "prompt_token_count": getattr(um, 'prompt_token_count', 0),
                    "candidates_token_count": getattr(um, 'candidates_token_count', 0),
                    "total_token_count": getattr(um, 'total_token_count', 0)
                }
            return result
        except Exception as e:
            result["latency"] = time.time() - start_time
            result["error"] = str(e)
            result["text"] = str(e)
            return result


class MassAnnotationRunner:
    def __init__(self, repo_root: Path, mode: str, config_path: Path, max_batches: int | None = None):
        self.repo_root = repo_root
        self.mode = mode
        self.max_batches = max_batches
        self.rm = RotationManager(config_path)
        
        self.workspace = repo_root / "outputs" / "amazon_v5_rebuild" / "annotation" / "mass_workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.ckpt_path = self.workspace / f"mass_checkpoint_v224_{mode}.json"
        self.results_path = self.workspace / f"mass_results_v224_{mode}.jsonl"
        self.manifest_path = self.workspace / f"annotation_manifest_v224_{mode}.json"
        
        self.taxonomy_aspects = set(v222.ASPECTS)
        
        self.load_checkpoint()
        
    def load_checkpoint(self):
        if self.ckpt_path.exists():
            with open(self.ckpt_path, "r") as f:
                self.ckpt = json.load(f)
            self.ckpt.setdefault("telemetry", {
                "batch_count": len(self.ckpt.get("completed_batch_ids", [])),
                "batch_success_count": len(self.ckpt.get("completed_batch_ids", [])),
                "json_parse_failures": 0,
                "completeness_failures": 0,
                "retry_count": self.ckpt.get("stats", {}).get("429_count", 0),
                "total_latency": 0.0,
                "total_output_items": self.ckpt.get("stats", {}).get("processed_count", 0)
            })
            print(f"Resumed {self.mode} checkpoint. Completed items: {len(self.ckpt['completed_item_ids'])}")
        else:
            self.ckpt = {
                "completed_item_ids": [],
                "completed_batch_ids": [],
                "failed_item_ids": [],
                "stats": {
                    "processed_count": 0,
                    "failed_count": 0,
                    "429_count": 0,
                    "unknown_aspect_count": 0,
                    "total_aspect_count": 0,
                    "empty_extraction_count": 0,
                    "mixed_review_count": 0,
                    "aspect_distribution": {},
                    "polarity_distribution": {}
                },
                "telemetry": {
                    "batch_count": 0,
                    "batch_success_count": 0,
                    "json_parse_failures": 0,
                    "completeness_failures": 0,
                    "retry_count": 0,
                    "total_latency": 0.0,
                    "total_output_items": 0
                }
            }

    def save_checkpoint(self):
        with open(self.ckpt_path, "w") as f:
            json.dump(self.ckpt, f, indent=2)

    def update_stats(self, results):
        stats = self.ckpt["stats"]
        for r in results:
            if not r['is_valid']:
                stats["failed_count"] += 1
                continue
                
            stats["processed_count"] += 1
            targets = r.get('final_targets', [])
            
            if not targets:
                stats["empty_extraction_count"] += 1
                
            if r.get('Review_Mixed_Flag'):
                stats["mixed_review_count"] += 1
                
            for target in targets:
                asp = target.get('aspect')
                pol = target.get('polarity')
                
                stats["total_aspect_count"] += 1
                if asp not in self.taxonomy_aspects:
                    stats["unknown_aspect_count"] += 1
                    
                stats["aspect_distribution"][asp] = stats["aspect_distribution"].get(asp, 0) + 1
                stats["polarity_distribution"][pol] = stats["polarity_distribution"].get(pol, 0) + 1

    def print_drift_warning(self):
        stats = self.ckpt["stats"]
        total_aspects = stats["total_aspect_count"]
        processed = stats["processed_count"]
        
        if processed == 0: return
        
        unknown_rate = stats["unknown_aspect_count"] / max(total_aspects, 1)
        avg_aspects = total_aspects / processed
        empty_rate = stats["empty_extraction_count"] / processed
        mixed_rate = stats["mixed_review_count"] / processed
        neg_rate = stats["polarity_distribution"].get("Negative", 0) / max(total_aspects, 1)
        
        thresholds = self.rm.warning_threshold
        
        print(f"--- DRIFT MONITORING ---")
        print(f"Processed: {processed} | Avg Aspects/Review: {avg_aspects:.2f}")
        print(f"Unknown Rate: {unknown_rate:.2%} | Empty Extr: {empty_rate:.2%} | Mixed: {mixed_rate:.2%} | Neg Polarity: {neg_rate:.2%}")
        
        if unknown_rate * 100 > thresholds.get("unknown_rate", 5):
            print(f"CRITICAL WARNING: UNKNOWN aspect rate ({unknown_rate:.2%}) exceeds threshold ({thresholds.get('unknown_rate', 5)}%).")
            
        print("------------------------")

    def get_dataset(self) -> pd.DataFrame:
        # verified_episode_review_membership_v1 is the review-level table with
        # reviewerID, episode_id, reviewText, summary, overall, asin columns.
        parquet_path = self.repo_root / "outputs" / "amazon_v5_rebuild" / "data" / "verified_episode_review_membership_v1.parquet"
        if not parquet_path.exists():
            print(f"Source parquet not found: {parquet_path}")
            sys.exit(1)

        df = pd.read_parquet(parquet_path)
        # Build annotation_item_id consistently with calibration manifest convention
        df['annotation_item_id'] = df['reviewerID'].astype(str) + '_' + df['episode_id'].astype(str)

        # Ensure text columns exist
        for col in ('reviewText', 'summary'):
            if col not in df.columns:
                df[col] = ''

        if self.mode == "qa":
            sample_size = min(5000, len(df))
            stratify_cols = ['overall']
            # Detect category column if present
            for cat_col in ('category', 'main_cat', 'asin_category'):
                if cat_col in df.columns:
                    stratify_cols.append(cat_col)
                    break

            print(f"Stratifying QA Mode by: {stratify_cols} | target={sample_size}")
            df = df.groupby(stratify_cols, group_keys=False).apply(
                lambda x: x.sample(max(1, int(np.ceil(len(x) / len(df) * sample_size))), random_state=42)
            ).reset_index(drop=True)
            df = df.head(sample_size)

        print(f"Dataset loaded: {len(df):,} items (mode={self.mode})")
        return df

    def save_manifest(self, total_expected):
        manifest = {
            "prompt_version": "v2.2.4",
            "model": self.rm.model_name,
            "taxonomy_version": "v2.2.4",
            "total_reviews": total_expected,
            "annotated_reviews": self.ckpt["stats"]["processed_count"],
            "annotation_date": datetime.now(timezone.utc).isoformat(),
            "keys_used": len(self.rm.dead_keys) + 1,
            "final_stats": self.ckpt["stats"]
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to {self.manifest_path}")

    def run(self):
        df = self.get_dataset()
        all_indices = df.index.tolist()
        batch_size = 10
        batches_started = 0
        
        for i in range(0, len(all_indices), batch_size):
            if self.max_batches is not None and batches_started >= self.max_batches:
                print(f"\nStopped after --max-batches={self.max_batches}.")
                break

            batch_id = f"batch_{i//batch_size}"
            if batch_id in self.ckpt['completed_batch_ids']:
                continue
                
            b_df = df.iloc[i:i+batch_size]
            
            # Filter out already completed items
            pending_items = [row for _, row in b_df.iterrows() if row['annotation_item_id'] not in self.ckpt['completed_item_ids']]
            if not pending_items:
                self.ckpt['completed_batch_ids'].append(batch_id)
                self.save_checkpoint()
                continue
                
            batches_started += 1
            print(f"\nProcessing {batch_id} ({len(pending_items)} items) with Key [{self.rm.get_active_key_hash()}]...")
            start_time_str = datetime.now(timezone.utc).isoformat()
            prompt = v224.render_prompt([
                {
                    "item_position": str(idx),
                    "summary": str(row.get('summary', '')),
                    "reviewText": str(row.get('reviewText', ''))
                } for idx, row in enumerate(pending_items)
            ])
            
            # Create directories and paths for audit
            raw_logs_dir = self.workspace / "raw_logs"
            raw_logs_dir.mkdir(parents=True, exist_ok=True)
            ledger_path = self.workspace / "retry_ledger.jsonl"
            
            t = self.ckpt["telemetry"]
            t["batch_count"] += 1
            
            max_retry = 3
            attempt = 0
            parse_ok = False
            llm_items = []
            final_latency = 0.0
            consecutive_429s = 0
            
            while attempt < max_retry and not parse_ok:
                attempt += 1
                if attempt > 1:
                    print(f"  Retry attempt {attempt}/{max_retry}...")
                    t["retry_count"] += 1
                
                # Provider call loop (only retries on 429/503)
                provider_success = False
                gen_result = {}
                while not provider_success:
                    gen_result = self.rm.generate(prompt)
                    final_latency += gen_result.get("latency", 0.0)
                    
                    if gen_result["status"] == "ERROR":
                        raw_text = gen_result["text"]
                        if "429" in raw_text or "Quota" in raw_text or "RESOURCE_EXHAUSTED" in raw_text:
                            consecutive_429s += 1
                            self.ckpt["stats"]["429_count"] += 1
                            if consecutive_429s >= 2:
                                print("Persistent 429. Rotating Key.")
                                if not self.rm.mark_current_dead():
                                    self.save_checkpoint()
                                    sys.exit(0)
                                consecutive_429s = 0
                            else:
                                print(f"429/Quota Encountered. Sleeping 65s... (Error: {raw_text[:80]}...)")
                                time.sleep(65)
                        elif "503" in raw_text:
                            print("503 Server Unavailable. Sleeping 60s...")
                            time.sleep(60)
                        else:
                            print(f"CRITICAL API ERROR (Not 429): {raw_text}")
                            print("Likely causes: invalid API key, provider access disabled, billing issue, or a non-quota provider error.")
                            self.save_checkpoint()
                            sys.exit(1)
                    else:
                        provider_success = True
                
                # We have a successful provider response
                raw_text = gen_result["text"]
                raw_text_clean = raw_text.strip()
                if raw_text_clean.startswith("```json"): raw_text_clean = raw_text_clean[7:-3].strip()
                elif raw_text_clean.startswith("```"): raw_text_clean = raw_text_clean[3:-3].strip()
                
                # Save raw response
                raw_log_file = raw_logs_dir / f"{batch_id}_attempt{attempt:02d}.txt"
                with open(raw_log_file, "w", encoding="utf-8") as rf:
                    rf.write(f"Finish Reason: {gen_result.get('finish_reason')}\n")
                    rf.write(f"Usage: {json.dumps(gen_result.get('usage_metadata'))}\n")
                    rf.write(f"Attempt: {attempt}\n")
                    rf.write("--- RAW TEXT ---\n")
                    rf.write(raw_text)
                    
                # Parsing
                try:
                    data = json.loads(raw_text_clean)
                    current_llm_items = data.get("items", [])
                    
                    if len(current_llm_items) >= len(pending_items):
                        llm_items = current_llm_items
                        parse_ok = True
                    else:
                        if len(current_llm_items) > len(llm_items):
                            llm_items = current_llm_items
                        raise ValueError(f"Output completeness FAIL: expected {len(pending_items)}, got {len(current_llm_items)}")
                except Exception as e:
                    parse_ok = False
                    is_parse_error = "Expecting" in str(e) or "JSON" in str(e) or "decode" in str(e)
                    if is_parse_error:
                        t["json_parse_failures"] += 1
                    else:
                        t["completeness_failures"] += 1
                        
                    print(f"    -> Validation failed: {str(e)}")
                    print(f"       Finish Reason: {gen_result.get('finish_reason')} | Tokens: {gen_result.get('usage_metadata')}")
                    
                    # Ledger entry
                    ledger_entry = {
                        "batch_id": batch_id,
                        "attempt": attempt,
                        "error_type": "parse_failed" if is_parse_error else "completeness_failed",
                        "reason": str(e),
                        "response_chars": len(raw_text),
                        "finish_reason": gen_result.get("finish_reason"),
                        "usage_metadata": gen_result.get("usage_metadata")
                    }
                    with open(ledger_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(ledger_entry) + "\n")

            t["total_latency"] += final_latency

            if not parse_ok:
                print(f"Batch {batch_id} failed to reach 100% completeness after {max_retry} attempts. Discarding partial results and skipping batch to maintain reproducibility.")
                self.save_checkpoint()
                continue

            # ── Build results ─────────────────────────────────────────────
            results = []
            for idx, item in enumerate(llm_items):
                if idx >= len(pending_items): break
                row = pending_items[idx]
                iid = row['annotation_item_id']

                targets = item.get("targets", [])
                normalized = v224.normalize_targets(row.to_dict(), targets)

                has_pos = any(t2['polarity'] == 'Positive' for t2 in normalized)
                has_neg = any(t2['polarity'] == 'Negative' for t2 in normalized)

                results.append({
                    'annotation_item_id': iid,
                    'provider_targets': targets,
                    'final_targets': normalized,
                    'Review_Mixed_Flag': has_pos and has_neg,
                    'is_valid': True
                })

            t["total_output_items"] += len(results)
            if len(results) == len(pending_items):
                t["batch_success_count"] += 1

            # ── Log results ───────────────────────────────────────────────
            end_time_str = datetime.now(timezone.utc).isoformat()
            with open(self.results_path, "a") as f:
                per_item_latency = final_latency / max(len(results), 1)
                for r in results:
                    r['batch_id'] = batch_id
                    r['start_time'] = start_time_str
                    r['end_time'] = end_time_str
                    r['latency'] = per_item_latency
                    r['api_key_hash'] = self.rm.get_active_key_hash()
                    r['model_version'] = self.rm.model_name
                    f.write(json.dumps(r) + "\n")
                    self.ckpt['completed_item_ids'].append(r['annotation_item_id'])

            self.ckpt['completed_batch_ids'].append(batch_id)
            self.update_stats(results)
            self.print_drift_warning()
            self.save_checkpoint()

        self.save_manifest(len(df))
        self.print_batch_summary()
        print(f"\n{self.mode.upper()} Annotation Complete.")

    def print_batch_summary(self):
        t = self.ckpt["telemetry"]
        bc = t["batch_count"]
        if bc == 0:
            return
        avg_latency = t["total_latency"] / bc
        success_rate = t["batch_success_count"] / bc * 100
        avg_output = t["total_output_items"] / bc
        print("\n=== BATCH QA SUMMARY ===")
        print(f"  Batch Success Rate    : {success_rate:.1f}%  ({t['batch_success_count']}/{bc})")
        print(f"  Avg Latency / batch   : {avg_latency:.1f}s")
        print(f"  Avg Output Items      : {avg_output:.1f}")
        print(f"  Retry Count           : {t['retry_count']}")
        print(f"  JSON Parse Failures   : {t['json_parse_failures']}")
        print(f"  Completeness Failures : {t['completeness_failures']}")
        print(f"  Output Completeness   : {t['total_output_items']}/{self.ckpt['stats']['processed_count']} reviews")
        print("========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass Annotation Runner (Gate 7)")
    parser.add_argument("--mode", choices=["qa", "full"], required=True, help="Execution mode")
    parser.add_argument("--config", type=str, default="api_keys.json", help="Path to JSON config with keys and model")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional safety limit for provider batches")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    runner = MassAnnotationRunner(repo_root, args.mode, Path(args.config), args.max_batches)
    runner.run()
