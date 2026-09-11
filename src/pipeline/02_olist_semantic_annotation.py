import pandas as pd
import json
import time
import os
import sys
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.framework.semantic_engine import SemanticEngine

# ─── Constants ────────────────────────────────────────────────────────────────
PROMPT_VERSION  = "v1"
SCHEMA_VERSION  = "v1"
TAXONOMY        = [
    "Customer_Service_Returns",
    "Delivery_Fulfillment",
    "Domain_Experience",
    "Price_Value",
    "Product_Condition_Quality",
    "Product_Performance_Usability",
    "Other_Specific",
]
BATCH_SIZE      = 25
BATCH_SLEEP_SEC = 4          
ROTATE_SLEEP_SEC = 10         
ERROR_SLEEP_SEC  = 60        
MAX_RETRIES     = 15          

def load_api_config():
    config_path = repo_root / "api_keys.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def build_prompt(reviews_chunk: pd.DataFrame) -> str:
    taxonomy_str = "\n".join(f"- {t}" for t in TAXONOMY)
    prompt = f"""You are an expert e-commerce data annotator.
Analyze the provided review texts (which may be in Portuguese or English) and extract up to TWO most prominent aspects for each.

Taxonomy (use ONLY these labels):
{taxonomy_str}

Rules:
1. Return EXACTLY ONE JSON object with a single key `results`.
2. The value of `results` must be a list, one object per review.
3. Each object MUST have:
   - `id`: the exact episode_id string provided.
   - `items`: a list of up to TWO aspects. Each has `aspect` (from Taxonomy) and `polarity` ("Positive" or "Negative").
   - If the review is generic or no aspect is identifiable, use `items: []`.
4. Return ONLY valid JSON — no markdown, no explanation.

Reviews:
"""
    for _, row in reviews_chunk.iterrows():
        text = str(row['origin_review_text']).strip().replace('"', "'").replace('\n', ' ')
        prompt += f'ID: {row["origin_order_id"]} | Text: "{text}"\n'
    return prompt


def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        return json.loads(checkpoint_path.read_text(encoding='utf-8'))
    return {}

def save_checkpoint(checkpoint_path: Path, checkpoint: dict):
    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding='utf-8')


def get_model_version(response, fallback_name):
    # Attempt to extract actual model version if the SDK returns it in model_version or something
    try:
        if hasattr(response, 'model_version') and response.model_version:
            return response.model_version
        if hasattr(response, 'usage_metadata') and hasattr(response.usage_metadata, 'model_version'):
            return response.usage_metadata.model_version
        if hasattr(response, 'candidates') and len(response.candidates) > 0 and hasattr(response.candidates[0], 'model_version'):
            return response.candidates[0].model_version
    except:
        pass
    return f"{fallback_name}_(unavailable)"

def hash_input(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def dry_run(df_all: pd.DataFrame, model_name: str):
    print("\n" + "="*50)
    print(" DRY RUN MODE")
    print("="*50)
    chunk = df_all.head(5)
    
    prompt = build_prompt(chunk)
    print(f"\n[Prompt preview length: {len(prompt)} chars]\n")
    
    model = genai.GenerativeModel(model_name)
    try:
        start_t = time.time()
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        duration = time.time() - start_t
        
        actual_model = get_model_version(response, model_name)
        
        print(f"MODEL: {actual_model}")
        print(f"TIMESTAMP: {datetime.now(timezone.utc).isoformat()}")
        print(f"INPUT_COUNT: 5")
        print(f"STATUS: REAL_API_SUCCESS (took {duration:.2f}s)")
        print("\n--- RAW TEXT RESPONSE ---")
        print(response.text)
        print("-------------------------\n")
        
        # Verify JSON
        raw = response.text.strip()
        if raw.startswith("```json"): raw = raw[7:]
        if raw.endswith("```"): raw = raw[:-3]
        data = json.loads(raw.strip())
        results = data.get("results", [])
        print(f"OUTPUT_COUNT: {len(results)}")
        
        print("\n[Dry Run successful. Run without --dry-run to process full cohort]")
    except Exception as e:
        print(f"\n[DRY RUN FAILED]: {e}")
        sys.exit(1)

def annotate_all(df_all: pd.DataFrame, checkpoint: dict, errors_path: Path, model_name: str, keys: list) -> dict:
    already_done = set(
        eid for eid, rec in checkpoint.items()
        if rec.get('status') == 'annotated_real'
    )
    to_annotate = df_all[~df_all['origin_order_id'].isin(already_done)].reset_index(drop=True)
    total_remaining = len(to_annotate)

    print(f"[Annotation] {len(already_done)} already annotated. {total_remaining} remaining.")

    if total_remaining == 0:
        print("[Annotation] Nothing to do — all episodes already annotated.")
        return checkpoint

    total_batches = (total_remaining + BATCH_SIZE - 1) // BATCH_SIZE
    key_idx = 0
    genai.configure(api_key=keys[key_idx])
    model = genai.GenerativeModel(model_name)

    for b in range(total_batches):
        chunk = to_annotate.iloc[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        episode_ids_in_chunk = chunk['origin_order_id'].tolist()
        review_ids_in_chunk = chunk['review_id'].tolist()
        texts_in_chunk = chunk['origin_review_text'].tolist()
        
        prompt = build_prompt(chunk)
        success = False
        retries_left = MAX_RETRIES

        while retries_left > 0 and not success:
            try:
                time.sleep(4)
                response = model.generate_content(
                    prompt,
                    generation_config=GenerationConfig(response_mime_type="application/json"),
                    request_options={"timeout": 900}
                )
                
                actual_model = get_model_version(response, model_name)
                
                raw = response.text.strip()
                if raw.startswith("```json"): raw = raw[7:]
                if raw.endswith("```"): raw = raw[:-3]

                data = json.loads(raw.strip())
                results = data.get("results", [])
                id_to_targets = {str(r["id"]): r.get("items", []) for r in results}

                now = datetime.now(timezone.utc).isoformat()
                for eid, rid, txt in zip(episode_ids_in_chunk, review_ids_in_chunk, texts_in_chunk):
                    targets = id_to_targets.get(eid, None)
                    if targets is None:
                        checkpoint[eid] = {
                            "episode_id": eid,
                            "review_id": rid,
                            "input_hash": hash_input(str(txt)),
                            "status": "parse_failed",
                            "model_used": actual_model,
                            "prompt_version": PROMPT_VERSION,
                            "timestamp": now,
                            "targets": None,
                        }
                    else:
                        valid = [
                            item for item in targets
                            if item.get("aspect") in TAXONOMY and item.get("polarity") in ("Positive", "Negative")
                        ]
                        checkpoint[eid] = {
                            "episode_id": eid,
                            "review_id": rid,
                            "input_hash": hash_input(str(txt)),
                            "status": "annotated_real",
                            "model_used": actual_model,
                            "prompt_version": PROMPT_VERSION,
                            "timestamp": now,
                            "targets": valid,
                        }

                save_checkpoint(errors_path.parent / "annotations_checkpoint.json", checkpoint)
                success = True
                print(f"  Batch {b+1}/{total_batches} OK [{actual_model}] | "
                      f"annotated: {sum(1 for v in checkpoint.values() if v.get('status')=='annotated_real')}/{len(df_all)}")

            except Exception as e:
                retries_left -= 1
                err = str(e)
                is_quota = "429" in err or "403" in err or "Quota" in err or "quota" in err
                print(f"  Batch {b+1} failed. Retries left: {retries_left}. "
                      f"{'[QUOTA]' if is_quota else '[ERROR]'} {err[:100]}")

                if is_quota and len(keys) > 1:
                    key_idx = (key_idx + 1) % len(keys)
                    genai.configure(api_key=keys[key_idx])
                    model = genai.GenerativeModel(model_name)
                    print(f"  -> Rotated API key. Sleeping {ROTATE_SLEEP_SEC}s")
                    time.sleep(ROTATE_SLEEP_SEC)
                else:
                    time.sleep(ERROR_SLEEP_SEC)

        if not success:
            print(f"\n[CRITICAL] Batch {b+1} COMPLETELY FAILED after retries.")
            print("Stopping the pipeline to preserve checkpoint and avoid silent failures.")
            sys.exit(1)

        time.sleep(BATCH_SLEEP_SEC)

    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help="Run 5 samples and print full API response")
    args = parser.parse_args()

    config = load_api_config()
    keys = config.get("keys", []) if config else []
    
    # Try reading from ENV if config not found
    if not keys and "GEMINI_API_KEY" in os.environ:
        keys = [os.environ["GEMINI_API_KEY"]]
        
    if not keys:
        print("Error: No GEMINI_API_KEY found in api_keys.json or environment.")
        sys.exit(1)
        
    model_name = config.get("model", "gemini-1.5-flash-latest") if config else "gemini-1.5-flash-latest"

    genai.configure(api_key=keys[0])

    olist_dir  = repo_root / "data/processed/olist"
    cp_path    = olist_dir / "annotations_checkpoint.json"
    err_path   = olist_dir / "annotation_errors.jsonl"

    if not (olist_dir / "olist_train_analytical.csv").exists():
        print("Error: Dataset not found. Please run 01_olist_dataset_builder.py first.")
        sys.exit(1)

    df_train = pd.read_csv(olist_dir / "olist_train_analytical.csv")
    df_test  = pd.read_csv(olist_dir / "olist_test_analytical.csv")
    df_train['_split'] = 'train'
    df_test['_split']  = 'test'
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    print(f"Dataset: train={len(df_train)}, test={len(df_test)}, total={len(df_all)}")
    print(f"Configured Model: {model_name}")

    if args.dry_run:
        dry_run(df_all, model_name)
        sys.exit(0)

    checkpoint = load_checkpoint(cp_path)
    if cp_path.exists():
        print(f"Loaded checkpoint: {len(checkpoint)} records (episode-id keyed)")

    checkpoint = annotate_all(df_all, checkpoint, err_path, model_name, keys)
    
    print("\nAnnotation phase completed successfully.")
    
    # Only proceed to build the semantic csv if 100% covered
    already_done = sum(1 for eid, rec in checkpoint.items() if rec.get('status') == 'annotated_real')
    if already_done < len(df_all):
        print(f"WARNING: Coverage is {already_done}/{len(df_all)}. Run script again to finish.")
        sys.exit(1)
        
    # Build semantic df
    def build_semantic_df(df, checkpoint):
        rows = []
        for _, row in df.iterrows():
            eid = row['origin_order_id']
            rec = checkpoint.get(eid, {})
            targets_str = json.dumps(rec['targets'], ensure_ascii=False) if rec.get('status') == 'annotated_real' else None
            rows.append({**row.to_dict(), 'final_targets': targets_str, 'annotation_status': rec.get('status', 'not_annotated')})
        return pd.DataFrame(rows)

    df_train_sem = build_semantic_df(df_train, checkpoint)
    df_test_sem  = build_semantic_df(df_test,  checkpoint)

    # Run Universal Semantic Engine
    print("Running Universal Semantic Engine on annotated rows...")
    df_train_real = df_train_sem[df_train_sem['annotation_status'] == 'annotated_real'].copy()
    df_test_real  = df_test_sem[df_test_sem['annotation_status']  == 'annotated_real'].copy()

    for df in [df_train_real, df_test_real]:
        df['CustomerID']   = df['customer_unique_id']
        df['episode_id']   = df['origin_order_id']
        df['episode_start'] = pd.to_datetime(df['origin_purchase_time'])
        df['Review_Mixed_Flag'] = False

    se = SemanticEngine()
    df_train_out = se.process(df_train_real)
    df_test_out  = se.process(df_test_real)

    df_train_out['annotation_status'] = 'annotated_real'
    df_test_out['annotation_status']  = 'annotated_real'

    df_train_out.to_csv(olist_dir / "olist_train_semantic.csv", index=False)
    df_test_out.to_csv(olist_dir / "olist_test_semantic.csv",   index=False)

    print(f"Saved olist_train_semantic.csv: {len(df_train_out)} rows")
    print(f"Saved olist_test_semantic.csv:  {len(df_test_out)} rows")
    print("\nPhase 3 Semantic Annotation complete. Run 02b_olist_adapter.py next.")

if __name__ == "__main__":
    main()
