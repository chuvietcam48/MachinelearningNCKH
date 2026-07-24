import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY environment variable not set.")
        sys.exit(1)
    genai.configure(api_key=api_key)

class OutputValidator:
    def __init__(self, taxonomy_path):
        with open(taxonomy_path, 'r') as f:
            self.taxonomy_text = f.read()
            
        # Parse taxonomy aspects from markdown
        self.allowed_aspects = []
        for line in self.taxonomy_text.split('\n'):
            if line.startswith('- `'):
                aspect = line.split('`')[1]
                self.allowed_aspects.append(aspect)
                
        self.allowed_polarities = ["Positive", "Negative", "Neutral", "NotApplicable"]

    def validate_item(self, item, summary, review):
        errors = []
        
        # Check Aspect1
        if item.get('Aspect1') not in self.allowed_aspects:
            errors.append(f"Invalid Aspect1: {item.get('Aspect1')}")
            
        if item.get('Polarity1') not in self.allowed_polarities:
            errors.append(f"Invalid Polarity1: {item.get('Polarity1')}")
            
        # Check Aspect2 (can be null)
        if item.get('Aspect2') is not None and item.get('Aspect2') not in self.allowed_aspects:
            errors.append(f"Invalid Aspect2: {item.get('Aspect2')}")
            
        if item.get('Polarity2') is not None and item.get('Polarity2') not in self.allowed_polarities:
            errors.append(f"Invalid Polarity2: {item.get('Polarity2')}")
            
        # Evidence checks
        for i in [1, 2]:
            evidence = item.get(f'Evidence{i}')
            source = item.get(f'Evidence_Source{i}')
            start = item.get(f'Evidence_Start{i}')
            end = item.get(f'Evidence_End{i}')
            
            if evidence:
                if source not in ['summary', 'reviewText']:
                    errors.append(f"Invalid Evidence_Source{i}: {source}")
                    continue
                    
                target_text = summary if source == 'summary' else review
                if target_text is None: target_text = ""
                
                if evidence not in target_text:
                    errors.append(f"Evidence{i} is not a substring of {source}")
                else:
                    # Auto-correct the LLM's hallucinated indices
                    real_start = target_text.find(evidence)
                    real_end = real_start + len(evidence)
                    item[f'Evidence_Start{i}'] = real_start
                    item[f'Evidence_End{i}'] = real_end

        # Mixed flag check
        mixed = item.get('Review_Mixed_Flag')
        if not isinstance(mixed, bool):
            errors.append("Review_Mixed_Flag must be boolean")
        else:
            has_pos = False
            has_neg = False
            for i in [1, 2]:
                pol = item.get(f'Polarity{i}')
                asp = item.get(f'Aspect{i}')
                if asp and asp != 'None':
                    if pol == 'Positive': has_pos = True
                    if pol == 'Negative': has_neg = True
            if mixed and not (has_pos and has_neg):
                # Auto-correct instead of failing
                item['Review_Mixed_Flag'] = False
            elif not mixed and (has_pos and has_neg):
                item['Review_Mixed_Flag'] = True
                
        # Heuristic check for ratings / forbidden info in aspect
        for k, v in item.items():
            if isinstance(v, str) and k.startswith('Aspect'):
                if 'star' in v.lower() or 'rating' in v.lower():
                    errors.append(f"Forbidden text in {k}: {v}")
                    
        return errors

def call_gemini(model, prompt, retry_count=0):
    start_time = time.time()
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=8192,
            ),
            request_options={"timeout": 600},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        latency = time.time() - start_time
        return response.text, latency, "SUCCESS"
    except Exception as e:
        latency = time.time() - start_time
        return str(e), latency, "ERROR"

def run_phase2b1():
    print("Starting Phase 2B-1: Gemini API Quota-Aware Batching")
    setup_gemini()
    
    base_dir = Path("outputs/amazon_v5_rebuild/annotation")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = base_dir / "calibration_sample_manifest_v1.csv"
    df = pd.read_csv(manifest_path)
    # create annotation_item_id
    df['annotation_item_id'] = df['reviewerID'] + '_' + df['episode_id'].astype(str)
    
    with open(base_dir / "gemini_semantic_prompt_template_v1_1.md", "r") as f:
        prompt_template = f.read()
    with open(base_dir / "taxonomy_v1_1.md", "r") as f:
        taxonomy_text = f.read()
    
    schema_text = '''{
  "items": [
    {
      "annotation_item_id": "...",
      "Aspect1": "...",
      "Polarity1": "...",
      "Evidence_Source1": "...",
      "Evidence_Start1": 0,
      "Evidence_End1": 0,
      "Evidence1": "...",
      "Aspect2": null,
      "Polarity2": null,
      "Evidence_Source2": null,
      "Evidence_Start2": null,
      "Evidence_End2": null,
      "Evidence2": null,
      "Review_Mixed_Flag": false
    }
  ]
}'''
        
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()
    taxonomy_hash = hashlib.sha256(taxonomy_text.encode()).hexdigest()
    manifest_hash = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    
    validator = OutputValidator(base_dir / "taxonomy_v1_1.md")
    
    model_id = "gemini-flash-latest"
    model = genai.GenerativeModel(model_name=model_id)
    
    # Checkpoint logic
    ckpt_path = base_dir / "gemini_batch_run_checkpoint_v1.json"
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            checkpoint = json.load(f)
        print(f"Resuming from checkpoint. Completed items: {len(checkpoint['completed_annotation_item_ids'])}")
    else:
        checkpoint = {
            "completed_batch_ids": [],
            "completed_annotation_item_ids": [],
            "failed_item_ids": [],
            "request_count": 0,
            "repair_request_count": 0,
            "provider": "google",
            "model": model_id,
            "prompt_hash": prompt_hash,
            "taxonomy_hash": taxonomy_hash,
            "latest_error_type": None,
            "resume_safe_timestamp": None,
            "all_results": []
        }
    
    # Select deterministic 10 reviews
    df['text_len'] = df['reviewText'].fillna('').str.len()
    idx1 = df['text_len'].idxmin()
    idx2 = df['text_len'].idxmax()
    idx3 = df[(df['overall'] == 5) & (df['text_len'] < 100)].index[0]
    idx4 = df[(df['overall'] == 1) & (df['text_len'] < 100)].index[0]
    other_indices = df[~df.index.isin([idx1, idx2, idx3, idx4])].index[:6]
    preflight_indices = [idx1, idx2, idx3, idx4] + list(other_indices)
    
    # Order: preflight first, then rest
    rest_indices = df[~df.index.isin(preflight_indices)].index
    
    batches = []
    # Batch 0 is preflight
    batches.append({
        "batch_id": "batch_0_preflight",
        "indices": preflight_indices,
        "is_preflight": True
    })
    
    # Rest are calibration batches of 10
    rest_list = list(rest_indices)
    for i in range(0, len(rest_list), 10):
        b_idx = i // 10 + 1
        batches.append({
            "batch_id": f"batch_{b_idx}_calib",
            "indices": rest_list[i:i+10],
            "is_preflight": False
        })
        
    def save_checkpoint(error_type=None):
        checkpoint['resume_safe_timestamp'] = datetime.now(timezone.utc).isoformat()
        if error_type:
            checkpoint['latest_error_type'] = error_type
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def process_batch(batch_desc, is_repair=False):
        batch_id = batch_desc['batch_id']
        indices = batch_desc['indices']
        
        # filter out completed
        indices = [idx for idx in indices if df.loc[idx, 'annotation_item_id'] not in checkpoint['completed_annotation_item_ids']]
        if not indices:
            return True, [] # already done
            
        b_df = df.loc[indices]
        
        input_text = ""
        for _, row in b_df.iterrows():
            summary = str(row.get('summary', ''))
            review = str(row.get('reviewText', ''))
            input_text += f"\n--- ITEM ---\nannotation_item_id: {row['annotation_item_id']}\nsummary: {summary}\nreviewText: {review}\n"
            
        prompt = prompt_template + f"\n\n**Taxonomy:**\n{taxonomy_text}\n\n**Schema Requirement:**\nProduce a JSON object matching this schema exactly:\n```json\n{schema_text}\n```\n\nCRITICAL INSTRUCTION: Keep 'Evidence' strings EXTREMELY short (MAXIMUM 5 to 7 words). Do not copy full sentences. The 'Evidence' MUST BE AN EXACT SUBSTRING of the text. Do not cut off words halfway (e.g. do not write 'disappoint' if the text says 'disappointing!'). You must output exactly {len(b_df)} items in the array. Evidence_Source must be exactly 'summary' or 'reviewText'.\n\n**Input:**\n{input_text}"
        
        if is_repair:
            prompt += "\n\nCRITICAL REPAIR: Previous response was structurally invalid or TRUNCATED. Make your Evidence quotes even shorter (1-3 words max). Ensure ALL items are completed exactly. Start/End must be exact 0-indexed character offsets of the evidence string within the source text."
            checkpoint['repair_request_count'] += 1
        else:
            checkpoint['request_count'] += 1
            
        raw_text, latency, status = call_gemini(model, prompt)
        
        if status == "ERROR":
            print(f"API ERROR RETURNED: {raw_text}")
            if "GenerateRequestsPerDay" in raw_text or "429" in raw_text:
                if "GenerateRequestsPerDay" in raw_text:
                    print("Daily Quota Exhausted! Checkpointing and stopping.")
                    save_checkpoint("DAILY_QUOTA_EXHAUSTED")
                    return False, []
                else:
                    # Short term 429
                    print("Short term 429. Retrying after 65 seconds...")
                    time.sleep(65)
                    raw_text, latency2, status = call_gemini(model, prompt)
                    latency += latency2
                    if "429" in raw_text:
                        print(f"API ERROR RETURNED (2nd try): {raw_text}")
                        print("Short term 429 again. Retrying 65s...")
                        time.sleep(65)
                        raw_text, latency3, status = call_gemini(model, prompt)
                        latency += latency3
                        if "429" in raw_text:
                            print(f"API ERROR RETURNED (3rd try): {raw_text}")
                            print("Daily Quota likely exhausted on 3rd try. Checkpointing.")
                            save_checkpoint("DAILY_QUOTA_EXHAUSTED")
                            return False, []
            else:
                print(f"Unknown error: {raw_text}")
                save_checkpoint(f"API_ERROR")
                return False, []
        
        raw_text_clean = raw_text.strip()
        if raw_text_clean.startswith("```json"):
            raw_text_clean = raw_text_clean[7:].strip()
            if raw_text_clean.endswith("```"):
                raw_text_clean = raw_text_clean[:-3].strip()
        elif raw_text_clean.startswith("```"):
            raw_text_clean = raw_text_clean[3:].strip()
            if raw_text_clean.endswith("```"):
                raw_text_clean = raw_text_clean[:-3].strip()
                
        try:
            data = json.loads(raw_text_clean)
            items = data.get("items", [])
        except json.JSONDecodeError as e:
            print(f"JSON decode error in batch! Raw text was:\n{raw_text}")
            return "REPAIR_NEEDED", []
            
        if not isinstance(items, list):
            return "REPAIR_NEEDED", []
            
        results = []
        expected_ids = set(b_df['annotation_item_id'].tolist())
        found_ids = set()
        
        for item in items:
            iid = item.get('annotation_item_id')
            if not iid or iid not in expected_ids:
                continue
            if iid in found_ids:
                continue # duplicate
            found_ids.add(iid)
            
            row = df[df['annotation_item_id'] == iid].iloc[0]
            summary = str(row.get('summary', ''))
            review = str(row.get('reviewText', ''))
            
            errors = validator.validate_item(item, summary, review)
            
            res = {
                'annotation_item_id': iid,
                'raw_response_snippet': json.dumps(item), # save specific item
                'parsed_json': item,
                'is_valid': len(errors) == 0,
                'errors': errors,
                'latency_per_item': latency / len(expected_ids),
                'batch_id': batch_id,
                'is_repaired': is_repair
            }
            results.append(res)
            
        # check missing
        missing = expected_ids - found_ids
        for m in missing:
            results.append({
                'annotation_item_id': m,
                'raw_response_snippet': "",
                'parsed_json': None,
                'is_valid': False,
                'errors': ['Missing from LLM output'],
                'latency_per_item': latency / len(expected_ids),
                'batch_id': batch_id,
                'is_repaired': is_repair
            })
            
        return True, results

    for b in batches:
        b_id = b['batch_id']
        if b_id in checkpoint['completed_batch_ids']:
            continue
            
        print(f"Processing {b_id}...")
        status, results = process_batch(b)
        
        if status is False:
            print("Execution halted due to quota or critical error.")
            sys.exit(0)
            
        if status == "REPAIR_NEEDED" or any(not r['is_valid'] for r in results):
            print(f"Batch {b_id} needs repair. Executing repair batch...")
            # Collect invalid IDs
            if status == "REPAIR_NEEDED":
                invalid_indices = [idx for idx in b['indices'] if df.loc[idx, 'annotation_item_id'] not in checkpoint['completed_annotation_item_ids']]
            else:
                invalid_ids = [r['annotation_item_id'] for r in results if not r['is_valid']]
                invalid_indices = [idx for idx in b['indices'] if df.loc[idx, 'annotation_item_id'] in invalid_ids]
                
            repair_b = {
                'batch_id': b_id + "_repair",
                'indices': invalid_indices
            }
            r_status, r_results = process_batch(repair_b, is_repair=True)
            if r_status is False:
                print("Execution halted during repair due to quota.")
                sys.exit(0)
            if r_status is True:
                # Merge results, prioritize repair
                if status != "REPAIR_NEEDED":
                    valid_results = [r for r in results if r['is_valid']]
                    results = valid_results + r_results
                else:
                    results = r_results
                
        # Validate post-repair
        batch_success = True
        if not results:
            batch_success = False
            
        for r in results:
            if r['is_valid']:
                checkpoint['completed_annotation_item_ids'].append(r['annotation_item_id'])
            else:
                checkpoint['failed_item_ids'].append(r['annotation_item_id'])
                batch_success = False
            checkpoint['all_results'].append(r)
            
        if b['is_preflight'] and not batch_success:
            print("PREFLIGHT_FAILED. Check failed_item_ids or logs.")
            save_checkpoint("PREFLIGHT_FAILED")
            break
            
        checkpoint['completed_batch_ids'].append(b_id)
        save_checkpoint()
        
    # Write exports v1_1
    all_res = checkpoint['all_results']
    
    if len(all_res) > 0:
        preflight_res = [r for r in all_res if 'preflight' in r['batch_id']]
        calib_res = [r for r in all_res if 'calib' in r['batch_id'] or 'preflight' in r['batch_id']]
        
        # Preflight artifacts
        if len(preflight_res) > 0:
            with open(base_dir / "gemini_preflight_responses_v1_1.jsonl", "w") as f:
                for r in preflight_res: f.write(json.dumps(r) + "\n")
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'is_valid': r['is_valid'],
                'errors': str(r['errors']),
            } for r in preflight_res]).to_csv(base_dir / "gemini_preflight_validation_v1_1.csv", index=False)
            
            pf_passed = all(r['is_valid'] for r in preflight_res)
            with open(base_dir / "gemini_preflight_report_v1_1.md", "w") as f:
                if pf_passed:
                    f.write("# Preflight Report\n\n**Verdict:** PASSED\n\nAll samples parsed mechanically.")
                else:
                    f.write("# Preflight Report\n\n**Verdict:** FAILED\n\nSome samples failed.")
                    
        # Calibration artifacts
        with open(base_dir / "llm_calibration_raw_responses_v1_1.jsonl", "w") as f:
            for r in calib_res: f.write(json.dumps(r) + "\n")
            
        valid_res = [r for r in calib_res if r['is_valid']]
        invalid_res = [r for r in calib_res if not r['is_valid']]
        
        if valid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                **r['parsed_json']
            } for r in valid_res]).to_parquet(base_dir / "llm_calibration_labels_v1_1.parquet", index=False)
            
        if invalid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'errors': str(r['errors'])
            } for r in invalid_res]).to_csv(base_dir / "llm_calibration_invalid_queue_v1_1.csv", index=False)
        else:
            pd.DataFrame(columns=['annotation_item_id', 'errors']).to_csv(base_dir / "llm_calibration_invalid_queue_v1_1.csv", index=False)
            
        pd.DataFrame([{
            'annotation_item_id': r['annotation_item_id'],
            'is_valid': r['is_valid'],
            'errors': str(r['errors'])
        } for r in calib_res]).to_csv(base_dir / "llm_calibration_validation_v1_1.csv", index=False)
        
        manifest = {
            'provider': 'google',
            'model': model_id,
            'sdk_version': genai.__version__,
            'temperature': 0,
            'max_output_tokens': 8192,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prompt_hash': prompt_hash,
            'taxonomy_hash': taxonomy_hash,
            'manifest_hash': manifest_hash
        }
        with open(base_dir / "gemini_call_manifest_v1_1.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(base_dir / "llm_calibration_provenance_v1_1.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        # Report
        report = f"""# Phase 2B-1: LLM Calibration Report v1.1

## Descriptive Statistics
- **Total Processed:** {len(calib_res)}
- **Valid Count:** {len(valid_res)}
- **Invalid Count:** {len(invalid_res)}
- **JSON / Taxonomy Validity Rate:** {len(valid_res) / max(len(calib_res), 1):.1%}
- **Request Count Used:** {checkpoint['request_count']}
- **Repair Request Count Used:** {checkpoint['repair_request_count']}

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding semantic accuracy, human agreement, causal evidence, predictive utility, or generalization.
"""
        with open(base_dir / "phase2b1_llm_calibration_report_v1_1.md", "w") as f:
            f.write(report)
            
    print("Run completed successfully or gracefully suspended.")

if __name__ == "__main__":
    run_phase2b1()
