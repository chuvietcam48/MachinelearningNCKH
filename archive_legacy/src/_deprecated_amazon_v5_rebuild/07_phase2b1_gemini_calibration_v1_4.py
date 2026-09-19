import os
import sys
import json
import time
import hashlib
import argparse
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
            
            if evidence:
                if source not in ['summary', 'reviewText']:
                    errors.append(f"Invalid Evidence_Source{i}: {source}")
                    continue
                    
                target_text = summary if source == 'summary' else review
                if target_text is None: target_text = ""
                
                start_idx = target_text.find(evidence)
                last_idx = target_text.rfind(evidence)
                
                if start_idx == -1:
                    errors.append(f"Evidence{i} not found in {source}.")
                elif start_idx != last_idx:
                    errors.append(f"Evidence{i} occurs multiple times in {source}. Cannot unambiguously derive bounds.")
                else:
                    item[f'Evidence_Start{i}'] = start_idx
                    item[f'Evidence_End{i}'] = start_idx + len(evidence)

        mixed = item.get('Review_Mixed_Flag')
        if not isinstance(mixed, bool):
            errors.append("Review_Mixed_Flag must be boolean")
            
        for k, v in item.items():
            if isinstance(v, str) and k.startswith('Aspect'):
                if 'star' in v.lower() or 'rating' in v.lower():
                    errors.append(f"Forbidden text in {k}: {v}")
                    
        return errors

def call_gemini(model, prompt, is_dry_run=False, b_df=None, is_repair=False):
    import json, time
    start_time = time.time()
    
    if is_dry_run:
        import re
        batch_name = re.search(r'Batch ID: (batch_\d+_(preflight|calib))', prompt).group(1)
        print(f"DEBUG: batch_name = '{batch_name}'")
        items = []
        
        # Test malformed JSON
        if not is_repair and batch_name == "batch_1_calib":
            return "{bad_json", 0.5, "SUCCESS"
            
        for i, row in b_df.iterrows():
            iid = row['annotation_item_id']
            text = str(row.get('reviewText', ''))
            
            # Missing item (skip appending entirely)
            if not is_repair and batch_name == "batch_0_preflight" and iid == b_df.iloc[4]['annotation_item_id']:
                continue
                
            item = {
                "annotation_item_id": iid,
                "Aspect1": "Customer_Service_Returns",
                "Polarity1": "Negative",
                "Evidence_Source1": "reviewText",
                "Evidence1": text[:15] if len(text) > 15 else text,
                "Aspect2": None,
                "Polarity2": None,
                "Evidence_Source2": None,
                "Evidence2": None,
                "Review_Mixed_Flag": False
            }
            
            if not is_repair and batch_name == "batch_0_preflight":
                if iid == b_df.iloc[1]['annotation_item_id']:
                    # Missing annotation_item_id key
                    del item["annotation_item_id"]
                elif iid == b_df.iloc[2]['annotation_item_id']:
                    # Altered annotation_item_id
                    item["annotation_item_id"] = "altered_id_123"
                elif iid == b_df.iloc[3]['annotation_item_id']:
                    # Duplicate annotation_item_id (copy first item's ID)
                    item["annotation_item_id"] = b_df.iloc[0]['annotation_item_id']
                    
            if not is_repair and batch_name == "batch_2_calib":
                if iid == b_df.iloc[0]['annotation_item_id']:
                    item["Aspect1"] = "Invalid_Taxonomy_Value"
                elif iid == b_df.iloc[1]['annotation_item_id']:
                    item["Evidence1"] = "This exact string will never be in the text 123456789"
                elif iid == b_df.iloc[2]['annotation_item_id']:
                    if " " in text and text.count(" ") > 5:
                        item["Evidence1"] = " " 
                    elif len(text) > 2:
                        item["Evidence1"] = text[0]
                        
            if is_repair:
                item["Evidence1"] = text[:15] if len(text) > 15 else text
                
            items.append(item)
            
        # Test top-level list
        if is_repair and batch_name == "batch_0_preflight_repair":
            return json.dumps(items), 0.5, "SUCCESS"
            
        return json.dumps({"items": items}), 0.5, "SUCCESS"
        
    # Real call
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


def run_contract_tests(schema_text, prompt_template):
    print("Running v1.4 Contract Tests...")
    schema = json.loads(schema_text)
    
    # Prove every field required by the parser is required by the schema
    props = schema['items'][0].keys()
    parser_reqs = ['annotation_item_id', 'Aspect1', 'Polarity1', 'Evidence_Source1', 'Evidence1', 'Review_Mixed_Flag']
    for req in parser_reqs:
        assert req in props, f"Contract failure: {req} missing from schema"
        
    # Prove schema is represented in prompt
    assert "annotation_item_id" in prompt_template, "Contract failure: annotation_item_id not explicitly required in prompt"
    assert "items" in prompt_template, "Contract failure: top level items object not in prompt"
    
    print("Contract Tests Passed.")


def run_phase2b1(is_dry_run=False):
    print(f"Starting Phase 2B-1: Gemini API Quota-Aware Batching v1.4 (Dry Run: {is_dry_run})")
    if not is_dry_run:
        setup_gemini()
    
    base_dir = Path("outputs/amazon_v5_rebuild/annotation")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = base_dir / "calibration_sample_manifest_v1_1.csv"
    df = pd.read_csv(manifest_path)
    df['annotation_item_id'] = df['reviewerID'] + '_' + df['episode_id'].astype(str)
    
    with open(base_dir / "gemini_semantic_prompt_template_v1_4.md", "r") as f:
        prompt_template = f.read()
    with open(base_dir / "taxonomy_v1_1.md", "r") as f:
        taxonomy_text = f.read()
    with open(base_dir / "semantic_output_schema_v1_4.json", "r") as f:
        schema_text = f.read()
        
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()
    taxonomy_hash = hashlib.sha256(taxonomy_text.encode()).hexdigest()
    manifest_hash = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    
    validator = OutputValidator(base_dir / "taxonomy_v1_1.md")
    if is_dry_run:
        run_contract_tests(schema_text, prompt_template)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash") if not is_dry_run else None
    
    ckpt_path = base_dir / ("gemini_batch_run_checkpoint_v1_4" + ("_dryrun" if is_dry_run else "") + ".json")
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
            "all_results": []
        }
    
    # Select deterministic 8 reviews for preflight
    df['text_len'] = df['reviewText'].fillna('').str.len()
    preflight_indices = df.index[:8] # just take first 8 deterministic
    rest_indices = df.index[8:]
    
    batches = []
    batches.append({
        "batch_id": "batch_0_preflight",
        "indices": preflight_indices,
        "is_preflight": True
    })
    
    rest_list = list(rest_indices)
    for i in range(0, len(rest_list), 8):
        b_idx = i // 8 + 1
        batches.append({
            "batch_id": f"batch_{b_idx}_calib",
            "indices": rest_list[i:i+8],
            "is_preflight": False
        })
        
    total_planned = 15
    repair_budget = 3
    
    if is_dry_run:
        budget_report = {
            "planned_requests": total_planned,
            "max_repair_budget": repair_budget,
            "total_reviews": len(df),
            "preflight_size": len(preflight_indices),
            "calibration_batches": len(batches) - 1,
            "batch_size": 8
        }
        with open(base_dir / "gemini_request_budget_v1_4.json", "w") as f:
            json.dump(budget_report, f, indent=2)

    def save_checkpoint():
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def process_batch(batch_desc, is_repair=False):
        batch_id = batch_desc['batch_id']
        indices = batch_desc['indices']
        
        # Filter completed (this enforces rule: "do not call them again")
        indices = [idx for idx in indices if df.loc[idx, 'annotation_item_id'] not in checkpoint['completed_annotation_item_ids']]
        if not indices:
            return True, [] 
            
        if not is_repair:
            if checkpoint['request_count'] >= total_planned:
                print("Budget exceeded.")
                return False, []
            checkpoint['request_count'] += 1
        else:
            if checkpoint['repair_request_count'] >= repair_budget:
                print("Repair budget exceeded.")
                return False, []
            checkpoint['repair_request_count'] += 1
            
        b_df = df.loc[indices]
        
        input_text = ""
        for _, row in b_df.iterrows():
            summary = str(row.get('summary', ''))
            review = str(row.get('reviewText', ''))
            input_text += f"\\n--- ITEM ---\\nannotation_item_id: {row['annotation_item_id']}\\nsummary: {summary}\\nreviewText: {review}\\n"
            
        prompt = prompt_template + f"\\n\\n**Taxonomy:**\\n{taxonomy_text}\\n\\n**Schema Requirement:**\\nProduce a JSON object matching this schema exactly:\\n```json\\n{schema_text}\\n```\\n\\nCRITICAL INSTRUCTION: Keep 'Evidence' strings EXTREMELY short (MAXIMUM 5 to 7 words). Do not copy full sentences. The 'Evidence' MUST BE AN EXACT SUBSTRING of the text. Do not cut off words halfway (e.g. do not write 'disappoint' if the text says 'disappointing!'). You must output exactly {len(b_df)} items in the array. Evidence_Source must be exactly 'summary' or 'reviewText'. Batch ID: {batch_id}\\n\\n**Input:**\\n{input_text}"
        
        raw_text, latency, status = call_gemini(model, prompt, is_dry_run=is_dry_run, b_df=b_df, is_repair=is_repair)
        
        if status == "ERROR" and "429" in raw_text:
            print("Quota error. Halting.")
            return False, []
        
        results = []
        response_shape = 'invalid_shape'
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            parsed = json.loads(cleaned)
            
            if isinstance(parsed, dict) and 'items' in parsed:
                items = parsed['items']
                response_shape = 'object_items'
            elif isinstance(parsed, list):
                items = parsed
                response_shape = 'top_level_list'
                print("Warning: Received top-level list, deviating from strictly requested schema.")
            else:
                raise ValueError("JSON must be a list or a dict containing 'items'")
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return "REPAIR_NEEDED", []
            
        expected_ids = set(b_df['annotation_item_id'].tolist())
        found_ids = set()
        
        for item in items:
            iid = item.get('annotation_item_id')
            if not iid or iid not in expected_ids:
                continue
            if iid in found_ids:
                continue 
            found_ids.add(iid)
            
            row = df[df['annotation_item_id'] == iid].iloc[0]
            summary = str(row.get('summary', ''))
            review = str(row.get('reviewText', ''))
            
            errors = validator.validate_item(item, summary, review)
            
            res = {
                'annotation_item_id': iid,
                'raw_response_snippet': json.dumps(item),
                'parsed_json': item,
                'is_valid': len(errors) == 0,
                'errors': errors,
                'batch_id': batch_id,
                'is_repaired': is_repair,
                'response_shape': response_shape
            }
            results.append(res)
            
        missing = expected_ids - found_ids
        for m in missing:
            results.append({
                'annotation_item_id': m,
                'raw_response_snippet': "",
                'parsed_json': None,
                'is_valid': False,
                'errors': ['Missing from LLM output'],
                'batch_id': batch_id,
                'is_repaired': is_repair,
                'response_shape': response_shape
            })
            
        return True, results

    for b in batches:
        b_id = b['batch_id']
        if b_id in checkpoint['completed_batch_ids']:
            continue
            
        print(f"Processing {b_id}...")
        status, results = process_batch(b)
        
        if status is False:
            print("Execution halted.")
            break
            
        if status == "REPAIR_NEEDED" or any(not r['is_valid'] for r in results):
            print(f"Batch {b_id} needs repair. Executing repair batch...")
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
                print("Execution halted during repair.")
                break
            if r_status is True:
                if status != "REPAIR_NEEDED":
                    valid_results = [r for r in results if r['is_valid']]
                    results = valid_results + r_results
                else:
                    results = r_results
                
        batch_success = True
        if not results:
            batch_success = False
            
        for r in results:
            if r['is_valid']:
                if r['annotation_item_id'] not in checkpoint['completed_annotation_item_ids']:
                    checkpoint['completed_annotation_item_ids'].append(r['annotation_item_id'])
            else:
                checkpoint['failed_item_ids'].append(r['annotation_item_id'])
                batch_success = False
            checkpoint['all_results'].append(r)
            
        if b['is_preflight'] and not batch_success:
            print("PREFLIGHT_FAILED. Check failed_item_ids or logs.")
            checkpoint['completed_batch_ids'].append(b_id)
            save_checkpoint()
            break
            
        checkpoint['completed_batch_ids'].append(b_id)
        save_checkpoint()
        
    all_res = checkpoint['all_results']
    if len(all_res) > 0 and is_dry_run:
        preflight_res = [r for r in all_res if 'preflight' in r['batch_id']]
        pf_passed = all(r['is_valid'] for r in preflight_res) if preflight_res else False
        with open(base_dir / "gemini_batch_dry_run_report_v1_4.md", "w") as f:
            f.write("# Dry Run Report v1.4\n\n")
            f.write(f"**Preflight Passed:** {pf_passed}\n")
            f.write(f"**Total Requests Simulated:** {checkpoint['request_count']}\n")
            f.write(f"**Total Repair Requests Simulated:** {checkpoint['repair_request_count']}\n")
            f.write(f"**Items Completed:** {len(checkpoint['completed_annotation_item_ids'])}\n")
    elif len(all_res) > 0 and not is_dry_run:
        from datetime import datetime, timezone
        calib_res = [r for r in all_res if 'calib' in r['batch_id'] or 'preflight' in r['batch_id']]
        
        preflight_res = [r for r in all_res if 'preflight' in r['batch_id']]
        if preflight_res:
            with open(base_dir / "gemini_preflight_responses_v1_4.jsonl", "w") as f:
                for r in preflight_res: f.write(json.dumps(r) + "\n")
            
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'is_valid': r['is_valid'],
                'errors': str(r['errors'])
            } for r in preflight_res]).to_csv(base_dir / "gemini_preflight_validation_v1_4.csv", index=False)
            
            valid_pf = [r for r in preflight_res if r['is_valid']]
            invalid_pf = [r for r in preflight_res if not r['is_valid']]
            
            report_pf = f"""# Phase 2B-1: Gemini Preflight Report v1.4
            
## Preflight Statistics
- **Total Processed:** {len(preflight_res)}
- **Valid Count:** {len(valid_pf)}
- **Invalid Count:** {len(invalid_pf)}
- **Passed:** {len(invalid_pf) == 0}

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding human verification, ground facts, or utility for downstream tasks.
"""
            with open(base_dir / "gemini_preflight_report_v1_4.md", "w") as f:
                f.write(report_pf)
                
        with open(base_dir / "llm_calibration_raw_responses_v1_4.jsonl", "w") as f:
            for r in calib_res: f.write(json.dumps(r) + "\n")
            
        valid_res = [r for r in calib_res if r['is_valid']]
        invalid_res = [r for r in calib_res if not r['is_valid']]
        
        if valid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                **r['parsed_json']
            } for r in valid_res]).to_parquet(base_dir / "llm_calibration_labels_v1_4.parquet", index=False)
            
        if invalid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'errors': str(r['errors'])
            } for r in invalid_res]).to_csv(base_dir / "llm_calibration_invalid_queue_v1_4.csv", index=False)
        else:
            pd.DataFrame(columns=['annotation_item_id', 'errors']).to_csv(base_dir / "llm_calibration_invalid_queue_v1_4.csv", index=False)
            
        pd.DataFrame([{
            'annotation_item_id': r['annotation_item_id'],
            'is_valid': r['is_valid'],
            'errors': str(r['errors'])
        } for r in calib_res]).to_csv(base_dir / "llm_calibration_validation_v1_4.csv", index=False)
        
        shapes_seen = list(set([r.get('response_shape', 'unknown') for r in all_res]))
        manifest = {
            'provider': 'google',
            'model': "gemini-2.5-flash",
            'shapes_seen': shapes_seen,
            'schema_deviation_warning': 'schema_deviation_top_level_list' if 'top_level_list' in shapes_seen else False,
            'sdk_version': genai.__version__,
            'temperature': 0,
            'max_output_tokens': 8192,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prompt_hash': prompt_hash,
            'taxonomy_hash': taxonomy_hash,
            'manifest_hash': manifest_hash
        }
        with open(base_dir / "gemini_call_manifest_v1_4.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(base_dir / "llm_calibration_provenance_v1_4.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        report = f"""# Phase 2B-1: LLM-assisted silver calibration labels Report v1.4

## Descriptive Statistics
- **Total Processed:** {len(calib_res)}
- **Valid Count:** {len(valid_res)}
- **Invalid Count:** {len(invalid_res)}
- **JSON / Taxonomy / Evidence Validity Rate:** {len(valid_res) / max(len(calib_res), 1):.1%}
- **Request Count Used:** {checkpoint['request_count']}
- **Repair Request Count Used:** {checkpoint['repair_request_count']}

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding human verification, ground facts, or utility for downstream tasks.
"""
        with open(base_dir / "phase2b1_llm_calibration_report_v1_4.md", "w") as f:
            f.write(report)
            
    print("Run completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without calling API")
    args = parser.parse_args()
    run_phase2b1(is_dry_run=args.dry_run)
