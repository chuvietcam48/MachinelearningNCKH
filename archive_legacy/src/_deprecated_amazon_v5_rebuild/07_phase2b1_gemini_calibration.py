import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
import pandas as pd
import openai
from pathlib import Path

client = None

def setup_openai():
    global client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

class OutputValidator:
    def __init__(self, taxonomy_path, schema_path):
        with open(taxonomy_path, 'r') as f:
            self.taxonomy_text = f.read()
            
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)
            
        # Parse taxonomy aspects from markdown
        self.allowed_aspects = []
        for line in self.taxonomy_text.split('\n'):
            if line.startswith('- `'):
                aspect = line.split('`')[1]
                self.allowed_aspects.append(aspect)
                
        self.allowed_polarities = ["Positive", "Negative", "Neutral", "NotApplicable"]
        
    def validate(self, response_text, summary, review):
        result = {
            'is_valid': True,
            'errors': [],
            'parsed_json': None
        }
        
        try:
            data = json.loads(response_text)
            result['parsed_json'] = data
        except json.JSONDecodeError as e:
            result['is_valid'] = False
            result['errors'].append(f"Invalid JSON: {str(e)}")
            return result
            
        # Check allowed fields
        expected_keys = set(self.schema.keys())
        actual_keys = set(data.keys())
        if actual_keys - expected_keys:
            result['is_valid'] = False
            result['errors'].append(f"Unexpected fields: {actual_keys - expected_keys}")
        
        # Check Aspect1
        if data.get('Aspect1') not in self.allowed_aspects:
            result['is_valid'] = False
            result['errors'].append(f"Invalid Aspect1: {data.get('Aspect1')}")
            
        if data.get('Polarity1') not in self.allowed_polarities:
            result['is_valid'] = False
            result['errors'].append(f"Invalid Polarity1: {data.get('Polarity1')}")
            
        # Check Aspect2 (can be null)
        if data.get('Aspect2') is not None and data.get('Aspect2') not in self.allowed_aspects:
            result['is_valid'] = False
            result['errors'].append(f"Invalid Aspect2: {data.get('Aspect2')}")
            
        if data.get('Polarity2') is not None and data.get('Polarity2') not in self.allowed_polarities:
            result['is_valid'] = False
            result['errors'].append(f"Invalid Polarity2: {data.get('Polarity2')}")
            
        # Evidence checks
        for i in [1, 2]:
            evidence = data.get(f'Evidence{i}')
            source = data.get(f'Evidence_Source{i}')
            start = data.get(f'Evidence_Start{i}')
            end = data.get(f'Evidence_End{i}')
            
            if evidence:
                if source not in ['summary', 'reviewText']:
                    result['is_valid'] = False
                    result['errors'].append(f"Invalid Evidence_Source{i}: {source}")
                    continue
                    
                target_text = summary if source == 'summary' else review
                if target_text is None: target_text = ""
                
                if evidence not in target_text:
                    result['is_valid'] = False
                    result['errors'].append(f"Evidence{i} is not a substring of {source}")
                    
                if start is not None and end is not None:
                    recovered = target_text[start:end]
                    if recovered != evidence:
                        result['is_valid'] = False
                        result['errors'].append(f"Evidence{i} bounds mismatch: recovered '{recovered}' vs '{evidence}'")

        # Mixed flag check
        mixed = data.get('Review_Mixed_Flag')
        if not isinstance(mixed, bool):
            result['is_valid'] = False
            result['errors'].append("Review_Mixed_Flag must be boolean")
        elif mixed:
            has_pos = False
            has_neg = False
            for i in [1, 2]:
                pol = data.get(f'Polarity{i}')
                asp = data.get(f'Aspect{i}')
                if asp and asp != 'None':
                    if pol == 'Positive': has_pos = True
                    if pol == 'Negative': has_neg = True
            if not (has_pos and has_neg):
                result['is_valid'] = False
                result['errors'].append("Review_Mixed_Flag is true but review lacks both positive and negative concrete aspects")
                
        # Heuristic check for ratings / forbidden info in aspect
        for k, v in data.items():
            if isinstance(v, str) and k.startswith('Aspect'):
                if 'star' in v.lower() or 'rating' in v.lower():
                    result['is_valid'] = False
                    result['errors'].append(f"Forbidden text in {k}: {v}")
                    
        return result

def call_openai(prompt, schema_obj, retry_count=0):
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        latency = time.time() - start_time
        return response.choices[0].message.content, latency, "SUCCESS"
    except Exception as e:
        latency = time.time() - start_time
        return str(e), latency, "ERROR"

def run_phase2b1():
    print("Starting Phase 2B-1: OpenAI API Preflight")
    setup_openai()
    
    base_dir = Path("outputs/amazon_v5_rebuild/annotation")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = base_dir / "calibration_sample_manifest_v1.csv"
    df = pd.read_csv(manifest_path)
    
    with open(base_dir / "gemini_semantic_prompt_template_v1_1.md", "r") as f:
        prompt_template = f.read()
    with open(base_dir / "taxonomy_v1_1.md", "r") as f:
        taxonomy_text = f.read()
    with open(base_dir / "semantic_output_schema_v1_1.json", "r") as f:
        schema_text = f.read()
        schema_obj = json.loads(schema_text)
        
    prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()
    taxonomy_hash = hashlib.sha256(taxonomy_text.encode()).hexdigest()
    manifest_hash = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    
    validator = OutputValidator(base_dir / "taxonomy_v1_1.md", base_dir / "semantic_output_schema_v1_1.json")
    
    model_id = "gpt-4o-mini"
    
    # Select deterministic 10 reviews
    df['text_len'] = df['reviewText'].fillna('').str.len()
    
    # 1. Shortest review
    idx1 = df['text_len'].idxmin()
    # 2. Longest review
    idx2 = df['text_len'].idxmax()
    # 3. High rating generic praise
    idx3 = df[(df['overall'] == 5) & (df['text_len'] < 100)].index[0]
    # 4. Low rating generic complaint
    idx4 = df[(df['overall'] == 1) & (df['text_len'] < 100)].index[0]
    # 5-10. Next 6 diverse
    other_indices = df[~df.index.isin([idx1, idx2, idx3, idx4])].index[:6]
    
    preflight_indices = [idx1, idx2, idx3, idx4] + list(other_indices)
    preflight_df = df.loc[preflight_indices].copy()
    
    def process_row(row, allow_retry=True):
        summary = str(row.get('summary', ''))
        review = str(row.get('reviewText', ''))
        
        prompt = prompt_template + f"\n\n**Schema Requirement:**\nProduce a JSON object matching this schema exactly:\n```json\n{schema_text}\n```\n\n**Input:**\nSUMMARY: {summary}\nREVIEW: {review}"
        
        raw_text, latency, status = call_openai(prompt, schema_obj)
        if "429" in raw_text:
            print("Rate limited! Sleeping 5s...")
            time.sleep(5)
            raw_text, latency2, status = call_openai(prompt, schema_obj)
            latency += latency2
        
        val_res = validator.validate(raw_text, summary, review)
        retries = 0
        
        # Bounded Retry
        if not val_res['is_valid'] and allow_retry and status == "SUCCESS":
            retries = 1
            print(f"Retrying row {row.name} due to: {val_res['errors']}")
            repair_prompt = prompt + f"\n\nYOUR PREVIOUS OUTPUT WAS INVALID DUE TO:\n{val_res['errors']}\nPLEASE REPAIR AND RETURN VALID JSON."
            time.sleep(1) # rate limit backoff
            raw_text, latency2, status2 = call_openai(repair_prompt, schema_obj)
            if "429" in raw_text:
                time.sleep(5)
                raw_text, latency2, status2 = call_openai(repair_prompt, schema_obj)
            val_res = validator.validate(raw_text, summary, review)
            latency += latency2
            status = status2
            
        return {
            'annotation_item_id': f"{row['reviewerID']}_{row['episode_id']}",
            'raw_response': raw_text,
            'parsed_json': val_res['parsed_json'],
            'is_valid': val_res['is_valid'],
            'errors': val_res['errors'],
            'latency': latency,
            'status': status,
            'retries': retries
        }

    # =========================================================================
    # PREFLIGHT
    # =========================================================================
    print("Running Preflight...")
    preflight_results = []
    
    for _, row in preflight_df.iterrows():
        res = process_row(row)
        preflight_results.append(res)
        time.sleep(1) # Avoid rate limits
        
    failures = [r for r in preflight_results if not r['is_valid'] or r['status'] == "ERROR"]
    total_retries = sum(r['retries'] for r in preflight_results)
    
    preflight_passed = len(failures) == 0 and total_retries <= 1
    
    # Save preflight artifacts
    with open(base_dir / "gemini_preflight_responses_v1.jsonl", "w") as f:
        for r in preflight_results:
            f.write(json.dumps(r) + "\n")
            
    pd.DataFrame([{
        'annotation_item_id': r['annotation_item_id'],
        'is_valid': r['is_valid'],
        'errors': str(r['errors']),
        'retries': r['retries'],
        'latency': r['latency']
    } for r in preflight_results]).to_csv(base_dir / "gemini_preflight_validation_v1.csv", index=False)
    
    manifest = {
        'provider': 'openai',
        'model': model_id,
        'sdk_version': openai.__version__,
        'temperature': 0,
        'max_output_tokens': 1024,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'prompt_hash': prompt_hash,
        'taxonomy_hash': taxonomy_hash,
        'manifest_hash': manifest_hash
    }
    with open(base_dir / "gemini_call_manifest_v1.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    if not preflight_passed:
        with open(base_dir / "gemini_preflight_report_v1.md", "w") as f:
            f.write(f"# Preflight Report\n\n**Verdict:** PREFLIGHT_FAILED_PENDING_PROMPT_REPAIR\n\nFailures: {len(failures)}\nRetries: {total_retries}")
        print("PREFLIGHT_FAILED_PENDING_PROMPT_REPAIR")
        return
        
    with open(base_dir / "gemini_preflight_report_v1.md", "w") as f:
        f.write("# Preflight Report\n\n**Verdict:** PASSED\n\nAll 10 samples parsed mechanically according to schema and rules.")

    # =========================================================================
    # CALIBRATION
    # =========================================================================
    print("Preflight passed. Running Calibration...")
    remaining_df = df[~df.index.isin(preflight_indices)]
    
    calibration_results = list(preflight_results) # start with the 10 already done
    
    for i, (_, row) in enumerate(remaining_df.iterrows()):
        res = process_row(row)
        calibration_results.append(res)
        time.sleep(1)
        if i % 10 == 0:
            print(f"Processed {len(calibration_results)}/120")
            
    # Save artifacts
    with open(base_dir / "llm_calibration_raw_responses_v1.jsonl", "w") as f:
        for r in calibration_results:
            f.write(json.dumps(r) + "\n")
            
    valid_results = [r for r in calibration_results if r['is_valid']]
    invalid_results = [r for r in calibration_results if not r['is_valid']]
    
    if valid_results:
        valid_df = pd.DataFrame([{
            'annotation_item_id': r['annotation_item_id'],
            **r['parsed_json']
        } for r in valid_results])
        valid_df.to_parquet(base_dir / "llm_calibration_labels_v1.parquet", index=False)
        
    if invalid_results:
        pd.DataFrame([{
            'annotation_item_id': r['annotation_item_id'],
            'errors': str(r['errors']),
            'raw_response': r['raw_response']
        } for r in invalid_results]).to_csv(base_dir / "llm_calibration_invalid_queue_v1.csv", index=False)
    else:
        # Create empty
        pd.DataFrame(columns=['annotation_item_id', 'errors', 'raw_response']).to_csv(base_dir / "llm_calibration_invalid_queue_v1.csv", index=False)
        
    pd.DataFrame([{
        'annotation_item_id': r['annotation_item_id'],
        'is_valid': r['is_valid'],
        'errors': str(r['errors']),
        'retries': r['retries']
    } for r in calibration_results]).to_csv(base_dir / "llm_calibration_validation_v1.csv", index=False)
    
    with open(base_dir / "llm_calibration_provenance_v1.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    # Descriptive frequencies
    aspects = []
    mixed_count = 0
    for r in valid_results:
        j = r['parsed_json']
        if j.get('Aspect1'): aspects.append(j['Aspect1'])
        if j.get('Aspect2'): aspects.append(j['Aspect2'])
        if j.get('Review_Mixed_Flag'): mixed_count += 1
        
    asp_freq = pd.Series(aspects).value_counts().to_dict()
    
    report = f"""# Phase 2B-1: LLM Calibration Report v1

## Descriptive Statistics
- **Total Processed:** {len(calibration_results)}
- **Valid Count:** {len(valid_results)}
- **Invalid Count:** {len(invalid_results)}
- **JSON / Taxonomy Validity Rate:** {len(valid_results) / len(calibration_results):.1%}
- **Total Retries Used:** {sum(r['retries'] for r in calibration_results)}
- **Mixed Flag True:** {mixed_count}

## Aspect Frequencies
```json
{json.dumps(asp_freq, indent=2)}
```

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding semantic accuracy, human agreement, causal evidence, predictive utility, or generalization.
"""
    with open(base_dir / "phase2b1_llm_calibration_report_v1.md", "w") as f:
        f.write(report)
        
    print("Phase 2B-1 complete.")

if __name__ == "__main__":
    run_phase2b1()
