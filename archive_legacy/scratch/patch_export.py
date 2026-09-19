import re

file_path = "c:\\Users\\Admin\\OneDrive\\Máy tính\\MachineLearning-1\\MachinelearningNCKH\\src\\amazon_v5_rebuild\\07_phase2b1_gemini_calibration_v1_2.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

export_logic = """
    elif len(all_res) > 0 and not is_dry_run:
        calib_res = [r for r in all_res if 'calib' in r['batch_id'] or 'preflight' in r['batch_id']]
        
        with open(base_dir / "llm_calibration_raw_responses_v1_2.jsonl", "w") as f:
            for r in calib_res: f.write(json.dumps(r) + "\\n")
            
        valid_res = [r for r in calib_res if r['is_valid']]
        invalid_res = [r for r in calib_res if not r['is_valid']]
        
        if valid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                **r['parsed_json']
            } for r in valid_res]).to_parquet(base_dir / "llm_calibration_labels_v1_2.parquet", index=False)
            
        if invalid_res:
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'errors': str(r['errors'])
            } for r in invalid_res]).to_csv(base_dir / "llm_calibration_invalid_queue_v1_2.csv", index=False)
        else:
            pd.DataFrame(columns=['annotation_item_id', 'errors']).to_csv(base_dir / "llm_calibration_invalid_queue_v1_2.csv", index=False)
            
        pd.DataFrame([{
            'annotation_item_id': r['annotation_item_id'],
            'is_valid': r['is_valid'],
            'errors': str(r['errors'])
        } for r in calib_res]).to_csv(base_dir / "llm_calibration_validation_v1_2.csv", index=False)
        
        manifest = {
            'provider': 'google',
            'model': "gemini-2.5-flash",
            'sdk_version': genai.__version__,
            'temperature': 0,
            'max_output_tokens': 8192,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'prompt_hash': prompt_hash,
            'taxonomy_hash': taxonomy_hash,
            'manifest_hash': manifest_hash
        }
        with open(base_dir / "gemini_call_manifest_v1_2.json", "w") as f:
            json.dump(manifest, f, indent=2)
        with open(base_dir / "llm_calibration_provenance_v1_2.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        report = f\"\"\"# Phase 2B-1: LLM-assisted silver calibration labels Report v1.2

## Descriptive Statistics
- **Total Processed:** {len(calib_res)}
- **Valid Count:** {len(valid_res)}
- **Invalid Count:** {len(invalid_res)}
- **JSON / Taxonomy / Evidence Validity Rate:** {len(valid_res) / max(len(calib_res), 1):.1%}
- **Request Count Used:** {checkpoint['request_count']}
- **Repair Request Count Used:** {checkpoint['repair_request_count']}

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding human verification, ground facts, or utility for downstream tasks.
\"\"\"
        with open(base_dir / "phase2b1_llm_calibration_report_v1_2.md", "w") as f:
            f.write(report)
"""

# Find where to inject it
# it's right after `if len(all_res) > 0 and is_dry_run:...`
content = content.replace("f.write(f\"**Items Completed:** {len(checkpoint['completed_annotation_item_ids'])}\\n\")", 
                          "f.write(f\"**Items Completed:** {len(checkpoint['completed_annotation_item_ids'])}\\n\")" + export_logic)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
"""
