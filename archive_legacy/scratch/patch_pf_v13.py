import re

file_path = "c:\\Users\\Admin\\OneDrive\\Máy tính\\MachineLearning-1\\MachinelearningNCKH\\src\\amazon_v5_rebuild\\07_phase2b1_gemini_calibration_v1_3.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add preflight specific exports
old_export_logic = """        with open(base_dir / "llm_calibration_raw_responses_v1_3.jsonl", "w") as f:
            for r in calib_res: f.write(json.dumps(r) + "\\n")"""

new_export_logic = """        preflight_res = [r for r in all_res if 'preflight' in r['batch_id']]
        if preflight_res:
            with open(base_dir / "gemini_preflight_responses_v1_3.jsonl", "w") as f:
                for r in preflight_res: f.write(json.dumps(r) + "\\n")
            
            pd.DataFrame([{
                'annotation_item_id': r['annotation_item_id'],
                'is_valid': r['is_valid'],
                'errors': str(r['errors'])
            } for r in preflight_res]).to_csv(base_dir / "gemini_preflight_validation_v1_3.csv", index=False)
            
            valid_pf = [r for r in preflight_res if r['is_valid']]
            invalid_pf = [r for r in preflight_res if not r['is_valid']]
            
            report_pf = f\"\"\"# Phase 2B-1: Gemini Preflight Report v1.3
            
## Preflight Statistics
- **Total Processed:** {len(preflight_res)}
- **Valid Count:** {len(valid_pf)}
- **Invalid Count:** {len(invalid_pf)}
- **Passed:** {len(invalid_pf) == 0}

**Disclaimer:** These metrics represent structural processing and raw distribution only. No claim is made regarding human verification, ground facts, or utility for downstream tasks.
\"\"\"
            with open(base_dir / "gemini_preflight_report_v1_3.md", "w") as f:
                f.write(report_pf)
                
        with open(base_dir / "llm_calibration_raw_responses_v1_3.jsonl", "w") as f:
            for r in calib_res: f.write(json.dumps(r) + "\\n")"""

content = content.replace(old_export_logic, new_export_logic)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
