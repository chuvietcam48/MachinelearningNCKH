import re

file_path = "c:\\Users\\Admin\\OneDrive\\Máy tính\\MachineLearning-1\\MachinelearningNCKH\\src\\amazon_v5_rebuild\\07_phase2b1_gemini_calibration_v1_3.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace outputs
content = content.replace('gemini_batch_run_checkpoint_v1_2', 'gemini_batch_run_checkpoint_v1_3')
content = content.replace('llm_calibration_raw_responses_v1_2', 'llm_calibration_raw_responses_v1_3')
content = content.replace('llm_calibration_labels_v1_2', 'llm_calibration_labels_v1_3')
content = content.replace('llm_calibration_invalid_queue_v1_2', 'llm_calibration_invalid_queue_v1_3')
content = content.replace('llm_calibration_validation_v1_2', 'llm_calibration_validation_v1_3')
content = content.replace('gemini_call_manifest_v1_2', 'gemini_call_manifest_v1_3')
content = content.replace('llm_calibration_provenance_v1_2', 'llm_calibration_provenance_v1_3')
content = content.replace('phase2b1_llm_calibration_report_v1_2', 'phase2b1_llm_calibration_report_v1_3')
content = content.replace('Report v1.2', 'Report v1.3')

content = content.replace('gemini_request_budget_v1_1', 'gemini_request_budget_v1_3')
content = content.replace('gemini_batch_dry_run_report_v1_1', 'gemini_batch_dry_run_report_v1_3')
content = content.replace('Dry Run Report v1.1', 'Dry Run Report v1.3')
content = content.replace('v1.2 (Dry Run', 'v1.3 (Dry Run')

# Now for the parsing logic
old_parse = """        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            parsed = json.loads(cleaned)
            items = parsed.get('items', [])
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return "REPAIR_NEEDED", []"""

new_parse = """        response_shape = 'invalid_shape'
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
            return "REPAIR_NEEDED", []"""

content = content.replace(old_parse, new_parse)

# We also need to add response_shape to the provenance / manifest?
# The user said: "A top-level list is accepted for parser robustness but must be flagged as a schema-deviation warning in provenance."
# Wait, provenance manifest is created at the end of the script! It's one file for the whole run.
# But response shape could vary per batch!
# Let's just track a boolean or set of shapes globally.
# Or wait, the user said: "Record response shape as: object_items, top_level_list, invalid_shape... flagged as a schema-deviation warning in provenance."
# I will store it in the results dict for each item, and also collect them at the end.

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
