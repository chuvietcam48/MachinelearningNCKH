import re

file_path = "c:\\Users\\Admin\\OneDrive\\Máy tính\\MachineLearning-1\\MachinelearningNCKH\\src\\amazon_v5_rebuild\\07_phase2b1_gemini_calibration_v1_3.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace call_gemini to inject the new edge cases
old_call_gemini = content[content.find("def call_gemini(model, prompt, is_dry_run=False, b_df=None, is_repair=False):") : content.find("# Real call")]

new_call_gemini = """def call_gemini(model, prompt, is_dry_run=False, b_df=None, is_repair=False):
    start_time = time.time()
    
    if is_dry_run:
        batch_name = prompt.split("Batch ID: ")[-1].split("\\n")[0].strip()
        items = []
        
        # Determine fixture based on batch_name
        
        # Test malformed JSON
        if not is_repair and batch_name == "batch_2_calib":
            return "{bad_json", 0.5, "SUCCESS"
            
        for i, row in b_df.iterrows():
            iid = row['annotation_item_id']
            text = str(row.get('reviewText', ''))
            
            # Missing annotation ID
            if not is_repair and batch_name == "batch_0_preflight" and iid == b_df.iloc[1]['annotation_item_id']:
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
                if len(b_df) > 2 and iid == b_df.iloc[2]['annotation_item_id']:
                    item["Aspect1"] = "Invalid_Taxonomy_Value"
                if len(b_df) > 3 and iid == b_df.iloc[3]['annotation_item_id']:
                    item["Evidence1"] = "This exact string will never be in the text 123456789"
                if len(b_df) > 4 and iid == b_df.iloc[4]['annotation_item_id']:
                    if " " in text and text.count(" ") > 5:
                        item["Evidence1"] = " " 
                    elif len(text) > 2:
                        item["Evidence1"] = text[0]
                        
            # Duplicate ID
            if not is_repair and batch_name == "batch_1_calib":
                if len(b_df) > 1 and iid == b_df.iloc[1]['annotation_item_id']:
                    item["annotation_item_id"] = b_df.iloc[0]['annotation_item_id']
                    
            if is_repair:
                item["Evidence1"] = text[:15] if len(text) > 15 else text
                
            items.append(item)
            
        # Test top-level list
        if is_repair and batch_name == "batch_0_preflight_repair":
            return json.dumps(items), 0.5, "SUCCESS"
            
        return json.dumps({"items": items}), 0.5, "SUCCESS"
        
    """

content = content.replace(old_call_gemini, new_call_gemini)

# Add response_shape tracking to results
old_res_append = """            res = {
                'annotation_item_id': iid,
                'raw_response_snippet': json.dumps(item),
                'parsed_json': item,
                'is_valid': len(errors) == 0,
                'errors': errors,
                'batch_id': batch_id,
                'is_repaired': is_repair
            }"""

new_res_append = """            res = {
                'annotation_item_id': iid,
                'raw_response_snippet': json.dumps(item),
                'parsed_json': item,
                'is_valid': len(errors) == 0,
                'errors': errors,
                'batch_id': batch_id,
                'is_repaired': is_repair,
                'response_shape': response_shape
            }"""

content = content.replace(old_res_append, new_res_append)

old_missing_append = """            results.append({
                'annotation_item_id': m,
                'raw_response_snippet': "",
                'parsed_json': None,
                'is_valid': False,
                'errors': ['Missing from LLM output'],
                'batch_id': batch_id,
                'is_repaired': is_repair
            })"""

new_missing_append = """            results.append({
                'annotation_item_id': m,
                'raw_response_snippet': "",
                'parsed_json': None,
                'is_valid': False,
                'errors': ['Missing from LLM output'],
                'batch_id': batch_id,
                'is_repaired': is_repair,
                'response_shape': response_shape
            })"""
content = content.replace(old_missing_append, new_missing_append)

# Also update the provenance manifest to add any warning if we got top_level_list
# We will do this at the end of the script before writing provenance
provenance_patch_old = """        manifest = {
            'provider': 'google',
            'model': "gemini-2.5-flash","""

provenance_patch_new = """        shapes_seen = list(set([r.get('response_shape', 'unknown') for r in all_res]))
        manifest = {
            'provider': 'google',
            'model': "gemini-2.5-flash",
            'shapes_seen': shapes_seen,
            'schema_deviation_warning': 'top_level_list' in shapes_seen,"""

content = content.replace(provenance_patch_old, provenance_patch_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
