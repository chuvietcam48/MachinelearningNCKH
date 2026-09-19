import re

file_path = "c:\\Users\\Admin\\OneDrive\\Máy tính\\MachineLearning-1\\MachinelearningNCKH\\src\\amazon_v5_rebuild\\07_phase2b1_gemini_calibration_v1_4.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace outputs
content = content.replace('_v1_3', '_v1_4')
content = content.replace('_v1_2', '_v1_4')
content = content.replace('Report v1.3', 'Report v1.4')
content = content.replace('v1.3 (Dry Run', 'v1.4 (Dry Run')

# Replace repair budget
content = content.replace('repair_budget = 5', 'repair_budget = 3')

# Replace call_gemini to inject the NEW edge cases
old_call_gemini = content[content.find("def call_gemini(model, prompt, is_dry_run=False, b_df=None, is_repair=False):") : content.find("# Real call")]

new_call_gemini = """def call_gemini(model, prompt, is_dry_run=False, b_df=None, is_repair=False):
    import json, time
    start_time = time.time()
    
    if is_dry_run:
        batch_name = prompt.split("Batch ID: ")[-1].split("\\n")[0].strip()
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
        
    """

content = content.replace(old_call_gemini, new_call_gemini)

# Add contract tests function
contract_tests = """
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

"""

content = content.replace('def run_phase2b1(is_dry_run=False):', contract_tests + '\ndef run_phase2b1(is_dry_run=False):')

run_tests_injection = """    if is_dry_run:
        run_contract_tests(schema_text, prompt_template)"""

content = content.replace('validator = OutputValidator(base_dir / "taxonomy_v1_1.md")', 'validator = OutputValidator(base_dir / "taxonomy_v1_1.md")\n' + run_tests_injection)

# Add tracking for schema deviation correctly
prov_old = "'schema_deviation_warning': 'top_level_list' in shapes_seen,"
prov_new = "'schema_deviation_warning': 'schema_deviation_top_level_list' if 'top_level_list' in shapes_seen else False,"
content = content.replace(prov_old, prov_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
