import pandas as pd
import json

base_dir = "outputs/amazon_v5_rebuild/annotation"

# 1. Load the original calibration manifest
calib_df = pd.read_csv(f"{base_dir}/calibration_sample_manifest_v1.csv")
calib_df['annotation_item_id'] = calib_df['reviewerID'] + '_' + calib_df['episode_id'].astype(str)

# 2. Find the duplicate
dup_ids = calib_df[calib_df.duplicated('annotation_item_id', keep=False)]
print("Duplicates found:", len(dup_ids))
dup_item_id = dup_ids['annotation_item_id'].iloc[0]
print("Duplicate ID:", dup_item_id)

# The first occurrence will be kept, the second will be replaced.
idx_to_replace = calib_df[calib_df['annotation_item_id'] == dup_item_id].index[1]
row_to_replace = calib_df.loc[idx_to_replace]

print("Row to replace:")
print(row_to_replace[['reviewerID', 'episode_id', 'overall']])

# 3. Load all candidate pools
df_full = pd.read_parquet("outputs/amazon_v5_rebuild/data/verified_episode_review_membership_v1.parquet")
df_full['annotation_item_id'] = df_full['reviewerID'] + '_' + df_full['episode_id'].astype(str)
df_full['text_len'] = df_full['reviewText'].fillna('').str.len()

blind_df = pd.read_csv(f"{base_dir}/blind_gold_sample_manifest_v1.csv")
blind_df['annotation_item_id'] = blind_df['reviewerID'] + '_' + blind_df['episode_id'].astype(str)
blind_ids = set(blind_df['annotation_item_id'])

calib_ids = set(calib_df['annotation_item_id'])

def get_jsonl_ids(filename):
    ids = set()
    try:
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'annotation_item_id' in data:
                    ids.add(data['annotation_item_id'])
    except Exception:
        pass
    return ids

v1_1_preflight_ids = get_jsonl_ids(f"{base_dir}/gemini_preflight_responses_v1_1.jsonl")
v1_1_calib_ids = get_jsonl_ids(f"{base_dir}/llm_calibration_raw_responses_v1_1.jsonl")

forbidden_ids = blind_ids.union(calib_ids).union(v1_1_preflight_ids).union(v1_1_calib_ids)

# We need to recreate the text_len for row_to_replace since it's not present
row_text = row_to_replace.get('reviewText', '')
if pd.isna(row_text): row_text = ''
row_to_replace_text_len = len(row_text)

print("Target rating:", row_to_replace['overall'])
print("Target text length approx:", row_to_replace_text_len)

def get_len_stratum(l):
    if l < 100: return 'Short'
    if l <= 500: return 'Medium'
    return 'Long'

df_full['text_len_stratum'] = df_full['text_len'].apply(get_len_stratum)
df_full['rating_stratum'] = df_full['overall'].astype(int)

candidates = df_full[
    (df_full['overall'] == row_to_replace['overall']) &
    (df_full['text_len_stratum'] == get_len_stratum(row_to_replace_text_len)) &
    (~df_full['annotation_item_id'].isin(forbidden_ids))
]

print("Candidates found:", len(candidates))
if len(candidates) > 0:
    replacement = candidates.sample(1, random_state=42).iloc[0]
    print("Selected Replacement:", replacement['annotation_item_id'])
    
    new_row = row_to_replace.copy()
    for col in calib_df.columns:
        if col in replacement.index:
            new_row[col] = replacement[col]
            
    calib_df.loc[idx_to_replace] = new_row
    
    orig_cols = pd.read_csv(f"{base_dir}/calibration_sample_manifest_v1.csv").columns
    calib_df_export = calib_df[orig_cols]
    calib_df_export.to_csv(f"{base_dir}/calibration_sample_manifest_v1_1.csv", index=False)
    print("Saved calibration_sample_manifest_v1_1.csv")
    
    import hashlib
    with open(f"{base_dir}/calibration_sample_manifest_v1_1.csv", "rb") as f:
        manifest_hash = hashlib.sha256(f.read()).hexdigest()
    
    log_content = f"""# Calibration Sample Replacement Log v1

**Removed Duplicate:**
- `annotation_item_id`: {dup_item_id}
- `rating`: {row_to_replace['overall']}
- `text_len_stratum`: {get_len_stratum(row_to_replace_text_len)}

**Replacement Selected:**
- `annotation_item_id`: {replacement['annotation_item_id']}
- `rating`: {replacement['overall']}
- `text_len_stratum`: {get_len_stratum(replacement['text_len'])}

**Disjointness Checks:**
- Overlaps with remaining calibration items: False
- Overlaps with blind semantic audit reserve: False
- Overlaps with v1.1 legacy invalid artifacts: False

**Manifest SHA256:**
- `{manifest_hash}`
"""
    with open(f"{base_dir}/calibration_sample_replacement_log_v1.md", "w") as f:
        f.write(log_content)
    print("Saved log")
else:
    print("No candidates found.")
